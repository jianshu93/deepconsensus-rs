use anyhow::{Result, ensure};
use clap::Parser;
use deepconsensus_rs::{
    alignment::{alignment_loss, softmax},
    checkpoint,
    dataset::DatasetReader,
    model::{self, ExecutionOptions},
    quality::Calibration,
    training::edit_distance,
};
use rayon::prelude::*;
use serde_json::json;
use std::{collections::HashSet, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    data: PathBuf,
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "reference,embedding,qkv,attention,flatten,combined"
    )]
    variants: Vec<String>,
    #[arg(long)]
    dump_logits: Option<PathBuf>,
    #[arg(long)]
    reference_logits: Option<PathBuf>,
}

fn options(name: &str) -> Result<ExecutionOptions> {
    let mut o = ExecutionOptions::REFERENCE;
    match name {
        "reference" => (),
        "embedding" => o.folded_embeddings = true,
        "qkv" => o.fused_qkv = true,
        "attention" => o.fused_attention = true,
        "flatten" => o.flattened_linears = true,
        "combined" => {
            o.folded_embeddings = true;
            o.fused_qkv = true;
            o.fused_attention = true;
            o.flattened_linears = true;
        }
        "selected" => o = ExecutionOptions::default(),
        "ort-reference" | "ort-optimized" => (),
        _ => anyhow::bail!("unknown variant {name}"),
    }
    Ok(o)
}

type Decoded = (Vec<u8>, Vec<u8>, Vec<u8>);

fn decode(logits: &[f32], calibration: &Calibration) -> Result<Decoded> {
    let probabilities = softmax(logits)?;
    let mut calls = Vec::new();
    let mut sequence = Vec::new();
    let mut qualities = Vec::new();
    for p in probabilities.as_chunks::<5>().0 {
        let (base, &probability) = p
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
            .unwrap();
        calls.push(base as u8);
        if base != 0 {
            sequence.push(base as u8);
            qualities.push(calibration.apply(-10.0 * (1.0 - probability).max(1e-300).log10(), 93));
        }
    }
    Ok((calls, sequence, qualities))
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.batch_size > 0 && a.repeats > 0 && a.threads > 0,
        "invalid dimensions"
    );
    ensure!(
        a.reference_logits.is_some() || a.variants.first().is_some_and(|v| v == "reference"),
        "reference must be first or --reference-logits provided"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()?;
    let device = model::device("cpu")?;
    let before = Instant::now();
    let model = checkpoint::load(&a.model, &device)?;
    let setup_seconds = before.elapsed().as_secs_f64();
    let c = &model.config;
    let before = Instant::now();
    let mut reader = DatasetReader::new(&a.data, c)?;
    let mut windows = Vec::new();
    let mut unique = HashSet::new();
    while let Some(w) = reader.next_window()? {
        ensure!(
            unique.insert((w.name.clone(), w.window_pos)),
            "duplicate window"
        );
        windows.push(w);
    }
    ensure!(!windows.is_empty(), "empty dataset");
    let read_seconds = before.elapsed().as_secs_f64();
    let calibration = Calibration::parse(&c.dc_calibration)?;
    let mut reference = Vec::new();
    let mut reference_decoded = Vec::new();
    if let Some(path) = &a.reference_logits {
        let raw = std::fs::read(path)?;
        ensure!(
            raw.len() == windows.len() * c.max_length * 5 * 4,
            "reference length mismatch"
        );
        reference = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        reference_decoded = reference
            .par_chunks_exact(c.max_length * 5)
            .map(|x| decode(x, &calibration))
            .collect::<Result<_>>()?;
    }
    let mut variants = serde_json::Map::new();
    let mut ort_reference = Vec::new();
    let mut ort_decoded: Vec<Decoded> = Vec::new();
    for name in &a.variants {
        let options = options(name)?;
        #[cfg(feature = "onnx")]
        let mut ort = if name.starts_with("ort-") {
            let path = a.output.with_extension(format!("{name}.onnx"));
            deepconsensus_rs::onnx::export_with_options(&model, &path, name == "ort-optimized")?;
            Some(deepconsensus_rs::onnx::OrtEngine::new(
                &path, c, "cpu", a.threads,
            )?)
        } else {
            None
        };
        #[cfg(not(feature = "onnx"))]
        ensure!(!name.starts_with("ort-"), "ORT requires --features onnx");
        let warm_windows = &windows[..a.batch_size.min(windows.len())];
        #[cfg(feature = "onnx")]
        if let Some(engine) = &mut ort {
            let _ = engine.predict(warm_windows)?;
        }
        if !name.starts_with("ort-") {
            let warm =
                model.forward_with_options(model::batch(warm_windows, c, &device)?, false, options);
            let _ = warm.to_data();
        }
        let mut samples = Vec::new();
        let mut values = Vec::new();
        for _ in 0..a.repeats {
            values.clear();
            let timer = Instant::now();
            let mut encoding = 0.0;
            let mut transformer = 0.0;
            for chunk in windows.chunks(a.batch_size) {
                #[cfg(feature = "onnx")]
                if let Some(engine) = &mut ort {
                    let before = Instant::now();
                    values.extend(engine.predict(chunk)?);
                    transformer += before.elapsed().as_secs_f64();
                    continue;
                }
                let rows = model::batch(chunk, c, &device)?;
                let before = Instant::now();
                let x = model.encode_with_options(rows, false, options);
                encoding += before.elapsed().as_secs_f64();
                let before = Instant::now();
                let y = model
                    .forward_encoded_with_options(x, false, options)
                    .to_data()
                    .try_to_vec::<f32>()?;
                transformer += before.elapsed().as_secs_f64();
                values.extend(y);
            }
            let seconds = timer.elapsed().as_secs_f64();
            samples.push(json!({"seconds":seconds,"encoding_seconds":encoding,"transformer_seconds":transformer}));
            eprintln!(
                "variant={name} windows={} seconds={seconds:.6} encoding={encoding:.6} transformer={transformer:.6}",
                windows.len()
            );
        }
        let decoded = values
            .par_chunks_exact(c.max_length * 5)
            .map(|x| decode(x, &calibration))
            .collect::<Result<Vec<_>>>()?;
        let versus_original_onnx = if name == "ort-optimized" && !ort_reference.is_empty() {
            json!({
                "max_absolute_logit_difference":values.iter().zip(&ort_reference).map(|(x,y):(&f32,&f32)| (x-y).abs()).fold(0.0f32,f32::max),
                "changed_logit_values":values.iter().zip(&ort_reference).filter(|(x,y)| x != y).count(),
                "changed_sequences":decoded.iter().zip(&ort_decoded).filter(|(x,y)| x.1 != y.1).count(),
                "changed_quality_windows":decoded.iter().zip(&ort_decoded).filter(|(x,y)| x.2 != y.2).count(),
            })
        } else {
            serde_json::Value::Null
        };
        if name == "ort-reference" {
            ort_reference.clone_from(&values);
            ort_decoded.clone_from(&decoded);
        }
        if name == "reference" {
            reference.clone_from(&values);
            reference_decoded.clone_from(&decoded);
            if let Some(path) = &a.dump_logits {
                let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(path, bytes)?;
            }
        }
        let max_logit_error = reference
            .iter()
            .zip(&values)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let changed_columns: usize = reference_decoded
            .iter()
            .zip(&decoded)
            .map(|(x, y)| x.0.iter().zip(&y.0).filter(|(x, y)| x != y).count())
            .sum();
        let changed_sequences = reference_decoded
            .iter()
            .zip(&decoded)
            .filter(|(x, y)| x.1 != y.1)
            .count();
        let changed_quality_windows = reference_decoded
            .iter()
            .zip(&decoded)
            .filter(|(x, y)| x.2 != y.2)
            .count();
        let metrics = windows
            .par_iter()
            .zip(values.par_chunks_exact(c.max_length * 5))
            .zip(decoded.par_iter())
            .map(|((w, x), pred)| -> Result<(f64, usize, usize)> {
                let label = w
                    .label
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("label missing"))?;
                let truth: Vec<u8> = label.iter().copied().filter(|&x| x != 0).collect();
                Ok((
                    alignment_loss(x, &truth, c)?.loss,
                    edit_distance(&truth, &pred.1),
                    truth.len(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let loss = metrics.iter().map(|x| x.0).sum::<f64>() / windows.len() as f64;
        let edits: usize = metrics.iter().map(|x| x.1).sum();
        let truth_bases: usize = metrics.iter().map(|x| x.2).sum();
        if let Some(existing) = variants.get_mut(name) {
            existing["samples"].as_array_mut().unwrap().extend(samples);
        } else {
            variants.insert(name.clone(),json!({"samples":samples,"max_absolute_logit_difference":max_logit_error,
            "changed_argmax_columns":changed_columns,"changed_sequences":changed_sequences,"changed_quality_windows":changed_quality_windows,
            "mean_alignment_loss":loss,"edit_errors":edits,"truth_bases":truth_bases,
            "versus_original_onnx":versus_original_onnx}));
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec_pretty(
            &json!({"windows":windows.len(),"unique_windows":unique.len(),
        "batch_size":a.batch_size,"threads":a.threads,"setup_seconds":setup_seconds,"dataset_read_seconds":read_seconds,
        "timed_scope":"CPU batch construction, encoding, transformer, logit collection; warmup, TFRecord reading, model loading, loss and decoding excluded",
        "variants":variants}),
        )?,
    )?;
    Ok(())
}
