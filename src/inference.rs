use crate::{
    alignment::softmax,
    bam::MoleculeReader,
    checkpoint,
    config::ModelConfig,
    dataset,
    model::{self, DeepConsensus},
    preprocess::{self, Window},
    quality::{Calibration, avg_phred},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

pub enum Engine {
    Burn {
        model: Box<DeepConsensus>,
        device: burn::tensor::Device,
    },
    #[cfg(feature = "onnx")]
    Ort(Box<crate::onnx::OrtEngine>),
}
impl Engine {
    pub fn predict(&mut self, windows: &[Window], config: &ModelConfig) -> Result<Vec<f32>> {
        match self {
            Self::Burn { model, device } => Ok(model
                .forward(model::batch(windows, config, device)?, false)
                .to_data()
                .try_to_vec::<f32>()?),
            #[cfg(feature = "onnx")]
            Self::Ort(engine) => engine.predict(windows),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Options {
    pub model: PathBuf,
    pub config: Option<PathBuf>,
    pub device: String,
    pub provider: Option<String>,
    pub ort_threads: usize,
    pub batch_size: usize,
    pub block_molecules: usize,
    pub smart_windows: bool,
    pub skip_above: f64,
    pub min_quality: f64,
    pub min_length: usize,
    pub max_quality: u8,
    pub dc_calibration: Option<String>,
    pub ccs_calibration: String,
    pub limit: Option<usize>,
}
#[derive(Default, Debug, Serialize)]
pub struct Summary {
    pub molecules: usize,
    pub written: usize,
    pub filtered: usize,
    pub model_windows: usize,
    pub skipped_windows: usize,
    pub preprocess_seconds: f64,
    pub model_seconds: f64,
    pub total_seconds: f64,
    pub output_bases: usize,
}

pub fn onnx_provider(device: &str, provider: Option<&str>) -> Result<String> {
    crate::cli::device_name(device).map_err(anyhow::Error::msg)?;
    let selected = provider.unwrap_or(match device {
        "cpu" => "cpu",
        "metal" => "coreml",
        _ => "cuda",
    });
    match (device, selected) {
        ("cpu", "cpu") => Ok("cpu".into()),
        ("cpu", "coreml") => Ok("coreml:cpu".into()),
        ("metal", "coreml") => Ok("coreml:gpu".into()),
        (device, "cuda" | "tensorrt") if device.starts_with("cuda:") => {
            Ok(format!("{selected}:{}", &device[5..]))
        }
        _ => anyhow::bail!(
            "ONNX provider {selected} conflicts with --device {device}; CPU uses cpu, CUDA/TensorRT use cuda:N, CoreML uses metal or cpu"
        ),
    }
}

pub fn model_config(o: &Options) -> Result<ModelConfig> {
    crate::cli::device_name(&o.device).map_err(anyhow::Error::msg)?;
    if o.model.is_dir() {
        ensure!(
            o.config.is_none(),
            "--config is only for ONNX; native checkpoints use their own params.json"
        );
        ensure!(o.provider.is_none(), "--provider is only for ONNX models");
        ensure!(
            o.model.join("model.burnpack").is_file(),
            "native checkpoint is missing model.burnpack: {}",
            o.model.display()
        );
        return ModelConfig::load(o.model.join("params.json"));
    }
    ensure!(
        o.model.is_file(),
        "model path does not exist: {}",
        o.model.display()
    );
    ensure!(
        o.model.extension().is_some_and(|e| e == "onnx"),
        "--model must be a native checkpoint directory or a .onnx file"
    );
    let provider = onnx_provider(&o.device, o.provider.as_deref())?;
    ensure!(
        cfg!(feature = "onnx"),
        "ONNX inference requires --features onnx, onnx-coreml/metal, or onnx-cuda/cuda/tensorrt"
    );
    if provider.starts_with("coreml:") {
        ensure!(cfg!(target_os = "macos"), "CoreML inference requires macOS");
        ensure!(
            cfg!(feature = "onnx-coreml"),
            "CoreML inference requires --features metal or onnx-coreml"
        );
    } else if provider != "cpu" {
        ensure!(
            cfg!(target_os = "linux"),
            "ONNX CUDA/TensorRT providers require Linux"
        );
        ensure!(
            cfg!(feature = "onnx-cuda"),
            "ONNX CUDA inference requires --features onnx-cuda"
        );
        if provider.starts_with("tensorrt:") {
            ensure!(
                cfg!(feature = "tensorrt"),
                "TensorRT inference requires --features tensorrt"
            );
        }
    }
    let sidecar = crate::onnx::config_path(&o.model);
    let path = o.config.as_deref().unwrap_or(&sidecar);
    ModelConfig::load(path).with_context(|| format!("ONNX model needs parameters at {}; for an older export supply --config /path/to/params.json", path.display()))
}

pub fn run(reader: &mut MoleculeReader, output: &Path, o: &Options) -> Result<Summary> {
    ensure!(
        o.batch_size > 0
            && o.block_molecules > 0
            && o.max_quality <= 93
            && o.ort_threads > 0
            && o.skip_above.is_finite()
            && o.skip_above >= 0.0
            && o.min_quality.is_finite()
            && o.min_quality >= 0.0,
        "invalid inference settings"
    );
    let c = model_config(o)?;
    let dc = Calibration::parse(o.dc_calibration.as_deref().unwrap_or(&c.dc_calibration))?;
    let ccs = Calibration::parse(&o.ccs_calibration)?;
    let mut engine = if o.model.is_file() {
        #[cfg(feature = "onnx")]
        {
            Engine::Ort(Box::new(crate::onnx::OrtEngine::new(
                &o.model,
                &c,
                &onnx_provider(&o.device, o.provider.as_deref())?,
                o.ort_threads,
            )?))
        }
        #[cfg(not(feature = "onnx"))]
        {
            anyhow::bail!("ONNX requires --features onnx");
        }
    } else {
        let device = model::device(&o.device)?;
        Engine::Burn {
            model: Box::new(checkpoint::load(&o.model, &device)?),
            device,
        }
    };
    let mut out = BufWriter::new(dataset::writer(output)?);
    let mut report = Summary::default();
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<Summary> {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        let config = c.clone();
        let producer = scope.spawn(move || {
            if let Err(error) = produce_blocks(reader, &config, o, &tx) {
                let _ = tx.send(Err(error));
            }
        });
        let result = (|| -> Result<Summary> {
            for block in rx.iter() {
                let block = block?;
                report.preprocess_seconds += block.seconds;
                let prepared = block.windows;
                let names = block.names;
                let preprocess = Instant::now();
                let mut predictions: Vec<Vec<(Vec<u8>, Vec<u8>)>> = prepared
                    .iter()
                    .map(|windows| vec![(Vec::new(), Vec::new()); windows.len()])
                    .collect();
                let mut pending = Vec::new();
                let mut locations = Vec::new();
                for (m, windows) in prepared.iter().enumerate() {
                    for (w, window) in windows.iter().enumerate() {
                        if window.overflow
                            || (o.skip_above > 0.0
                                && avg_phred(window.qualities.iter().copied()) > o.skip_above)
                        {
                            report.skipped_windows += 1;
                            let mut bases = Vec::new();
                            let mut qualities = Vec::new();
                            for (&base, &q) in window.ccs.iter().zip(&window.qualities) {
                                if base != 0 {
                                    bases.push(preprocess::VOCAB[base as usize]);
                                    qualities.push(ccs.apply(q.max(0) as f64, o.max_quality));
                                }
                            }
                            predictions[m][w] = (bases, qualities);
                        } else {
                            pending.push(window.clone());
                            locations.push((m, w));
                        }
                    }
                }
                report.preprocess_seconds += preprocess.elapsed().as_secs_f64();
                let model_timer = Instant::now();
                for (windows, locations) in pending
                    .chunks(o.batch_size)
                    .zip(locations.chunks(o.batch_size))
                {
                    let values = engine.predict(windows, &c)?;
                    ensure!(
                        values.len() == windows.len() * c.max_length * 5,
                        "prediction length mismatch"
                    );
                    ensure!(
                        o.skip_above.is_finite()
                            && o.skip_above >= 0.0
                            && o.min_quality.is_finite()
                            && o.min_quality >= 0.0,
                        "invalid quality threshold"
                    );
                    let decoded: Vec<_> = values
                        .par_chunks_exact(c.max_length * 5)
                        .zip(locations.par_iter())
                        .map(|(logits, &(m, w))| -> Result<_> {
                            let probabilities = softmax(logits)?;
                            let mut bases = Vec::new();
                            let mut qualities = Vec::new();
                            for probs in probabilities.as_chunks::<5>().0 {
                                let (base, &p) = probs
                                    .iter()
                                    .enumerate()
                                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                                    .unwrap();
                                if base == 0 {
                                    continue;
                                }
                                bases.push(preprocess::VOCAB[base]);
                                qualities.push(
                                    dc.apply(-10.0 * (1.0 - p).max(1e-300).log10(), o.max_quality),
                                );
                            }
                            Ok((m, w, bases, qualities))
                        })
                        .collect::<Result<_>>()?;
                    report.model_windows += decoded.len();
                    for (m, w, bases, qualities) in decoded {
                        predictions[m][w] = (bases, qualities);
                    }
                }
                report.model_seconds += model_timer.elapsed().as_secs_f64();
                for (name, predictions) in names.iter().zip(predictions) {
                    report.molecules += 1;
                    let mut bases = Vec::new();
                    let mut qualities = Vec::new();
                    for (b, q) in predictions {
                        bases.extend(b);
                        qualities.extend(q);
                    }
                    let average_quality = (avg_phred(qualities.iter().map(|&x| x as i16)) * 1e5)
                        .round_ties_even()
                        / 1e5;
                    if bases.is_empty()
                        || bases.len() < o.min_length
                        || average_quality < o.min_quality
                    {
                        report.filtered += 1;
                        continue;
                    }
                    ensure!(bases.len() == qualities.len(), "FASTQ length mismatch");
                    writeln!(out, "@{name}")?;
                    out.write_all(&bases)?;
                    out.write_all(b"\n+\n")?;
                    out.write_all(&qualities.iter().map(|&q| q + 33).collect::<Vec<_>>())?;
                    out.write_all(b"\n")?;
                    report.written += 1;
                    report.output_bases += bases.len();
                }
                eprintln!(
                    "molecules={} written={} model_windows={} skipped_windows={}",
                    report.molecules, report.written, report.model_windows, report.skipped_windows
                );
            }
            out.flush()?;
            out.into_inner().map_err(|e| e.into_error())?.finish()?;
            report.total_seconds = start.elapsed().as_secs_f64();
            Ok(report)
        })();
        drop(rx);
        producer
            .join()
            .map_err(|_| anyhow::anyhow!("preprocessing worker panicked"))?;
        result
    })
}

struct PreparedBlock {
    names: Vec<String>,
    windows: Vec<Vec<Window>>,
    seconds: f64,
}
fn produce_blocks(
    reader: &mut MoleculeReader,
    c: &ModelConfig,
    o: &Options,
    tx: &std::sync::mpsc::SyncSender<Result<PreparedBlock>>,
) -> Result<()> {
    let mut processed = 0;
    while o.limit.is_none_or(|n| processed < n) {
        let timer = Instant::now();
        let mut molecules = Vec::new();
        while molecules.len() < o.block_molecules
            && o.limit.is_none_or(|n| processed + molecules.len() < n)
        {
            if let Some(m) = reader.next_molecule()? {
                molecules.push(m);
            } else {
                break;
            }
        }
        if molecules.is_empty() {
            break;
        }
        processed += molecules.len();
        let windows = molecules
            .par_iter()
            .map(|m| preprocess::windows_with_gate(m, c, o.smart_windows, Some(o.skip_above)))
            .collect::<Result<_>>()?;
        let names = molecules.into_iter().map(|m| m.name).collect();
        let block = PreparedBlock {
            names,
            windows,
            seconds: timer.elapsed().as_secs_f64(),
        };
        if tx.send(Ok(block)).is_err() {
            return Ok(());
        }
    }
    Ok(())
}
