use anyhow::{Result, ensure};
use clap::Parser;
use deepconsensus_rs::onnx;
use deepconsensus_rs::{
    alignment, checkpoint,
    cli::{Cli, Command, config},
    config::ModelConfig,
    inference, model, preprocess,
    training::{self, TrainOptions},
};
use std::{path::PathBuf, time::Instant};

fn print_report(value: &serde_json::Value, path: Option<PathBuf>) -> Result<()> {
    let report = serde_json::to_string_pretty(value)?;
    if let Some(path) = path {
        std::fs::write(path, &report)?;
    }
    println!("{report}");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut pool = rayon::ThreadPoolBuilder::new();
    if cli.cpu_threads > 0 {
        pool = pool.num_threads(cli.cpu_threads);
    }
    pool.build_global()?;
    match cli.command {
        Command::Preprocess {
            input,
            truth,
            config: params,
            output,
        } => {
            let c = config(params)?;
            let count = training::prepare_dataset(
                &mut input.open(truth.as_deref())?,
                &output,
                &c,
                input.smart_windows,
                input.limit,
            )?;
            eprintln!("wrote {count} windows");
        }
        Command::Inference {
            input,
            model,
            output,
            config,
            device,
            provider,
            ort_threads,
            batch_size,
            block_molecules,
            skip_above,
            min_quality,
            min_length,
            max_quality,
            dc_calibration,
            ccs_calibration,
            report,
        } => {
            let options = inference::Options {
                model,
                config,
                device,
                provider: provider.map(|p| p.name().to_owned()),
                ort_threads,
                batch_size,
                block_molecules,
                smart_windows: input.smart_windows,
                skip_above,
                min_quality,
                min_length,
                max_quality,
                dc_calibration,
                ccs_calibration,
                limit: input.limit,
            };
            inference::model_config(&options)?;
            let summary = inference::run(&mut input.open(None)?, &output, &options)?;
            print_report(&serde_json::to_value(summary)?, report)?;
        }
        Command::Train {
            data,
            config: params,
            output,
            resume,
            teacher,
            epochs,
            batch_size,
            learning_rate,
            end_learning_rate,
            warmup_steps,
            weight_decay,
            seed,
            shuffle_buffer,
            max_examples,
            device,
            band_width,
            distillation_weight,
            export_onnx,
        } => {
            let export_path = export_onnx.map(|p| p.unwrap_or_else(|| output.join("model.onnx")));
            if let Some(path) = &export_path {
                onnx::validate_export_path(path)?;
            }
            let mut c = if params.is_none()
                && let Some(directory) = resume.as_ref()
            {
                ModelConfig::load(directory.join("params.json"))?
            } else {
                config(params)?
            };
            if let Some(width) = band_width {
                c.band_width = Some(width);
            }
            training::train(
                &c,
                &TrainOptions {
                    datasets: data,
                    output: output.clone(),
                    resume,
                    teacher,
                    epochs,
                    batch_size,
                    learning_rate,
                    end_learning_rate,
                    warmup_steps,
                    weight_decay,
                    seed,
                    shuffle_buffer,
                    max_examples,
                    device,
                    distillation_weight,
                },
            )?;
            if let Some(path) = export_path {
                // Export persisted final weights on CPU without loading an ORT runtime.
                onnx::export(&checkpoint::load(&output, &model::device("cpu")?)?, &path)?;
                eprintln!(
                    "Exported {} and {}",
                    path.display(),
                    onnx::config_path(&path).display()
                );
            }
        }
        Command::Evaluate {
            model: directory,
            data,
            batch_size,
            device,
            output,
        } => {
            let device = model::device(&device)?;
            let m = checkpoint::load(&directory, &device)?;
            let result = training::evaluate(&m, &data, batch_size, &device)?;
            print_report(&result, output)?;
        }
        Command::Benchmark {
            input,
            config: params,
            repetitions,
            band_width,
            output,
        } => {
            ensure!(repetitions > 0, "repetitions must be positive");
            let c = config(params)?;
            let mut reader = input.open(None)?;
            let mut molecules = Vec::new();
            while input.limit.is_none_or(|n| molecules.len() < n) {
                if let Some(m) = reader.next_molecule()? {
                    molecules.push(m);
                } else {
                    break;
                }
            }
            ensure!(!molecules.is_empty(), "no benchmark molecules");
            let mut samples = Vec::new();
            let mut windows = Vec::new();
            for _ in 0..repetitions {
                let t = Instant::now();
                windows = molecules
                    .iter()
                    .map(|m| preprocess::windows(m, &c, input.smart_windows))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                samples.push(t.elapsed().as_secs_f64());
            }
            let w = windows
                .iter()
                .find(|w| !w.overflow)
                .ok_or_else(|| anyhow::anyhow!("no model window"))?;
            let label: Vec<u8> = w.ccs.iter().copied().filter(|&x| x != 0).collect();
            let logits: Vec<f32> = (0..c.max_length * 5)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) * 0.13)
                .collect();
            let mut dense = c.clone();
            dense.band_width = None;
            let mut band = c.clone();
            band.band_width = Some(band_width);
            let mut dense_samples = Vec::new();
            let mut band_samples = Vec::new();
            let dense_result = alignment::alignment_loss(&logits, &label, &dense)?;
            let band_result = alignment::alignment_loss(&logits, &label, &band)?;
            for _ in 0..repetitions {
                let t = Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(alignment::alignment_loss(&logits, &label, &dense)?);
                }
                dense_samples.push(t.elapsed().as_secs_f64() / 100.0);
                let t = Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(alignment::alignment_loss(&logits, &label, &band)?);
                }
                band_samples.push(t.elapsed().as_secs_f64() / 100.0);
            }
            print_report(
                &serde_json::json!({"molecules":molecules.len(),"subreads":molecules.iter().map(|m|m.subreads.len()).sum::<usize>(),"windows":windows.len(),"preprocess_seconds":samples,"dense_loss_seconds":dense_samples,"banded_loss_seconds":band_samples,"dense_cells":dense_result.active_cells,"banded_cells":band_result.active_cells,"band_width":band_width,"dense_loss":dense_result.loss,"banded_loss":band_result.loss,"note":"Loss timing uses deterministic synthetic logits and the draft CCS as a proxy label. A restricted band changes the objective; this is not a truth accuracy benchmark."}),
                Some(output),
            )?;
        }
    }
    Ok(())
}
