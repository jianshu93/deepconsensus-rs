use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use deepconsensus_rs::onnx;
use deepconsensus_rs::{checkpoint, cli::config, model};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Optional DeepConsensus checkpoint preparation and conversion tools"
)]
#[command(
    after_help = "These CPU-only utilities need neither Python/TensorFlow nor an ONNX Runtime library. Tensor names are diagnostic for checking imports or unsupported checkpoints, not needed for ordinary training/inference. No model is downloaded implicitly. Paths are relative to the current directory unless absolute."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a seeded random native checkpoint; not suitable for correction before training.
    ///
    /// Optional: train already initializes weights when --resume is omitted. Writes
    /// params.json and model.burnpack; creates output and replaces those files if
    /// present. No optimizer or training state is created.
    Init {
        /// Parameters JSON; omitted uses v1.2-shaped defaults, not pretrained weights.
        #[arg(long, value_name = "JSON")]
        config: Option<PathBuf>,
        /// Required native checkpoint output directory; no default.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
        /// CPU model initialization seed.
        #[arg(long, default_value_t = 1, value_name = "N")]
        seed: u64,
    },
    /// Convert a supported TensorFlow TensorBundle checkpoint into native Burn weights in Rust.
    ///
    /// Requires the prefix's .index and .data-00000-of-00001 files and matching
    /// original parameters. Supports the published v1.2 learned-values transformer
    /// with little-endian F32 tensors, not arbitrary TensorFlow models. Validates
    /// names/shapes/checksums; replaces native model/parameters at output.
    ImportTf {
        /// Required checkpoint prefix, e.g. original/checkpoint; do not include .index.
        #[arg(long, value_name = "PREFIX")]
        checkpoint: PathBuf,
        /// Required original params.json; no architecture is guessed.
        #[arg(long, value_name = "JSON")]
        params: PathBuf,
        /// Required native output directory; created as needed; no default.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
    },
    /// List TensorFlow tensor names for import diagnostics; no model is modified.
    InspectTf {
        /// Required TensorBundle checkpoint prefix; do not include .index.
        #[arg(long, value_name = "PREFIX")]
        checkpoint: PathBuf,
    },
    /// Export existing native weights to FP32 ONNX without retraining or Python.
    ///
    /// Writes a dynamic-batch, fixed-window DeepConsensus graph and NAME.params.json
    /// sidecar for feature layout/calibration. Parents are created, existing graph
    /// and sidecar replaced. Training can instead use train --export-onnx [PATH].
    ExportOnnx {
        /// Required native directory with model.burnpack and params.json; no default.
        #[arg(long, value_name = "DIR")]
        model: PathBuf,
        /// ONNX path; omitted writes MODEL/model.onnx plus MODEL/model.params.json.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
        /// Opt into experimental QKV/Gemm graph rewrites; disabled by default.
        #[arg(long)]
        experimental_fusions: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Init {
            config: params,
            output,
            seed,
        } => {
            writable_weights_directory(&output)?;
            let c = config(params)?;
            let device = model::device("cpu")?;
            device.seed(seed);
            checkpoint::save(&model::DeepConsensus::new(&c, &device)?, &output)?;
        }
        Command::ImportTf {
            checkpoint: prefix,
            params,
            output,
        } => {
            writable_weights_directory(&output)?;
            let c = deepconsensus_rs::config::ModelConfig::load(params)?;
            let m = checkpoint::import_tensorflow(&prefix, &c, &model::device("cpu")?)?;
            checkpoint::save(&m, &output)?;
            eprintln!("Imported {} encoder layers", c.num_hidden_layers);
        }
        Command::InspectTf { checkpoint: prefix } => {
            for name in checkpoint::TensorflowCheckpoint::open(&prefix)?.names() {
                println!("{name}");
            }
        }
        Command::ExportOnnx {
            model: directory,
            output,
            experimental_fusions,
        } => {
            let path = output.unwrap_or_else(|| directory.join("model.onnx"));
            onnx::validate_export_path(&path)?;
            onnx::export_with_options(
                &checkpoint::load(&directory, &model::device("cpu")?)?,
                &path,
                experimental_fusions,
            )?;
            eprintln!(
                "Exported {} and {}",
                path.display(),
                onnx::config_path(&path).display()
            );
        }
    }
    Ok(())
}

fn writable_weights_directory(path: &std::path::Path) -> Result<()> {
    ensure!(
        !path.join("optimizer.burnpack").exists() && !path.join("training-state.json").exists(),
        "refusing to replace weights in a directory containing optimizer/training state; choose a fresh --output directory"
    );
    Ok(())
}
