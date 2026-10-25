use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

use crate::{bam::MoleculeReader, config::ModelConfig, quality::Calibration};

#[derive(Parser)]
#[command(
    version,
    about = "DeepConsensus sequence correction and training on CPU, Linux CUDA, or macOS Metal"
)]
#[command(
    after_help = "No model is downloaded or selected implicitly. Paths are relative to the current directory unless absolute. Use deepconsensus-model for checkpoint initialization, TensorFlow import/inspection, and standalone ONNX export."
)]
pub struct Cli {
    /// Rayon CPU worker count; 0 uses Rayon's automatic hardware/environment default.
    #[arg(long, global = true, default_value_t = 0, value_name = "N")]
    pub cpu_threads: usize,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Args, Clone)]
pub struct BamInput {
    /// PacBio subreads aligned to their CCS molecules, grouped by CCS reference (required).
    ///
    /// Not a genomic-mapping BAM. Each subread requires pw, ip, and four-value sn tags.
    #[arg(long, value_name = "BAM")]
    pub subreads: PathBuf,
    /// Matching CCS BAM in the same molecule order, with real base qualities (required).
    #[arg(long, value_name = "BAM")]
    pub ccs: PathBuf,
    /// HTSlib decompression workers per BAM reader; 0 disables threaded decompression.
    ///
    /// Independent of --cpu-threads.
    #[arg(long, default_value_t = 4, value_name = "N")]
    pub threads: usize,
    /// Discard insertion runs longer than this many bases; 0 retains all insertions.
    #[arg(long, default_value_t = 5, value_name = "BASES")]
    pub ins_trim: usize,
    /// Stop after this many input molecules (not windows); omitted means all molecules.
    #[arg(long, value_parser = positive, value_name = "N")]
    pub limit: Option<usize>,
    /// Use CCS wl-tag window boundaries; disabled by default, requires a valid wl tag.
    #[arg(long)]
    pub smart_windows: bool,
}

impl BamInput {
    pub fn open(&self, truth: Option<&Path>) -> Result<MoleculeReader> {
        MoleculeReader::open(
            &self.subreads,
            &self.ccs,
            truth,
            self.threads,
            self.ins_trim,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Provider {
    /// ONNX Runtime CPU; requires --device cpu and the onnx Cargo feature.
    Cpu,
    /// ONNX Runtime CUDA on the --device cuda:N GPU; requires Linux and cuda/onnx-cuda.
    #[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
    Cuda,
    /// TensorRT with CUDA fallback on --device cuda:N; requires Linux and tensorrt.
    #[cfg(all(target_os = "linux", feature = "tensorrt"))]
    Tensorrt,
    /// macOS CoreML: --device metal allows CPU/GPU; --device cpu uses CoreML CPUOnly.
    #[cfg(all(target_os = "macos", feature = "onnx-coreml"))]
    Coreml,
}

impl Provider {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(all(target_os = "linux", feature = "onnx-cuda"))]
            Self::Cuda => "cuda",
            #[cfg(all(target_os = "linux", feature = "tensorrt"))]
            Self::Tensorrt => "tensorrt",
            #[cfg(all(target_os = "macos", feature = "onnx-coreml"))]
            Self::Coreml => "coreml",
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Generate JSONL or original-compatible TFRecord feature windows from aligned BAMs.
    ///
    /// Use .jsonl[.gz] for JSONL, or .tfrecord[s][.gz] for TFRecord. A .gz suffix
    /// enables gzip. Existing output is overwritten; its parent must exist. Without
    /// --truth the windows are unlabeled and cannot be used for supervised training.
    Preprocess {
        #[command(flatten)]
        input: BamInput,
        /// Indexed truth-to-CCS BAM for labels; omitted means no truth labels.
        #[arg(long, value_name = "BAM")]
        truth: Option<PathBuf>,
        /// Model parameters JSON; omitted uses built-in v1.2-shaped architecture defaults.
        ///
        /// Defaults: 100 columns, 20 passes, width 280, 5 layers, 2 heads, CCS qualities.
        /// No weights are loaded. Use the intended model's params.json for matching features.
        #[arg(long, value_name = "JSON")]
        config: Option<PathBuf>,
        /// Required output dataset path; no default. Parent must exist; existing file is replaced.
        #[arg(long, value_name = "PATH", value_parser = dataset_path)]
        output: PathBuf,
    },
    /// Correct aligned subreads/CCS into stitched molecule FASTQ.
    #[command(alias = "run")]
    #[command(
        long_about = "Correct aligned subreads/CCS into stitched molecule FASTQ using Burn or ONNX Runtime. --model is a native checkpoint directory (params.json + model.burnpack) or a DeepConsensus .onnx file. ONNX requires its sibling NAME.params.json or an explicit --config. Truth alignments are never accepted. Existing output is overwritten; parent must exist."
    )]
    Inference {
        #[command(flatten)]
        input: BamInput,
        /// Required native checkpoint directory or .onnx file; no implicit model or download.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,
        /// Required FASTQ path; .gz enables gzip. No default; existing file is replaced.
        #[arg(long, value_name = "FASTQ")]
        output: PathBuf,
        /// ONNX-only parameters JSON; omitted uses MODEL with extension .params.json.
        ///
        /// Required for older ONNX exports without that sidecar. Rejected for native models.
        #[arg(long, value_name = "JSON")]
        config: Option<PathBuf>,
        /// cpu, cuda:N (Linux), or metal (macOS); GPUs require the matching build feature.
        ///
        /// Native CUDA requires cuda; native Metal requires metal. For ONNX, cuda:N
        /// uses ORT CUDA and metal uses macOS CoreML CPUAndGPU. ORT needs a matching
        /// runtime library. CoreML can use CPU partitions; it is not a GPU-only guarantee.
        #[arg(long, default_value = "cpu", value_parser = device_name, value_name = "DEVICE")]
        device: String,
        /// ONNX-only provider override; omitted maps cpu/cuda:N/metal to CPU/CUDA/CoreML.
        ///
        /// Only compiled providers for this OS are listed. TensorRT requires cuda:N.
        /// CoreML allows metal (CPUAndGPU) or cpu (CPUOnly), with MLProgram format and
        /// dynamic batches. Native checkpoints reject this option. Registration failures
        /// are errors; unsupported CoreML/TensorRT graph partitions can still use CPU.
        #[arg(long, value_enum)]
        provider: Option<Provider>,
        /// ONNX Runtime intra-op worker count; ignored for native Burn inference.
        #[arg(long, default_value_t = 4, value_parser = positive, value_name = "N")]
        ort_threads: usize,
        /// Maximum feature windows per model batch (not molecules); must be positive.
        #[arg(long, default_value_t = 32, value_parser = positive, value_name = "N")]
        batch_size: usize,
        /// Molecules per bounded preprocessing block; larger values use more memory.
        #[arg(long, default_value_t = 16, value_parser = positive, value_name = "N")]
        block_molecules: usize,
        /// Preserve CCS windows above this probability-averaged Phred quality; 0 disables gating.
        ///
        /// Overflow windows are always preserved. Skipped windows are not model predictions.
        #[arg(long, default_value_t = 45.0, value_parser = nonnegative, value_name = "Q")]
        skip_above: f64,
        /// Minimum probability-averaged Phred quality of a stitched output read; 0 disables filtering.
        #[arg(long, default_value_t = 0.0, value_parser = nonnegative, value_name = "Q")]
        min_quality: f64,
        /// Minimum stitched output length in bases; empty reads are always filtered.
        #[arg(long, default_value_t = 1, value_parser = positive, value_name = "BASES")]
        min_length: usize,
        /// Maximum output Phred quality for FASTQ; allowed range 0..=93.
        #[arg(long, default_value_t = 93, value_parser = clap::value_parser!(u8).range(..=93), value_name = "Q")]
        max_quality: u8,
        /// Predicted-quality calibration: skip or threshold,slope,intercept; omitted uses model config.
        #[arg(long, value_parser = calibration, value_name = "CALIBRATION")]
        dc_calibration: Option<String>,
        /// Skipped-CCS quality calibration: skip or threshold,slope,intercept.
        #[arg(long, default_value = "skip", value_parser = calibration, value_name = "CALIBRATION")]
        ccs_calibration: String,
        /// Optional JSON timing/yield report path; omitted writes JSON to stdout only.
        #[arg(long, value_name = "JSON")]
        report: Option<PathBuf>,
    },
    /// Train or resume the Burn transformer on labeled windows.
    #[command(
        long_about = "Train or resume the Burn transformer on labeled windows; optionally export final FP32 ONNX for inference. Starts with seeded random weights unless --resume is supplied; no init command is needed. Saves params.json, model.burnpack, optimizer.burnpack, training state and options after each epoch. --epochs is the total target, not extra epochs. CPU uses Rayon analytic alignment gradients; CUDA/Metal use device wavefront autodiff."
    )]
    Train {
        /// Required labeled TFRecord or JSONL datasets (optional gzip); one or more paths.
        #[arg(long, num_args = 1.., required = true, value_name = "PATH", value_parser = dataset_path)]
        data: Vec<PathBuf>,
        /// Parameters JSON; omitted uses resume/params.json, otherwise built-in architecture defaults.
        ///
        /// Built-in defaults: length 100, passes 20, width 280, 5 layers, 2 heads,
        /// FFN 2048, attention radius 12, CCS qualities, ReZero, dense alignment loss.
        /// An explicit config must exactly match a resumed checkpoint.
        #[arg(long, value_name = "JSON")]
        config: Option<PathBuf>,
        /// Required native checkpoint output directory; created as needed, existing checkpoint files replaced.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
        /// Native checkpoint directory to resume/fine-tune; omitted initializes random weights.
        ///
        /// Restores optimizer/training state when present. A published weights-only
        /// directory starts at epoch/step zero. Output may equal this directory.
        #[arg(long, value_name = "DIR")]
        resume: Option<PathBuf>,
        /// Frozen native teacher directory for logit-MSE distillation; omitted disables distillation.
        #[arg(long, value_name = "DIR")]
        teacher: Option<PathBuf>,
        /// Total target epochs including resumed epochs; must exceed the saved epoch.
        #[arg(long, default_value_t = 1, value_parser = positive, value_name = "N")]
        epochs: usize,
        /// Training windows per update; must be positive.
        #[arg(long, default_value_t = 16, value_parser = positive, value_name = "N")]
        batch_size: usize,
        /// Peak learning rate for AdamW with cosine decay; must be finite and positive.
        #[arg(long, default_value_t = 0.001, value_parser = positive_float, value_name = "RATE")]
        learning_rate: f64,
        /// Final cosine learning rate; must be finite and nonnegative.
        #[arg(long, default_value_t = 0.00001, value_parser = nonnegative, value_name = "RATE")]
        end_learning_rate: f64,
        /// Linear warmup update count; 0 disables warmup.
        #[arg(long, default_value_t = 0, value_name = "N")]
        warmup_steps: usize,
        /// AdamW decay coefficient; 0 disables weight decay.
        #[arg(long, default_value_t = 0.0069868, value_parser = nonnegative_f32, value_name = "RATE")]
        weight_decay: f32,
        /// Model initialization and per-epoch shuffle seed; not a cross-device determinism guarantee.
        #[arg(long, default_value_t = 1, value_name = "N")]
        seed: u64,
        /// Streaming shuffle buffer in windows; 1 preserves dataset order, larger values use more memory.
        #[arg(long, default_value_t = 1024, value_parser = positive, value_name = "N")]
        shuffle_buffer: usize,
        /// Maximum windows across all datasets per epoch; omitted uses all windows.
        #[arg(long, value_parser = positive, value_name = "N")]
        max_examples: Option<usize>,
        /// cpu, cuda:N (Linux --features cuda), or metal (macOS --features metal).
        #[arg(long, default_value = "cpu", value_parser = device_name, value_name = "DEVICE")]
        device: String,
        /// Override the alignment corridor radius; omitted uses config (built-in default: dense).
        ///
        /// A finite band, including 0, changes the soft objective and gradients. This
        /// is not a guaranteed exact replacement for dense alignment. Must match resume config.
        #[arg(long, value_name = "BASES")]
        band_width: Option<usize>,
        /// Teacher logit-MSE coefficient; used only with --teacher; 0 disables its contribution.
        #[arg(long, default_value_t = 1.0, value_parser = nonnegative, value_name = "WEIGHT")]
        distillation_weight: f64,
        /// Export final FP32 ONNX plus NAME.params.json; flag alone writes OUTPUT/model.onnx.
        ///
        /// Omitted disables export. An optional PATH must end in .onnx. Parent directories
        /// are created; existing graph/sidecar files are replaced. Works after CPU,
        /// Metal, or CUDA training without the onnx runtime feature or Python.
        #[arg(long, num_args = 0..=1, value_name = "PATH")]
        export_onnx: Option<Option<PathBuf>>,
    },
    /// Evaluate a native checkpoint's alignment loss and edit errors on labeled windows.
    ///
    /// Accepts native Burn checkpoints only. Evaluation does not
    /// update weights. Transformer execution uses --device; alignment/edit metrics
    /// are computed on CPU. JSON always prints to stdout.
    Evaluate {
        /// Required native checkpoint directory containing params.json and model.burnpack.
        #[arg(long, value_name = "DIR")]
        model: PathBuf,
        /// Required labeled TFRecord or JSONL datasets (optional gzip); one or more paths.
        #[arg(long, num_args = 1.., required = true, value_name = "PATH", value_parser = dataset_path)]
        data: Vec<PathBuf>,
        /// Evaluation windows per forward batch; must be positive.
        #[arg(long, default_value_t = 16, value_parser = positive, value_name = "N")]
        batch_size: usize,
        /// cpu, cuda:N (Linux --features cuda), or metal (macOS --features metal).
        #[arg(long, default_value = "cpu", value_parser = device_name, value_name = "DEVICE")]
        device: String,
        /// Optional JSON report path; omitted writes to stdout only; existing file is replaced.
        #[arg(long, value_name = "JSON")]
        output: Option<PathBuf>,
    },
    /// CPU preprocessing and synthetic-logit alignment microbenchmark, not model accuracy.
    ///
    /// Uses draft CCS as proxy truth and deterministic synthetic logits. Does not
    /// load or benchmark a trained transformer. A finite band changes the objective.
    Benchmark {
        #[command(flatten)]
        input: BamInput,
        /// Parameters JSON; omitted uses built-in v1.2-shaped defaults; no weights are loaded.
        #[arg(long, value_name = "JSON")]
        config: Option<PathBuf>,
        /// Repeated timing samples; each loss sample averages 100 calls.
        #[arg(long, default_value_t = 5, value_parser = positive, value_name = "N")]
        repetitions: usize,
        /// Corridor radius for comparison against dense loss; does not affect model training.
        #[arg(long, default_value_t = 8, value_name = "BASES")]
        band_width: usize,
        /// Required JSON result path; parent must exist; existing file is replaced.
        #[arg(long, value_name = "JSON")]
        output: PathBuf,
    },
}

pub fn config(path: Option<PathBuf>) -> Result<ModelConfig> {
    path.map(ModelConfig::load)
        .transpose()
        .map(|c| c.unwrap_or_default())
}

pub fn positive(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| "must be a positive integer".into())
}

pub fn nonnegative(value: &str) -> std::result::Result<f64, String> {
    value
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| "must be finite and nonnegative".into())
}

fn positive_float(value: &str) -> std::result::Result<f64, String> {
    nonnegative(value).and_then(|n| {
        if n > 0.0 {
            Ok(n)
        } else {
            Err("must be positive".into())
        }
    })
}

fn nonnegative_f32(value: &str) -> std::result::Result<f32, String> {
    let n = nonnegative(value)? as f32;
    if n.is_finite() {
        Ok(n)
    } else {
        Err("exceeds finite f32 range".into())
    }
}

pub fn device_name(value: &str) -> std::result::Result<String, String> {
    if matches!(value, "cpu" | "metal")
        || value
            .strip_prefix("cuda:")
            .is_some_and(|n| n.parse::<i32>().is_ok_and(|n| n >= 0))
    {
        Ok(value.into())
    } else {
        Err("expected cpu, metal, or cuda:N with a nonnegative GPU index".into())
    }
}

fn calibration(value: &str) -> std::result::Result<String, String> {
    Calibration::parse(value)
        .map(|_| value.into())
        .map_err(|e| e.to_string())
}

fn dataset_path(value: &str) -> std::result::Result<PathBuf, String> {
    let path = PathBuf::from(value);
    crate::dataset::is_tfrecord(&path).map_err(|e| e.to_string())?;
    Ok(path)
}
