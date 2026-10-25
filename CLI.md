# Command-Line Guide

The main binary has five workflow commands: `preprocess`, `inference`,
`train`, `evaluate`, and `benchmark`. Run any command with `--help` for
its complete option descriptions. The old `run` spelling is a hidden
compatibility alias; new scripts use `inference`.

`deepconsensus-model` is a separate, optional Rust helper for `init`,
`import-tf`, `inspect-tf`, and standalone `export-onnx`. These are not main
CLI commands. No Python or TensorFlow installation is required.

## Build and Device Rules

Use stable Rust >=1.95. Build both binaries with `--bins`.

| Purpose | Build features | Execution |
| --- | --- | --- |
| Native CPU training/inference | none | `--device cpu` (default) |
| Native macOS GPU training/inference | `metal` | `--device metal` |
| Native Linux NVIDIA training/inference | `cuda` | `--device cuda:N` |
| ONNX CPU inference (macOS/Linux) | `onnx`, or either native GPU feature | `--device cpu` |
| ONNX macOS CoreML inference | `onnx-coreml` or `metal` | `--device metal` (CPU/GPU), or `--device cpu --provider coreml` (CoreML CPUOnly) |
| ONNX Linux NVIDIA inference | `onnx-cuda` or `cuda` | `--device cuda:N` |
| ONNX Linux TensorRT inference | `tensorrt` | `--device cuda:N --provider tensorrt` |

For Linux Burn training plus ORT CUDA inference, build `--features cuda`.
For Burn training plus TensorRT inference, build `--features cuda,tensorrt`.
The GPU index N is nonnegative and is honored by both ORT CUDA and TensorRT.

`onnx` is the common CPU runtime feature. Platform-specific additions are
`onnx-coreml` (macOS), `onnx-cuda` (Linux), and `tensorrt` (Linux).
`metal` automatically includes `onnx-coreml`; `cuda` includes `onnx-cuda`.
ONNX export/load options remain available in Metal builds. Only providers
compiled for the target OS appear in `--provider`: CPU/CoreML on macOS,
CPU/CUDA/TensorRT on Linux as enabled. Export remains available without an
ORT runtime feature because it uses Rust protobuf serialization.

GPU selection is explicit; CPU remains the default even in GPU builds.
Unavailable GPU hardware is an error, not an automatic CPU fallback.
With a native checkpoint, `--device metal` uses Burn Metal; with an ONNX file,
it uses ORT CoreML with CPUAndGPU. CoreML can assign operations to CPU, and ORT
can place unsupported partitions on its CPU provider; this is not GPU-only
execution. Explicit provider registration failures are errors.
GPU runtime/precision/throughput still require testing on your hardware.

## Paths and Model Formats

- Paths are relative to the working directory unless absolute.
- No weights are downloaded or selected implicitly. Required paths have no default.
- Native checkpoints contain `params.json` and `model.burnpack`. Training
  also saves `optimizer.burnpack`, `training-state.json`, and
  `training-options.json` after each epoch.
- An ONNX export named `models/consensus.onnx` has a required configuration
  sidecar `models/consensus.params.json`. Transfer both files together.
- Older ONNX graphs can supply the correct parameters with
  `inference --model old.onnx --config native-model/params.json`.
  Configuration is never guessed from generic architecture defaults.
- The ONNX input is FP32 `rows[batch,features,window_length]`; output is FP32
  `logits[batch,window_length,5]`. This loader is for this DeepConsensus
  export convention, not arbitrary ONNX models.
- Dataset/output suffix `.gz` enables gzip. JSONL datasets use
  `.jsonl` or `.jsonl.gz`; TFRecord uses `.tfrecord` or `.tfrecords`,
  optionally followed by `.gz`. Unsupported suffixes are rejected.
- Training and export create their output directories. Other output/report
  parents must already exist. Existing output files are replaced.
- Built-in configuration describes an architecture, not pretrained weights:
  100-column windows, 20 passes, width 280, five layers, two heads, FFN 2048,
  local attention radius 12, CCS qualities, ReZero, and dense alignment loss.
  Use the published model's actual `params.json` to preserve its parameters
  and calibration. The example native directory name `models/v1.2` is not
  a CLI default; the server bundle instead uses `models/burn-v1.2`.

## Shared CPU and BAM Options

The global `--cpu-threads` default is **0**: Rayon chooses automatically,
respecting its environment configuration. It is separate from BAM/ORT workers.

`preprocess`, `inference`, and `benchmark` share these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--subreads BAM` | required | Subreads already aligned to their CCS, grouped by CCS reference; requires pw/ip/sn kinetic tags, not genomic-mapping BAMs |
| `--ccs BAM` | required | Matching CCS records in the same molecule order with real qualities |
| `--threads N` | 4 | HTSlib decompression workers per BAM reader; 0 disables threaded decompression |
| `--ins-trim BASES` | 5 | Discard insertion runs longer than BASES; 0 retains all insertions |
| `--limit N` | all molecules | Positive molecule cap, not a window cap |
| `--smart-windows` | off | Use CCS wl-tag boundaries; a valid wl tag is required |

## Preprocess

| Option | Default | Meaning |
| --- | --- | --- |
| `--truth BAM` | none | Indexed truth-to-CCS alignments for labels; without it, output is unlabeled |
| `--config JSON` | built-in architecture | Match the intended model's parameters; no weights loaded |
| `--output PATH` | required | JSONL/TFRecord dataset, optionally gzip |

No automatic train/validation split is performed. Split by molecule/contig
before training to avoid leakage. Inference does not accept `--truth`.

## Inference

| Option | Default | Meaning |
| --- | --- | --- |
| `--model PATH` | required | Native directory or .onnx file |
| `--output FASTQ` | required | Stitched molecule FASTQ; .gz enables gzip |
| `--device DEVICE` | cpu | Native CPU, macOS Metal, or Linux cuda:N as compiled |
| `--config JSON` | ONNX sidecar | ONNX only; rejected for native directories |
| `--provider PROVIDER` | from device | ONNX only: OS/build-specific cpu, coreml, cuda, tensorrt; must agree with --device |
| `--ort-threads N` | 4 | ONNX intra-op workers, independent of BAM threads; ignored for native Burn |
| `--batch-size N` | 32 | Maximum windows per model batch |
| `--block-molecules N` | 16 | Bounded CPU preprocessing block size |
| `--skip-above Q` | 45 | Preserve CCS windows above probability-averaged Q; 0 disables gating |
| `--min-quality Q` | 0 | Minimum stitched read quality; 0 disables quality filtering |
| `--min-length BASES` | 1 | Minimum output length; empty reads always filtered |
| `--max-quality Q` | 93 | FASTQ quality cap, allowed 0..=93 |
| `--dc-calibration SPEC` | model config | Predicted quality calibration: skip or threshold,slope,intercept |
| `--ccs-calibration SPEC` | skip | Calibration for preserved CCS qualities |
| `--report JSON` | stdout only | Additionally write JSON timings/yield to a file |

Overflow windows are always preserved from CCS. Skipped windows are not newly
predicted sequences. Calibration is linear above its threshold (threshold 0
applies everywhere), then clamped and rounded. The two calibration sources
must not be confused.

## Train

No initialization command is needed. Without `--resume`, training starts from
seeded random weights. A weights-only published checkpoint can be fine-tuned;
an actual training checkpoint additionally restores its optimizer/epoch state.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data PATH...` | required | One or more labeled TFRecord/JSONL datasets, optional gzip |
| `--output DIR` | required | Native checkpoint directory, created as needed |
| `--config JSON` | resume/params.json, else built-in | Explicit config must exactly match resume config |
| `--resume DIR` | none | Native checkpoint; output may equal this directory |
| `--teacher DIR` | none | Frozen native teacher; no ONNX teacher |
| `--epochs N` | 1 | Total target epochs including saved epochs, not additional epochs |
| `--batch-size N` | 16 | Windows per update |
| `--learning-rate RATE` | 0.001 | Positive peak AdamW/cosine learning rate |
| `--end-learning-rate RATE` | 0.00001 | Nonnegative final cosine rate |
| `--warmup-steps N` | 0 | Linear warmup update count, 0 disables |
| `--weight-decay RATE` | 0.0069868 | AdamW decay, 0 disables |
| `--seed N` | 1 | Initialization and per-epoch shuffle seed; not cross-device bitwise determinism |
| `--shuffle-buffer N` | 1024 | Streaming window buffer; 1 preserves dataset order |
| `--max-examples N` | all windows | Positive window cap across datasets per epoch |
| `--device DEVICE` | cpu | Burn CPU, Linux cuda:N, or macOS metal |
| `--band-width BASES` | config, built-in dense | Finite corridor including 0 changes objective/gradients; must agree with resume |
| `--distillation-weight WEIGHT` | 1 | Nonnegative teacher logit-MSE coefficient, relevant only with --teacher |
| `--export-onnx [PATH]` | off | All builds: final FP32 graph plus sidecar; flag alone writes OUTPUT/model.onnx |

AdamW uses beta1=0.9, beta2=0.997, epsilon=1e-9. This is not a promise to
reproduce the original TPU/Vizier/LAMB training recipe. CPU analytic loss
supports exact first-order gradients, not higher-order derivatives.

Export happens only after successful training, using the persisted final
checkpoint loaded on CPU. No ORT runtime or Python is needed for export.
An explicit export path must end in .onnx and is validated before training.
The native checkpoint remains available for evaluation/resume. To export
published or already-trained weights without retraining, use the separate helper.

## Evaluate and Benchmark

`evaluate` requires `--model DIR` (native) and `--data PATH...` (labeled).
Defaults are `--batch-size 16 --device cpu`. Optional `--output JSON`
additionally writes the stdout report. The transformer runs on the selected
device; alignment/edit metrics use CPU. No model parameters are changed.

`benchmark` accepts the shared BAM options, optional `--config JSON`,
`--repetitions 5`, `--band-width 8`, and required `--output JSON`.
It benchmarks CPU preprocessing and synthetic-logit alignment with CCS proxy
labels, not transformer inference, GPU throughput, or truth accuracy. Each loss
timing sample averages 100 calls.

## Separate Model Helper

| Command | Required options | Optional/defaults |
| --- | --- | --- |
| `init` | `--output DIR` | `--config JSON`: built-in; `--seed 1`; random weights only |
| `import-tf` | `--checkpoint PREFIX --params JSON --output DIR` | No guessed parameters or implicit download |
| `inspect-tf` | `--checkpoint PREFIX` | Read-only tensor-name diagnostics |
| `export-onnx` | `--model DIR` | `--output PATH` defaults to DIR/model.onnx plus DIR/model.params.json |

The checkpoint prefix excludes .index; e.g. `original/checkpoint` refers to
`checkpoint.index` and `checkpoint.data-00000-of-00001`. Import supports the
published learned-values transformer, not arbitrary TF architectures.
`inspect-tf` helps debug tensor mapping/import failures and is unnecessary
once native weights have been prepared. The helper refuses init/import into
directories containing optimizer/training state to avoid mixing new weights
with a stale training checkpoint.

## Migration

- Main `init/import-tf/inspect-tf/export-onnx` commands move to `deepconsensus-model`.
- Use `inference` instead of `run`; the latter remains a hidden compatibility alias.
- Replace `--model native --onnx graph.onnx` with `--model graph.onnx`.
  For old graphs without a sidecar, add `--config native/params.json`.
- ONNX GPU selection now requires explicit `--device cuda:N` or `metal`; provider and
  device conflicts are errors rather than silently ignored settings.
- BAM `--threads` no longer doubles as ORT intra-op thread count; use `--ort-threads`.
- Metal builds now include CoreML and retain all ONNX workflow options.

## Examples

Linux CUDA build (use `--features onnx` and `--device cpu` for CPU-only inference):

```sh
cargo +stable build --release --locked --features cuda --bins
target/release/deepconsensus-rs train \
  --data data/train/train.tfrecord.gz --resume models/burn-v1.2 \
  --output trained --device cuda:0 --epochs 1 --export-onnx
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
target/release/deepconsensus-rs inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model trained/model.onnx \
  --device cuda:0 --output corrected.fastq.gz
```

Metal build:

```sh
cargo +stable build --release --locked --features metal --bins
target/release/deepconsensus-rs train \
  --data train.tfrecord.gz --resume models/v1.2 \
  --output metal-trained --device metal --epochs 1 --export-onnx
target/release/deepconsensus-rs inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model metal-trained \
  --device metal --output metal.fastq.gz
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.dylib
target/release/deepconsensus-rs inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model metal-trained/model.onnx \
  --device metal --provider coreml --output coreml.fastq.gz
```

## CoreML Runtime

This implementation selects MLProgram format (macOS 12+) and permits dynamic
batch sizes. `--device metal` selects CPUAndGPU; `--device cpu --provider coreml`
selects CPUOnly. Plain `--device cpu` still selects ORT's ordinary CPU provider.
Native Burn training is unchanged; CoreML is an inference-only route.

Set ORT_DYLIB_PATH to an API-27-compatible macOS runtime that includes CoreML
(CPU validation uses runtime 1.31.0). The Rust Cargo feature alone cannot add an
execution provider to a runtime library built without it. Custom ORT builds
need `--use_coreml`. Missing registration errors are surfaced, not silently
swallowed. See the [official CoreML documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

The graph/input/output are FP32; low-precision GPU accumulation is explicitly
disabled, but this is not a guarantee of identical intermediate precision,
base calls, or qualities on every device. Verify parity and provider coverage.
CoreML model caching is not enabled by default, avoiding stale models after
in-place weight updates. Cold conversion/compilation can be expensive. Set
`DC_COREML_PROFILE=1` for CoreML compute-plan logging when diagnosing placement.
If a restricted environment denies CoreML's default compilation directory,
set TMPDIR to an existing writable directory before launching the process.
For example: `mkdir -p /private/tmp/deepconsensus-coreml` followed by
`export TMPDIR=/private/tmp/deepconsensus-coreml/`. No environment is mutated
inside the running Rust program. This workaround passed CPU-only CoreML parity
checks here; it does not grant GPU access.

```sh
cargo +stable test --release --locked --features metal --test coreml \
  coreml_cpu_matches_ort_cpu -- --ignored --nocapture --test-threads=1
cargo +stable test --release --locked --features metal --test coreml \
  coreml_gpu_matches_ort_cpu -- --ignored --nocapture --test-threads=1
```

With the runtime/TMPDIR configured, `DC_COREML_CPU_TEST=1 cargo +stable test
--features metal --test cli cli_training_inference_and_model_helper` also
exercises CoreML CPUOnly through the full synthetic BAM-to-FASTQ command.

For TensorRT, use the tensorrt feature and add --provider tensorrt to ONNX
inference. The graph uses FP32; DC_TRT_FP16=1 enables FP16 engine compilation,
off by default. Verify outputs before using reduced precision. TensorRT caches
engines in .trt-cache relative to the working directory; unsupported partitions
may fall back to CUDA/CPU, despite registration failures being errors.
