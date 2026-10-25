# deepconsensus-rs

Rust implementation of the DeepConsensus learned-values, gap-aware sequence transformer. 

The main CLI contains `preprocess`, `inference`, `train`, `evaluate`, and
`benchmark`. Optional checkpoint tools live in the separate
`deepconsensus-model` binary. See [CLI.md](CLI.md) for every option's defaults,
path/model rules, and build/device combinations; each command also has detailed
`--help`. No pretrained model is selected or downloaded implicitly.

The supplied 6,400-window truth-labeled subset and matched one-epoch checkpoints
are separate archives. Benchmark reports and reproduction scripts are included
under `validation/`.

Burn **0.22.0** supplies training and tensor execution. Optional `ort`
**2.0.0-rc.13** supplies ONNX Runtime inference with CPU, macOS CoreML, or Linux CUDA/TensorRT providers.

## Implemented Pipeline

- Standard PacBio BAM input through `rust-htslib` / HTSlib.
- Molecule-grouped subreads **already aligned to their CCS molecule**, plus a CCS BAM.
- CIGAR expansion, kinetic-tag orientation, long-insertion trimming, shared gap columns.
- Fixed or CCS `wl` smart windows; original feature order and gap-quality sentinel.
- Learned feature embeddings, TensorFlow-style sine/cosine positional encoding,
  local attention, ReZero or pre-normalized residuals, ReLU FFN, final normalization.
- Native TensorFlow TensorBundle checkpoint import, with checksums, without TensorFlow.
- Native Burn checkpoints, AdamW state, seeded buffered training, resume, and evaluation.
- Dense or endpoint-convex-hull banded differentiable alignment loss.
- Optional frozen-teacher logit-MSE distillation with configurable weight.
- Original-format TFRecord `tf.Example` read/write with CRC32C and optional gzip;
  JSONL datasets are also supported.
- ONNX export in Rust; dynamic batches, fixed feature/window dimensions.
- Batched Burn or ORT inference, quality calibration, quality-based window skipping,
  gap removal, molecule stitching, read filtering, FASTQ / FASTQ.gz output.

CCS generation, subread-to-CCS alignment (`ccs` / `actc`), and truth-to-CCS alignment
are upstream inputs, as in the original DeepConsensus workflow. This program does
not invent alignments for unaligned raw subreads. The unnecessary DADA2-style
aligner and its dependency were removed.

## Build on Linux

Install Rust 1.95 or newer and the native build dependencies:

```sh
sudo apt-get install build-essential pkg-config clang libclang-dev cmake curl git
cargo build --release --locked
cargo test --locked
```

CPU execution uses Burn Flex with SIMD and Rayon enabled. `--cpu-threads N`
controls the Rayon pool; zero selects its automatic default. The BAM `--threads`
option controls HTSlib decompression; `--ort-threads` independently controls
ONNX Runtime intra-op workers.

```sh
cargo build --release --locked --features onnx
cargo build --release --locked --features cuda
cargo build --release --locked --features onnx-coreml
cargo build --release --locked --features cuda,tensorrt
```

## macOS Metal

An optional native Metal path uses Burn 0.22's explicitly selected Metal/wgpu
backend with kernel fusion. CPU remains the default; no CPU fallback is selected
when `--device metal` is requested. CUDA remains Linux-only, and ORT GPU providers
are selected according to the model format. The `metal` feature includes
`onnx-coreml`: native directories use Burn Metal and ONNX files use ORT CoreML
with CPUAndGPU when `--device metal` is selected. All export/load options remain
available. macOS exposes CPU/CoreML provider choices, Linux exposes enabled
CPU/CUDA/TensorRT choices; unavailable choices are compiled out of the enum.

Use a stable Rust toolchain >=1.95. The installed 1.101 nightly encountered a
transitive `branches` intrinsic incompatibility; stable is used for Metal testing.

```sh
cargo +stable build --release --locked --features metal
target/release/deepconsensus-rs --cpu-threads 4 inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model models/v1.2 \
  --device metal --batch-size 64 --output metal.fastq.gz
target/release/deepconsensus-rs --cpu-threads 4 train \
  --data train.tfrecord.gz --resume warm-start --output metal-trained \
  --device metal --epochs 1 --batch-size 32
```

Feature construction/decoding remains Rayon CPU work. The transformer, optimizer,
and batched differentiable alignment loss run on Metal during training.
Training can also use `--export-onnx`; CoreML inference requires a matching
CoreML-enabled ORT shared library through ORT_DYLIB_PATH. MLProgram requires
macOS 12+. CoreML may execute CPU partitions; registration is not GPU coverage.
First-use shader compilation/autotuning is distinct from warm throughput.
Optimizer restoration explicitly uses the selected device through a small
vendored Burn optimizer compatibility patch; CPU resume does not initialize a GPU.

Hardware tests include dense/banded loss values and gradients, nonzero ReZero
forward parity, GPU training/checkpoint/optimizer resume, and published-model
parity on three real windows:

```sh
export DC_METAL_MODEL="$PWD/models/v1.2"
export DC_METAL_DATA="$PWD/data/train.tfrecord.gz"
cargo +stable test --locked --features metal --test metal \
  -- --ignored --nocapture --test-threads=1
```

For synchronized forward inference over all 6,400 labeled windows:

```sh
cargo +stable run --release --locked --features metal --example infer_windows -- \
  --model models/v1.2 --data data/train.tfrecord.gz --output results/metal-windows \
  --device metal --batch-size 64 --repeats 3
```

This produces raw logits, per-window FASTA, and timings; sampled windows are not
stitched into complete molecules. See `validation/METAL.md` for actual test status.

Burn CUDA uses CubeCL, including its CUDA and compiler dependencies. Its first
build may download a Tracel LLVM bundle into the normal user cache. Build on the
Linux GPU host with a compatible NVIDIA driver/toolkit. The optional ONNX path
uses dynamic linking: set `ORT_DYLIB_PATH` to your ONNX Runtime library, and make
its CUDA/cuDNN/TensorRT dependencies discoverable with `LD_LIBRARY_PATH`.
The runtime must expose ORT API 27 or newer; CPU validation used ONNX Runtime 1.31.
Do not assume arbitrary CUDA, cuDNN, and TensorRT versions are interchangeable.

## Published Model

```sh
bash scripts/fetch-model.sh models
```

The script downloads the published v1.2 checkpoint and parameters, verifies pinned
SHA-256 hashes, then imports it into `models/v1.2/` using `deepconsensus-model`.
Alternatively, import local upstream checkpoint files:

```sh
target/release/deepconsensus-model import-tf \
  --checkpoint original/checkpoint --params original/params.json --output models/v1.2
```

Import supports ordinary little-endian F32 TensorBundle tensors and the supported
learned-values transformer layout. The published five-layer v1.2 model was tested.
Legacy CNN/FC/DNABERT architectures are not supported; incompatible options are
rejected. The helper's `init` creates random weights, not a correction-ready
model; ordinary `train` initializes weights itself. `inspect-tf` is an optional
tensor-name diagnostic for import failures, never a training/inference step.

## Correct Reads

The subread BAM must be grouped by CCS reference name. The CCS BAM must contain
the matching CCS records in the same molecule order. Subreads need `pw`, `ip`,
and four-value `sn` tags. CCS records need real base qualities. `--smart-windows`
requires the CCS `wl` tag. BAMs aligned to genomic contigs are not valid inputs.

```sh
target/release/deepconsensus-rs --cpu-threads 8 inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam \
  --model models/v1.2 --output corrected.fastq.gz \
  --batch-size 64 --threads 4 --report run.json
```

Defaults: trim insertion runs longer than five bases; skip windows whose
probability-averaged CCS quality exceeds Q45; preserve skipped CCS qualities;
apply the model's DeepConsensus calibration; cap output quality at Q93.
Use `--skip-above 0` for inference on every non-overflow window. Filtering defaults
are permissive (`--min-length 1 --min-quality 0`), so choose production thresholds
explicitly. Skipped windows are not replaced with newly predicted bases.

Burn CUDA:

```sh
target/release/deepconsensus-rs --cpu-threads 8 inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam \
  --model models/v1.2 --device cuda:0 --batch-size 256 --output burn-cuda.fastq.gz
```

ORT CPU / CUDA / TensorRT:

```sh
target/release/deepconsensus-model export-onnx --model models/v1.2 --output model.onnx
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
target/release/deepconsensus-rs inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model model.onnx \
  --device cpu --output ort-cpu.fastq.gz
target/release/deepconsensus-rs inference \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --model model.onnx \
  --device cuda:0 --provider tensorrt --batch-size 256 --output trt.fastq.gz
```

Exports include `model.params.json`; move it with `model.onnx`. Older exports
without this sidecar require `--config models/v1.2/params.json`. Native models
reject `--config`/`--provider`. With an ONNX file, `--device cuda:N` selects ORT
CUDA automatically; the `onnx-cuda` feature and Linux are required. TensorRT
additionally requires `tensorrt` and `--provider tensorrt`. The GPU index is
honored by both providers. TensorRT registration errors are surfaced;
unsupported graph partitions may still fall back to CUDA/CPU. TensorRT caches
engines in `.trt-cache/`. Set `DC_TRT_FP16=1` to enable FP16 engine compilation,
then verify base calls, quality calibration, and yield against FP32. FP16 is off
by default. The graph's input is F32 `rows[batch,feature_rows,window_length]` and
its output is F32 `logits[batch,window_length,5]`; this is not a wrapper for
arbitrary third-party ONNX input/output conventions.

## Prepare and Train

An indexed truth-to-CCS BAM can supply labels. Keep held-out molecules/contigs
separate yourself; no automatic genomic split is inferred.

```sh
target/release/deepconsensus-rs preprocess \
  --subreads subreads_to_ccs.bam --ccs ccs.bam --truth truth_to_ccs.bam \
  --config models/v1.2/params.json --output train.tfrecord.gz
target/release/deepconsensus-rs --cpu-threads 8 train \
  --data train.tfrecord.gz --config models/v1.2/params.json \
  --output trained --epochs 5 --batch-size 32 --learning-rate 0.001
target/release/deepconsensus-rs train \
  --data train.tfrecord.gz --resume trained --output trained --epochs 10
target/release/deepconsensus-rs evaluate \
  --model trained --data heldout.tfrecord.gz --output evaluation.json
```

Training supports the original labeled TFRecords directly. `--epochs` is the
total desired number of epochs, not extra epochs. Checkpoints save at epoch
boundaries. `--max-examples` is useful for smoke tests. `--teacher teacher_dir`
adds frozen-teacher logit MSE; `--distillation-weight` controls its coefficient.
The configurable AdamW/cosine/warmup recipe is not a promise to reproduce the
original TPU/Vizier or distillation training run bit for bit.

`train --export-onnx` exports the successful final checkpoint
to `OUTPUT/model.onnx` and `OUTPUT/model.params.json`; an optional path selects
another graph/sidecar pair. Export needs neither Python nor the `onnx` runtime
feature. Native checkpoints are still saved for resume/evaluation. To export
existing weights without retraining, use the separate helper's `export-onnx`.

For GPU training, build `--features cuda` and add `--device cuda:0`. The CUDA
path uses a differentiable, batched tensor wavefront loss on device. The CPU path
uses a compact Rayon-parallel analytic forward/backward loss with an exact
first-order vector-Jacobian product injected into Burn autodiff. Higher-order
derivatives through that CPU loss are not supported.

`--band-width 8` selects the released endpoint-convex-hull corridor. Published
v1.2 parameters use an **unbanded** loss by default. A finite band restricts the
soft objective and may change gradients and accuracy. It is not an exact
unbanded-score guarantee or an error-rate-estimation algorithm. Empty truth
sequences may be unreachable under a narrow band and are rejected.

## Optimizations and Tests

Implemented exact-input optimizations: insertion-run event layout instead of
column-stepping every pass; compact maps only for selected feature passes while
all passes influence gaps; grouped embedding dispatch; Rayon kinetic expansion,
molecule preprocessing, loss batches, and decoding; SIMD/Rayon CPU tensor
execution; folded embedding scaling/gap masking; bounded preprocessing/inference overlap; early feature-allocation
avoidance for skipped windows; active-cell CPU soft-DP storage and computation.

CUDA uses Burn/CubeCL tensor kernels and fusion. The GPU wavefront is batched
and differentiable, but is **not** a custom fused sparse-band CUDA kernel.
Combined Q/K/V projections, flattened matrix scheduling, and Burn's attention
primitive are available as experimental `model::ExecutionOptions`, but are not
enabled by default: the CPU ablation did not establish an overall gain.
Training retains explicit attention dropout; the primitive is inference-only.
An experimental ONNX export emits combined Q/K/V weights and 2D affine matrices
to expose runtime Gemm/bias/activation fusion. It was slower on the tested CPU,
so the original export remains the default. Test the alternative explicitly with
`deepconsensus-model export-onnx --experimental-fusions`,
or `onnx::export_with_options(model, path, true)`.

Sparse attention, POA ambiguity routing, pass-selection heuristics, INT8,
and novel learned skip gates are not implemented: those change behavior or need
additional hardware/accuracy validation.

```sh
bash scripts/fetch-fixtures.sh fixtures
export DC_FIXTURE_ROOT="$PWD/fixtures/human_1m"
cargo test --locked --features onnx
cargo clippy --all-targets --features onnx -- -D warnings
target/release/deepconsensus-rs --cpu-threads 4 benchmark \
  --subreads "$DC_FIXTURE_ROOT/subreads_to_ccs.bam" \
  --ccs "$DC_FIXTURE_ROOT/ccs.bam" --limit 10 --output benchmark.json
```

Set `ORT_DYLIB_PATH` during tests to exercise native ORT parity. Tests otherwise
skip only the runtime-dependent ORT comparison; the ONNX exporter still runs.
CPU tests cover loss finite differences, tensor-wavefront gradients, randomized
gap layouts, TFRecord round trips, real BAM input, training, optimizer resume,
checkpoint reload, evaluation, and export. See `validation/RESULTS.md` for actual
measurements, scope, and remaining validation work.

A subsequent [matched Python CPU comparison](validation/PYTHON_COMPARISON.md)
measured 9.28 s for upstream SavedModel, 8.93 s for Rust/Burn, and 1.84 s for
Rust/ORT on the small fixture. See that report before extrapolating speedups.

The [larger matched training experiment](validation/LARGER_TRAINING_COMPARISON.md)
uses 6,400 unique truth-labeled windows from 40 molecules, with separate evaluation
and test subsets. CPU training was tied before further optimization, not 10x
faster. See [model optimization](validation/MODEL_OPTIMIZATION.md) for the
same-data output checks and before/after runtime measurements, and the
[gap-aware explanation](validation/GAP_AWARE_TRANSFORMER.md) for the model/loss
distinction.

On the Linux GPU host, run the explicitly hardware-dependent test:

```sh
cargo test --release --features cuda --test gpu -- --ignored --nocapture
```

## References

DeepConsensus: https://github.com/google/deepconsensus

Original paper: https://www.nature.com/articles/s41587-022-01435-7

Released banded loss: https://github.com/google/deepconsensus/tree/banded_alignment
