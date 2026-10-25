# Metal Support and Test Status

Added 2026-10-12. Burn remains pinned to 0.22.0.

## Implementation

- Optional `metal = ["burn/metal", "burn/fusion", "onnx-coreml"]` Cargo feature.
  The `onnx-coreml` feature enables ORT's macOS CoreML backend independently
  when only ONNX inference is needed.
- `--device metal` explicitly initializes the Metal/wgpu native-MSL backend on
  macOS. It returns an error if initialization fails; it does not select a CPU
  or another graphics API as a fallback. Repeated requests reuse one device.
- Transformer and optimizer execute on the selected device. Metal training
  uses the same batched differentiable device wavefront loss as CUDA rather
  than transferring logits to the CPU analytic-loss path.
- CPU and Linux CUDA selectors remain available. The reference-server source
  variant receives the same support without restoring experimental fusions.
- A minimal vendored `burn-optim` 0.22 patch adds explicit-device state restore.
  The original API restores to Device::default(), which changes when backend
  features are enabled; the application now restores on its chosen base device.
  Optimizer algebra and checkpoint schema are unchanged. See vendor PATCH.md.

## Checks Run

Stable Rust 1.99.0 on the available ARM64 macOS host:

- Metal feature compilation succeeded.
- With Metal (including ONNX/CoreML) compiled, 14 main-project unit tests, six CLI tests,
  and two workflow tests passed, including CPU optimizer resume, the actual BAM fixture, and
  native ORT parity. Three hardware tests are ignored in the ordinary suite.
- The reference-server variant passed its 13 unit tests, six CLI tests, and two workflow
  tests with the same features and real BAM/ORT fixtures.
- Clippy with warnings denied passed for all targets with `metal,onnx` enabled.
- Metal builds retain ONNX export/load options. macOS provider choices are
  CPU/CoreML; Linux choices are CPU/CUDA/TensorRT according to compiled features.
  CLI tests verify OS-specific choices and export/load workflows.
- Actual CoreML CPUOnly inference passed for a nonzero-ReZero toy encoder,
  batch sizes 1 and 3, with identical calls and max logit error 1.1920929e-7
  versus ordinary ORT CPU. Registration succeeded with ORT 1.31.0. The default
  compilation directory was denied; setting TMPDIR to a writable workspace
  temporary directory fixed it. CoreML GPU execution remains untested.
- An actual Metal alignment/gradient test was attempted but could not run:
  `AdapterUnavailable: no adapter found on the requested graphics API`.
- A separate native Apple Metal API probe returned zero devices and no default
  device. `system_profiler` reports Apple M4 Max / 40 GPU cores / Metal supported,
  but this execution environment cannot acquire that GPU. This is not a passed
  GPU test, and no Metal output parity, training success, or speed claim is made.

The default installed Rust 1.101 nightly failed in a transitive `branches` crate
because of a removed intrinsic. Use a stable toolchain >=1.95. No dependency
version upgrades were introduced to work around the nightly error.

## Tests to Run on an Accessible Mac GPU

```sh
cargo +stable build --release --locked --features metal
cargo +stable test --release --locked --features metal --test metal \
  -- --ignored --skip published_model_metal_matches_cpu --nocapture --test-threads=1
export DC_METAL_MODEL=/absolute/path/to/native-v1.2-model
export DC_METAL_DATA=/absolute/path/to/train.tfrecord.gz
cargo +stable test --release --locked --features metal --test metal \
  published_model_metal_matches_cpu -- --ignored --nocapture --test-threads=1
```

The synthetic tests check dense/banded loss values and gradients, encoder
forward parity with nonzero ReZero branches, actual Metal training, optimizer
save/resume, CPU checkpoint reload, and evaluation. The published-model test
compares all logits/calls for three real windows. They still need GPU execution.

The `infer_windows` example can process all 6,400 unique windows with synchronized
timings. Use the existing data/model archives; compare CPU and Metal outputs
before benchmarking warmed batches. Loading, shader compilation, and autotuning
costs should be reported separately from steady-state inference.

## ONNX/CoreML

`inference --model model.onnx --device metal` uses CoreML CPUAndGPU,
not Burn tensor kernels. CoreML and ORT may execute CPU partitions. Plain
`--device cpu` uses ORT CPU; `--device cpu --provider coreml` uses CoreML CPUOnly.
Training stays in Burn; `train --export-onnx` is available in Metal builds.
The runtime must include CoreML, and MLProgram requires macOS 12+.

See CLI.md for runtime/temporary-directory setup, compute-plan logging, and
`--test coreml` CPU/GPU parity commands. Three Burn hardware tests plus two
CoreML tests are normally ignored; the CPU-only CoreML test was separately run.
