# CLI and CoreML Validation

ARM64 macOS, stable Rust 1.99.0, 2026-10-12.

The current CLI keeps ONNX options in every build. metal includes onnx-coreml;
cuda includes onnx-cuda; tensorrt stays optional. Provider enums are compiled
by OS and enabled feature: macOS CPU/CoreML, Linux CPU/CUDA/TensorRT.

| Source variant | Features | Passed | Normally ignored |
| --- | --- | --- | --- |
| Current | onnx | 14 unit + 6 CLI + 2 workflow | none relevant |
| Current | metal | 14 unit + 6 CLI + 2 workflow | 3 Burn + 2 CoreML |
| Reference | onnx | 13 unit + 6 CLI + 2 workflow | none relevant |
| Reference | metal | 13 unit + 6 CLI + 2 workflow | 3 Burn + 2 CoreML |

Runtime-dependent CPU tests used ORT 1.31.0 through ORT_DYLIB_PATH and the real
BAM fixture through DC_FIXTURE_ROOT. Tests cover command defaults/invalid inputs,
OS-specific provider choices, default/custom train exports, resume, standalone
export, config sidecars, old-graph explicit config, native/ORT CPU FASTQ parity
on a nonempty tiny BAM fixture, and dataset plain/gzip suffix handling.
Main checkpoint utility commands remain in the separate Rust helper.

## CoreML CPU Execution

The ignored coreml_cpu_matches_ort_cpu test was separately run for the current
variant. It registered CoreML, compiled the exported toy encoder with nonzero
ReZero branches, and performed CPUOnly inference for batch sizes 1 and 3.
Both passed, with identical base calls and a maximum logit difference
1.1920929e-7 against ordinary ORT CPU.

The first attempt failed because CoreML could not create its compilation
working directory beneath the default /var/folders temporary URL.
A second run with TMPDIR set to an existing writable workspace temporary
directory succeeded. There is no unsafe environment mutation in the program.

```sh
env TMPDIR=/absolute/path/to/existing/writable/tmp/ \
  ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.dylib \
  cargo +stable test --config 'profile.dev.package.deepconsensus-rs.debug=0' \
  --locked --features metal --test coreml coreml_cpu_matches_ort_cpu \
  -- --ignored --nocapture --test-threads=1
```

This is a toy-model CPUOnly CoreML check, not published-model accuracy, all
6,400-window parity, Apple GPU parity, or a speed benchmark. It does establish
registration, conversion, dynamic-batch execution, and sampled output parity.

Both variants also passed their opt-in end-to-end CLI check with
DC_COREML_CPU_TEST=1 and a writable TMPDIR. This trains/resumes a tiny native
checkpoint, exports ONNX, reads a nonempty synthetic BAM, selects
`--device cpu --provider coreml`, and produces FASTQ identical to native Burn
CPU output. CoreML CPUOnly selection is checked in the subprocess logs.

## GPU Scope

No GPU execution was requested during these checks. Native Burn Metal GPU
access remains unavailable here; CoreML CPUAndGPU and Linux CUDA/TensorRT
precision, device assignment, training, and throughput require hardware tests.

CoreML selects MLProgram and permits dynamic input batches. CPUAndGPU is
allowed for --device metal; --device cpu --provider coreml selects CPUOnly.
Low-precision GPU accumulation is disabled. CoreML/ORT can still place graph
partitions on CPU, so selecting the provider does not imply GPU-only execution.
CoreML caching is disabled to avoid stale weights under in-place model updates.
DC_COREML_PROFILE=1 enables compute-plan logs.

Both source variants keep the minimal Burn optimizer restore patch. Reference
transformer/export paths remain unfused; no recent experiments were reintroduced.
All-target Clippy with warnings denied passed for both variants under metal
and combined metal,onnx-cuda,tensorrt on this macOS host. Formatting passed.
The inference-only onnx-coreml feature also passed all-target compilation.
These macOS checks do not validate Linux-specific code execution. Native Linux GPU
compilation is represented by the supplied CI configuration, not run here.
Debug Metal links emit a nonfatal large-unwind-section warning. Local test
runs reduce debug information for this package without changing release profiles.
