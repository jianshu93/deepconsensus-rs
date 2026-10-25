# CPU Validation

Recorded 2026-10-11 on the available macOS 15.7.9 ARM64 CPU host. Linux is the
deployment target, but **no Linux host or NVIDIA GPU was available here**.
Rust: 1.101.0-nightly. Release builds used SIMD/Rayon Burn Flex; inference used
four Rayon threads and four ORT threads. Hardware CPU/memory identification was
blocked by the host sandbox, so no processor model is inferred.

## Inputs and Model

- Upstream r1.2 commit `979108ddbc0a950ea493071690507ac32c8ee9cb`.
- Bundled `deepconsensus/testdata/human_1m` fixture: 10 molecules, 93 mapped
  subreads, 1,593 inference windows. This is a small development fixture, not the
  full publication benchmark or a representative sequencing run.
- Published v1.2 model: 5 encoder layers, hidden size 280, 2 heads, FFN 2048,
  length 100, max 20 feature passes, attention radius 12.
- Original TensorFlow checkpoint imported directly in Rust. All selected tensor
  checksums and the index-table checksums passed.
- TensorFlow 2.20 and ONNX Runtime 1.31 were used only for developer reference
  checks. They are not dependencies of native Burn execution.

## Correctness

The unmodified upstream Python gap-spacing/preparation function bodies were
executed in an isolated verification harness, then their inference features
were assembled in upstream row order. All 1,593 Rust windows matched the
reference F32 feature arrays, CCS quality arrays, molecule names, and starts
exactly, with insertion trimming set to five in both paths.

On the first three real windows, a TensorFlow reconstruction using the original
checkpoint tensors and upstream operations gave:

| Check | Result |
| --- | --- |
| Burn vs TensorFlow maximum absolute logit error | 9.536743e-6 |
| Exported ONNX vs TensorFlow maximum logit error | 1.239777e-5 |
| Argmax base calls | Identical |
| ONNX checker | Passed |
| Rust vs unmodified upstream loss maximum error | 3.227655e-6 |
| Rust vs upstream logit-gradient maximum error | 7.450581e-9 |

The upstream loss was extracted from r1.2 and executed with TensorFlow on the
same logits, using draft CCS tokens as proxy labels. This verifies numerical
loss/gradient behavior; those proxy labels are not an accuracy truth set.

Additional checks:

- 12 unit tests and 2 integration tests passed, both with default features and
  with `onnx` enabled. The native ORT comparison ran with `ORT_DYLIB_PATH` set.
- CPU tensor-wavefront loss values/gradients matched the compact analytic VJP.
- Finite-difference tests passed for dense and finite-band soft alignment.
- 200 randomized insertion layouts matched the original column-walk algorithm.
- Smart-window boundaries, overflow handling, and early skip feature suppression
  were checked with synthetic reads.
- Seeded tiny-model training, epoch checkpointing, optimizer resume, reload,
  evaluation, and ONNX export completed.
- A full-size model training update on two real labeled windows completed and
  saved a checkpoint. This was a smoke test, not a converged training experiment.
- Full Burn and native ORT inference on all ten molecules produced byte-identical
  decompressed FASTQ, with skipping disabled: 109,143 output bases.
- Forcing every window to skip reproduced original CCS bases and qualities
  exactly for all ten molecules: 116,018 bases.
- Changing preprocessing block size preserved the inferred FASTQ.
- Formatting and Clippy with warnings denied passed.

## Timing

These are small-input measurements, not controlled end-to-end speedup benchmarks.
The inference reports exclude model/session initialization; their timers start
after loading the model. Other builds/reference checks sometimes ran concurrently,
so the numbers should be rerun under controlled conditions on Linux.

| Operation | Observed Time |
| --- | --- |
| Burn CPU, 10 molecules, 1,593 windows, no skipping | 10.04 s |
| ORT CPU, same input, blocks of 2 molecules, no skipping | 2.23 s |
| ORT CPU, same input, one larger block | 1.78 s |
| All-window skip route, 10 molecules | 0.023 s |

The isolated loss benchmark used one real window's CCS tokens as a proxy label
and deterministic synthetic logits. Each sample averaged 100 invocations; five
samples were recorded. Its CPU kernel invocations were deliberately serial,
while production training dispatches independent window losses through Rayon.

| Loss | Median Time | Stored / Evaluated Cells |
| --- | --- | --- |
| Dense | 0.310 ms | 8,484 |
| Convex-hull band, half-width 8 | 0.102 ms | 2,038 |

This is about **3.0x** for that loss kernel and a 76% cell reduction. The loss
values were 152.5852720 and 152.5852808 respectively. Their proximity in this
one example is not a guarantee that banding preserves the unrestricted soft
objective. Serial Rust feature preparation for the ten molecules had a median
of about 11 ms, excluding BAM reading and model execution.

Raw reports are included beside this file. No 10x overall or training speedup
relative to optimized TensorFlow is established.

### Subsequent Matched Python Benchmark

The complete upstream Python pipeline has now also been run on this Mac, with
documented compatibility fixes. Three-run medians were 9.278 s for Python's
SavedModel path, 8.926 s for Rust/Burn, and 1.843 s for Rust/ORT. See
[PYTHON_COMPARISON.md](PYTHON_COMPARISON.md) for settings, samples, output checks,
profiling, and the approximately 5x ORT result. This supersedes the earlier
absence of a full Python comparison, not the historical measurements above.

## Remaining Validation

The subsequent [6,400-window training comparison](LARGER_TRAINING_COMPARISON.md)
uses real reference labels rather than draft proxy labels. Both implementations
made 145 eval and 139 test window edits after one matched warm-start epoch;
Python took 122.972 s and native Rust 124.343 s before further optimization.
See [MODEL_OPTIMIZATION.md](MODEL_OPTIMIZATION.md) for subsequent same-data
encoding/transformer experiments. These measurements do not replace the
larger accuracy and Linux/GPU validation still required below.

Linux CPU CI has been provided but has not run in this workspace. A local
all-GPU-feature compilation attempt reached Burn's Tracel LLVM installer and
stopped because its default user-cache directory was outside sandbox write
permissions. That is not a completed Linux/CUDA build check.

Run the ignored GPU test and real-input CLI on the Linux NVIDIA host. Verify
CUDA forward/backward values, gradients, memory, TensorRT partition assignment,
FP16 effects, runtime/library compatibility, throughput including startup, and
read/quality equivalence. The device loss is a batched tensor wavefront, not an
already-benchmarked fused sparse CUDA kernel.

Independent held-out sequencing evaluation, quality calibration, chemistry
coverage, homopolymer/indel analysis, and multi-seed time-to-quality training
are still needed before making production or publication-level accuracy claims.
Window-wise edit metrics from the smoke fixture are not a substitute for that.
