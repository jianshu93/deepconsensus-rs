# Encoding and Transformer Optimization

CPU experiment on macOS ARM64, using every one of the 6,400 unique labeled
training windows documented in [LARGER_TRAINING_COMPARISON.md](LARGER_TRAINING_COMPARISON.md).
This is not the earlier ten-molecule BAM timing and does not measure raw gap
construction. The real TFRecords already contain alignment features.

## Changes Tested

1. Fold embedding scaling and gap masking into the small lookup table before
   gathering, instead of multiplying and masking the much larger gathered
   activations. Preserve integer clipping/truncation, CCS-quality shift, row
   order, parameter names, and gradients. This is enabled by default.
2. Combine the three Q/K/V weight matrices into one projection, retaining the
   original trainable parameters and splitting the output afterward.
3. Use Burn's attention primitive for inference. Keep explicit softmax and
   attention dropout for training; there is no dropout argument in the primitive.
4. Explicitly flatten batch/position axes around affine matrices to test CPU
   matrix scheduling instead of relying on the backend's shape policy.
5. Export an alternative ONNX graph with precombined Q/K/V weights, 2D Gemm
   affines, and ReLU adjacent to its Gemm so the runtime can fuse activation.

Items 2-4 did not establish a native CPU throughput gain. They remain explicit
experimental `ExecutionOptions` and are not enabled by default. Item 5 was
slower in the final counterbalanced CPU experiment, so the original ONNX graph
remains the default. The alternative is exposed by `--experimental-fusions`.
No attention connectivity, precision, pass count, skip policy, or alignment
objective was changed to manufacture a speed result.

## Controlled Forward Timing

Four CPU model/Rayon threads, TensorFlow inter-op one, batch size 64, FP32
published v1.2 weights, and the identical 6,400 windows. The timer includes
batch construction, network execution, and output collection. File parsing,
model/session loading, warmup, alignment metrics, and decoding are excluded.
Python executes the original exported SavedModel, not a reconstructed network.

| Path | Median Forward Time | Encoding + Condenser |
| --- | ---: | ---: |
| Python / TensorFlow SavedModel | 29.602 s | Not separately measured |
| Rust / Burn, before folding | 35.210 s | 1.173 s |
| Rust / Burn, folded embeddings | 35.590 s | 1.063 s |
| Rust / ORT, original graph | 6.942 s | Included in graph |
| Rust / ORT, experimental fusion graph | 7.687 s | Included in graph |

Burn and Python values are medians of three sequential runs. Final ORT values
are medians of four full passes per graph, in a reference/alternative/
alternative/reference schedule with two passes per placement. No competing
build, training, or benchmark was run during these timed passes.

Embedding/condenser time improved by approximately 9.3%, or 1.10x. The overall
native Burn time did not improve: the approximately 1.1% difference is within
the observed run variation. Approximately 97% of reference forward time was
in the transformer/final-classifier stage, so a small encoding saving cannot
produce a large whole-network gain.

The experimental ONNX graph was approximately 10.7% slower than the original
in this counterbalanced test. Explicit fusion is not automatically a better
schedule for short windows on every backend. The original runtime already
optimizes the graph; the result is evidence against enabling the candidate by
default, not a claim that fusion can never help a GPU.

The existing Rust/ORT path is approximately 4.26x faster than upstream Python
SavedModel forward execution on this larger subset. That is a runtime/kernel
comparison, not an isolated Rust-language speedup. Native Burn is slower than
the optimized TensorFlow reference on this particular forward workload.

## Output Checks

- Native reference versus folded embeddings: all 3.2 million logits were
  bit-identical, all 640,000 base/gap decisions agreed, every decoded window
  sequence and every rounded calibrated quality agreed.
- Python SavedModel versus native reference: maximum probability difference
  5.72e-6; all 640,000 base/gap decisions agreed.
- Both ONNX graphs versus native reference: maximum logit difference 4.39e-5,
  no changed base/gap calls or decoded sequences, nine windows with changed
  rounded quality arrays. This is nine affected windows, not necessarily nine
  changed bases. A small floating-point difference can cross a Phred rounding
  boundary; qualities are not claimed bit-identical to Burn.
- A separate direct graph-to-graph check on all 6,400 windows found the original
  and experimental ONNX outputs bit-identical: zero changed logits, sequences,
  or qualities. See `ort-output-parity.json`; that extra run is not included
  in the counterbalanced timing medians above.
- All native/ONNX paths made 896 window-level edit errors against 492,324
  non-gap truth bases on this training subset. This is a correctness check,
  not held-out accuracy or a claim of improved generalization.

The native gradient regression uses nonzero ReZero branches, gaps, clipped and
fractional IDs, and CCS quality indexing. Forward values and selected parameter
gradients agree within 1e-5. CPU tests also exercise training, resume, native
checkpoint reload, original/experimental graph execution, and real BAM fixtures.

## Training Follow-Through

The same 200-step, 6,400-window matched recipe was rerun with folded embeddings:

| Epoch | Seconds | Eval Edits | Test Edits |
| --- | ---: | ---: | ---: |
| Python / TensorFlow | 122.972 | 145 | 139 |
| Native Rust before folding | 124.343 | 145 | 139 |
| Native Rust after folding | 119.372 | 145 | 139 |

The new native epoch was about 4.0% shorter than the previous native epoch in
this single-run comparison. This is an observed modest change, not proof of
causal or repeatable acceleration; there was one full training run per
configuration. It does not establish 10x faster training or convergence.
Maximum native before/after step-loss difference was 8.0e-6 using the CLI's
six-decimal loss reports; maximum Python/new-native difference was 6.31e-6.
Small gradient accumulation-order differences prevent a claim of identical
trained tensors, despite identical held-out edit counts.

Full epoch, step-loss, and held-out results are stored in
`rust-optimized-training.json` and `optimization-comparison.json`.
Do not equate forward-only ORT acceleration with a training acceleration:
the ORT path here is inference-only.

## Reproduction and Hardware Limits

See [REPRODUCE_LARGER.md](REPRODUCE_LARGER.md) and `examples/benchmark_model.rs`.
Raw samples are `model-optimization.json`, `ort-fusion-final.json`, and
`python-forward.json`. The initial ablation is exploratory; its first reference
pass overlapped the end of compilation and is not used for controlled timing.

The ignored Linux CUDA test now exercises experimental attention with nonzero
ReZero branches, but it has not been executed on NVIDIA hardware. The benchmark
example's detailed stage timers are CPU-specific. Validate GPU kernel selection
for the actual 100-column, 140-dimension head shape, forward/backward precision,
memory, output qualities, and TensorRT fallback before promoting a GPU option.
No Linux/GPU speed or 10x training claim is supported by these CPU measurements.
