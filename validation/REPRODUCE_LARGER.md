# Reproduce the Larger Benchmarks

The supplied `deepconsensus-training-subset-4x.zip` contains complete original
TFRecords, not generated or repeated copies. Extract it into `data/`.
Use the v1.2 checkpoint and the zero-dropout warm-start configuration described
in [LARGER_TRAINING_COMPARISON.md](LARGER_TRAINING_COMPARISON.md).

## Native CPU

```sh
cargo build --release --locked --features onnx --example benchmark_model
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
target/release/examples/benchmark_model warm-start data/train.tfrecord.gz \
  model-optimization.json --batch-size 64 --threads 4 --repeats 3 \
  --variants reference,selected,ort-reference,ort-optimized \
  --dump-logits reference-logits.f32
```

The example checks all record CRCs and uniqueness before running. File parsing,
model/session setup, warmup, decoding, and alignment metrics are outside the
forward timer. CPU batch assembly and output collection are inside it.
Embedding/condenser and transformer/final-classifier times are reported
separately for Burn. ORT is timed as a whole graph.

To counterbalance just the ORT graph experiments, reuse the complete reference
logits and use the schedule from the final comparison:

```sh
target/release/examples/benchmark_model warm-start data/train.tfrecord.gz \
  ort-fusion-final.json --batch-size 64 --threads 4 --repeats 2 \
  --reference-logits reference-logits.f32 \
  --variants ort-reference,ort-optimized,ort-optimized,ort-reference
```

The default native model folds embedding scaling and zero-row masking, while
leaving Q/K/V/attention/matrix-layout experiments disabled. The default ONNX
graph is the original graph; explicit fusion was slower on the tested CPU.
`--variants combined` requests all native experimental options in this example,
but still requires a preceding `reference` or an appropriate reference file.
The example is CPU-specific; do not treat its stage timers as asynchronous
CUDA timings. Use the ignored GPU test and the application CLI on Linux.

## Python Reference

The scripts are validation tooling, not application dependencies. Use a scratch
directory containing the pinned upstream r1.2 checkout, original `checkpoint.*`
and `params.json`, and the package directories described in
[PYTHON_COMPARISON.md](PYTHON_COMPARISON.md). Keep the upstream checkout unmodified.

```sh
export DC_BENCH_ROOT=/path/to/reference-work
export DC_DATA_ROOT=/path/to/data
export PYTHONPATH="$DC_BENCH_ROOT/validation-deps:$PYTHONPATH"
python validation/benchmark_upstream.py --export
python validation/train_python_matched.py --report python-training.json \
  --output python-trained
python validation/benchmark_python_windows.py --report python-forward.json
```

The compatibility loader resolves old library imports without rewriting the
network or alignment loss. `benchmark_python_windows.py` requires the exported
upstream SavedModel at `$DC_BENCH_ROOT/upstream-saved-model` and the complete
`reference-logits.f32` in `$DC_DATA_ROOT`. It uses the same 6,400 input records,
four intra-op threads, one inter-op thread, and batch size 64 by default.

`extract_larger_dataset.py` documents the original extraction/sampling and
expects previously fetched HTTP prefixes in `DC_DATA_ROOT`. Re-extraction is
not needed when using the distributed dataset. The prefix files and full
terabyte-scale source are intentionally excluded.

## Interpretation

Forward-only TFRecord timing is not BAM-to-FASTQ timing, and training epoch
timing includes backward/optimizer work that inference does not. Do not mix
those denominators. Repeat on otherwise-idle, fixed-thread hardware and compare
losses, decoded bases, qualities, and held-out metrics before claiming speed.
