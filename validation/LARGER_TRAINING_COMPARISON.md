# Matched Training on 6,400 Unique Windows

Measured on the available macOS 15.7.9 ARM64 CPU host on 2026-10-11.
This is a correctness and short-run performance experiment, not a reproduction
of the publication's full training or a Linux/NVIDIA benchmark.

## Data and Reference

The public [DeepConsensus training tutorial](https://github.com/google/deepconsensus/blob/r1.2/docs/train_model.md)
provides real, precomputed truth-labeled TFRecords at
`gs://brain-genomics-public/research/deepconsensus/training-tutorial/v1.2/`.
The full training split has 500 shards and approximately 1.014 TB compressed;
downloading it is unnecessary for this controlled small experiment.

| Split | Unique Windows | Molecules | Non-Gap Truth Bases | Compressed Bytes |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,400 | 40 | 492,324 | 26,098,066 |
| Eval | 1,600 | 6 | 122,929 | 5,974,137 |
| Test | 800 | 3 | 61,656 | 3,252,431 |

Training is 4.02 times the previous 1,593-window inference fixture, and four
times its molecule count. Every selected `(name, window_pos)` is unique.
The three molecule sets are mutually disjoint. Rust verified all TFRecord CRCs.

Only bounded gzip HTTP prefixes were fetched from shard 00000 in each split.
Complete original TFRecord frames were extracted into valid, deterministic gzip
files; truncated source prefixes are not distributed. For training, the first
40 complete molecules were collected, up to 160 windows per molecule were
sampled, shortages were filled from remaining unique windows, and all 6,400
were shuffled with seed 42. Evaluation/test retain the first complete records
from their separate official shards. This is not population-random sampling.

The TFRecords already contain aligned truth labels, so both programs can train
and evaluate without downloading the approximately 3.09 GB CHM13 reference
FASTA. The reference remains publicly available in the same tutorial directory;
it was not downloaded here. `chromosome-splits.txt` is a chromosome-name mapping,
not evidence establishing the genomic independence of these selected records.
The source metadata reports producer version 1.1.0 despite the v1.2 directory;
the actual 86-by-100 feature shape matches the model's expected input.

The dataset archive includes the original frame hashes, gzip SHA-256 hashes,
molecule IDs, and source shard names in `dataset-manifest.json`.

## Matched Recipe

- Warm-start the same published v1.2 checkpoint: five layers, hidden size 280,
  two attention heads, FFN width 2048, 100 alignment columns, 20 feature passes.
- One epoch, batch size 32, exactly 200 updates, identical pre-shuffled records.
- Constant learning rate 3e-6; Adam beta1 0.9, beta2 0.997, epsilon 1e-9.
- Native AdamW has zero weight decay; TensorFlow uses legacy Adam.
- All dropout is zero; no in-loop reshuffle (`--shuffle-buffer 1` in Rust).
- Dense/unbanded alignment loss: regularization 0.1, deletion cost 10.
- Four CPU model/Rayon threads; TensorFlow inter-op threads one.
- FP32 model tensors; native CPU soft-DP accumulation is FP64, TensorFlow FP32.

Python executes the actual upstream r1.2 model and alignment loss through a
controlled `tf.function` loop. The compatibility loader is described in
[PYTHON_COMPARISON.md](PYTHON_COMPARISON.md). This common optimizer experiment
does not use the original default LAMB/distillation trainer or its additional
per-step edit/identity metrics. Rust executes the native training CLI.

The full runs were sequential, with no concurrent performance experiments.
Two-step smoke checks were used only for correctness, never as timing evidence.

## Before Further Optimization

| Measurement | Python / TensorFlow 2.20 | Rust / Burn 0.22 |
| --- | ---: | ---: |
| Epoch, input reading through checkpoint saving | 122.972 s | 124.343 s |
| Mean training alignment loss | 0.471611347 | 0.471611 |
| Eval alignment loss | 0.289277096 | 0.289277020 |
| Eval edit errors / truth bases | 145 / 122,929 | 145 / 122,929 |
| Test alignment loss | 0.609978153 | 0.609977569 |
| Test edit errors / truth bases | 139 / 61,656 | 139 / 61,656 |

Python graph tracing took 2.483 s and is reported separately from epoch time.
Its median steady training step was 0.601 s. The complete native training
subprocess took 124.672 s; process/model startup is excluded from epoch time.

Across all 200 corresponding steps, maximum absolute loss difference was
8.52e-6 and mean absolute difference was 8.35e-7. The loss trajectories and
held-out edit counts support numerical correctness of the port.

CPU training is effectively tied: Rust was approximately 1.1% slower in this
single paired run. This is not evidence of a meaningful winner or a 10x gain.
Before fine-tuning, the published model made 147 eval and 140 test edits;
afterward both implementations made 145 and 139. Differences of two and one
errors on these tiny held-out sets do not establish an accuracy improvement.

See `training-comparison.json` and the individual reports/log for raw results.
Subsequent implementation optimization is reported separately in
[MODEL_OPTIMIZATION.md](MODEL_OPTIMIZATION.md).

The selected encoding change subsequently completed the same native epoch in
119.372 s, with unchanged held-out edit counts. This is a modest observed
single-run difference, not a demonstrated multi-run or GPU training speedup.

## Reproduce Native Training

Extract the data archive as `data/`. Start from the published native checkpoint
and set all three dropout fields in its copied `params.json` to zero. Keep this
warm-start directory free of optimizer and training-state files for a fresh run.

```sh
cargo build --release --locked
target/release/deepconsensus-rs --cpu-threads 4 train \
  --data data/train.tfrecord.gz --resume warm-start --output trained \
  --epochs 1 --batch-size 32 --shuffle-buffer 1 \
  --learning-rate 0.000003 --end-learning-rate 0.000003 --weight-decay 0
target/release/deepconsensus-rs --cpu-threads 4 evaluate \
  --model trained --data data/eval.tfrecord.gz --batch-size 64
target/release/deepconsensus-rs --cpu-threads 4 evaluate \
  --model trained --data data/test.tfrecord.gz --batch-size 64
```

## Limits

This is a warm-start single epoch, not de novo convergence or multi-seed
time-to-quality validation. Evaluation and test are separate from fine-tuning,
but overlap with the published checkpoint's original pretraining is unknown.
Only six evaluation and three test molecules were sampled. All accuracy metrics
are window-level, not fully stitched molecule accuracy, yield, or calibrated
quality. The TFRecord benchmark does not measure raw BAM gap processing.
Linux CPU, CUDA training, FP16, and TensorRT still require hardware validation.
