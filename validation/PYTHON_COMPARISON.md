# Matched Python and Rust CPU Comparison

Recorded on 2026-10-11 on the available macOS ARM64 CPU host. TensorFlow runs
successfully here. The previous evaluation had verified numerical behavior but
had not executed the complete upstream Python pipeline; this comparison now does.

## Processing Times

Three samples per mode, executed sequentially without competing benchmarks.
These medians cover BAM reading through FASTQ writing, **excluding model loading
and interpreter/import startup**. All ten molecules and 1,593 windows are processed.

| Implementation | Median | Three Samples |
| --- | --- | --- |
| Python/TensorFlow checkpoint, eager | 10.792 s | 10.785, 10.792, 10.950 |
| Python/TensorFlow SavedModel, serial preprocessing | 9.278 s | 9.149, 9.278, 9.326 |
| Python/TensorFlow SavedModel, four preprocessing processes | 10.932 s | 10.932, 10.922, 11.268 |
| Rust/Burn Flex | 8.926 s | 8.942, 8.926, 8.874 |
| Rust/ONNX Runtime CPU | 1.843 s | 1.831, 1.843, 2.092 |

Against the fastest measured Python mode, Rust/Burn is approximately **1.04x**,
essentially comparable on this fixture. Rust/ORT is approximately **5.03x** faster.
The latter comparison includes a different inference engine and pipeline design;
it is not evidence that changing programming language alone produces that gain.

Four Python preprocessing processes were slower for this small fixture. Their
worker startup/transfer costs are included; this does not establish their scaling
on large sequencing runs. Larger molecule blocks/batches, other thread settings,
other TensorFlow optimization settings, and larger data may change the result.

Rust's measured complete subprocess medians, including loading, were 8.944 s
(Burn) and 1.949 s (ORT). Python runs were repeated inside one interpreter with
the model reloaded each time; its interpreter/import cost was not measured as
three independent fresh processes. Consequently, do not use this table to claim
a matched cold-start speedup.

## Settings

- Same bundled human fixture at upstream r1.2 commit
  `979108ddbc0a950ea493071690507ac32c8ee9cb`.
- Same published v1.2 trained weights; Rust uses its validated native import or
  ONNX export, Python restores the original checkpoint.
- FP32 model tensors; 64 windows per model batch; two molecules per input block.
- Skipping disabled; insertion trimming five; minimum quality zero; minimum
  length one; maximum quality 93; published DC calibration and no CCS calibration.
- TensorFlow intra-op threads four, inter-op threads one; ORT intra-op threads
  four; Rayon threads four. Native BAM reading uses four threads; the upstream
  Python feeder defaults to one. The designs do not have identical thread pools.
- Python 3.12.14, TensorFlow 2.20.0, legacy `tf-keras` 2.20.1,
  `tf-models-official` 2.9.1, NumPy 2.5.4, pandas 2.3.3, pysam 0.24.1.
- Rust release build with Burn 0.22.0 / ORT rc.13; native ORT runtime 1.31.0.

## Upstream Compatibility

The original checkout remains unmodified. A benchmark import loader makes the
following modern-Python/TensorFlow compatibility adjustments:

1. Select legacy Keras through `TF_USE_LEGACY_KERAS=1`.
2. Replace `Read` dataclass empty ndarray defaults with default factories, because
   Python 3.12 rejects the old mutable defaults.
3. Omit the unused, training-only optimizer package import in `model_utils`.
4. Restrict official NLP package imports to the two original layer implementations
   actually used here, avoiding unrelated tokenizer/training dependencies.
5. Initialize Abseil flags in spawned macOS preprocessing workers.

The actual upstream BAM processing, spacing, window extraction, feature formatting,
transformer, checkpoint restoration, calibration, stitching, and FASTQ writing
functions are executed. There is no reconstructed replacement network in these
pipeline timings. The SavedModel is exported from that restored upstream model
with `tf.saved_model.save`, following its conversion script.

This is **not** the original pinned Python 3.8 / TensorFlow 2.9 environment.
Native framework/kernel changes and the compatibility shims must be considered
when interpreting performance. Empty-array factories also introduce small extra
constructor allocations. No claim about the original Linux wheel follows from
this Mac comparison.

## Output and Profile Checks

All nine Python runs produced the same ten sequences as Rust, totaling 109,143
bases. Compared with Rust, 30 base qualities differed by exactly one Phred point
in each Python run. Rust/Burn and Rust/ORT agree exactly. Model computation is
FP32, but the Rust host softmax/calibration uses FP64 while upstream prediction
probabilities are FP32; outputs are not asserted to be byte-identical to Python.

An additional instrumented SavedModel run took 9.109 s:

| Component | Time |
| --- | --- |
| Transformer forward calls | 7.907 s |
| Feature-to-model formatting/batch assembly | 0.209 s |
| Upstream preprocessing within molecule batches | 0.579 s |
| Stitching and FASTQ writing | 0.013 s |

The remaining time includes BAM/CIGAR preparation and other orchestration.
These components are from one profiling run, not separate median benchmarks.
Network execution dominates this CPU case. Thus the earlier approximately 60x
isolated Rust gap-spacing gain cannot be extrapolated to the full pipeline.

## Reproduction Artifacts

`matched-comparison.json` contains the aggregate samples, settings, output checks,
and limitations. The original Python and Rust sample reports and the profiling
report are included here. `benchmark_upstream.py` contains the compatibility loader
and upstream runner; `benchmark_rust_matched.py` runs the native CLIs serially.
They are developer benchmark tools, not application runtime dependencies.

Set `DC_BENCH_ROOT` to a scratch directory containing `deepconsensus/` (upstream
checkout), `upstream-deps/`, `validation-deps/`, `python-deps/`, the original
checkpoint files and `published-v1.2-params.json`, plus the imported
`published-model/` and exported `published-model.onnx`. Set `DC_RS_BINARY` to the
release binary. Install the documented packages into those dependency directories.
The native runner's dynamic-library path currently targets the validated macOS
ORT library; adjust it for another platform.

No Linux/GPU performance or training-throughput comparison has been completed.
**The measured result is about 5x for Rust/ORT on this CPU fixture, not 10x for
Rust generally.**
