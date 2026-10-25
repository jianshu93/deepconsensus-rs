"""Serial forward-only comparison on the exact native benchmark TFRecords."""
import argparse
import json
from pathlib import Path
import time

import benchmark_upstream as compat
import train_python_matched as data
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, default=data.DATA / "python-forward.json")
    args = parser.parse_args()
    tf = compat.tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    before = time.perf_counter()
    model = tf.saved_model.load(str(compat.ROOT / "upstream-saved-model"))
    forward = model.signatures["serving_default"]
    input_name = next(iter(forward.structured_input_signature[1]))
    setup = time.perf_counter() - before
    before = time.perf_counter()
    examples = list(data.records(data.DATA / "train.tfrecord.gz"))
    read = time.perf_counter() - before
    first = np.stack([x for x, _ in examples[:args.batch_size]])
    forward(**{input_name:tf.convert_to_tensor(first)})
    baseline = np.fromfile(data.DATA / "reference-logits.f32", dtype="<f4").reshape(6400, 100, 5)
    calls = np.argmax(baseline, axis=-1)
    samples = []
    for _ in range(args.repeats):
        outputs = []
        timer = time.perf_counter()
        for start in range(0, len(examples), args.batch_size):
            rows = np.stack([x for x, _ in examples[start:start + args.batch_size]])
            result = forward(**{input_name:tf.convert_to_tensor(rows)})
            probabilities = next(iter(result.values())).numpy()
            outputs.append(probabilities)
        predictions = np.concatenate(outputs, axis=0)
        elapsed = time.perf_counter() - timer
        samples.append(elapsed)
        print(json.dumps({"windows":len(examples), "seconds":elapsed}), flush=True)
    # Upstream exports probabilities, whereas the native graph returns logits.
    native_probs = np.exp(baseline - np.max(baseline, axis=-1, keepdims=True))
    native_probs /= np.sum(native_probs, axis=-1, keepdims=True)
    report = {"windows":len(examples),"batch_size":args.batch_size,"threads":4,
        "setup_seconds":setup,"read_seconds":read,"samples_seconds":samples,
        "max_probability_difference":float(np.max(np.abs(predictions-native_probs))),
        "changed_argmax_columns":int(np.sum(np.argmax(predictions,axis=-1)!=calls)),
        "timed_scope":"Batch stacking, original SavedModel inference, output collection; warmup, file reading and model loading excluded"}
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
