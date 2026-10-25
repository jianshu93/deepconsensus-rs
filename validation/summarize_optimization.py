"""Aggregate recorded reports without changing measurements."""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


native = read("model-optimization.json")
ort = read("ort-fusion-final.json")
python = read("python-forward.json")
train_before = read("rust-training.json")
train_after = read("rust-optimized-training.json")
train_python = read("python-training.json")

forward = {}
for name, result in native["variants"].items():
    if name.startswith("ort-"):
        continue
    forward[name] = {"seconds": median(s["seconds"] for s in result["samples"]),
        "encoding_seconds": median(s["encoding_seconds"] for s in result["samples"]),
        "changed_sequences": result["changed_sequences"],
        "changed_quality_windows":result["changed_quality_windows"],
        "max_logit_difference":result["max_absolute_logit_difference"]}
for name, result in ort["variants"].items():
    forward[name] = {"seconds":median(s["seconds"] for s in result["samples"]),
        "changed_sequences":result["changed_sequences"],
        "changed_quality_windows":result["changed_quality_windows"],
        "max_logit_difference":result["max_absolute_logit_difference"]}
forward["python"] = {"seconds":median(python["samples_seconds"]),
    "changed_argmax_columns":python["changed_argmax_columns"],
    "max_probability_difference":python["max_probability_difference"]}

paired = [abs(a["loss"] - b["loss"]) for a, b in zip(train_before["steps"], train_after["steps"], strict=True)]
paired_python = [abs(a["loss"] - b["loss"]) for a, b in zip(train_python["steps"], train_after["steps"], strict=True)]
training = {"python_seconds":train_python["epoch_seconds_including_input_and_checkpoint"],
    "native_before_seconds":train_before["epoch_seconds_including_input_and_checkpoint"],
    "native_after_seconds":train_after["epoch_seconds_including_input_and_checkpoint"],
    "native_before_after_max_step_loss_difference":max(paired),
    "native_before_after_mean_step_loss_difference":mean(paired),
    "python_vs_native_after_max_step_loss_difference":max(paired_python),
    "python_vs_native_after_mean_step_loss_difference":mean(paired_python),
    "after_evaluation":train_after["evaluation"],
    "limitation":"Single full training run per implementation/configuration; timing differences do not establish causal or repeatable improvement"}
summary = {"unique_training_windows":6400,"forward":forward,"training":training,
    "encoding_speedup":forward["reference"]["encoding_seconds"]/forward["selected"]["encoding_seconds"],
    "python_divided_by_original_ort_forward":forward["python"]["seconds"]/forward["ort-reference"]["seconds"],
    "experimental_ort_percent_slower":100*(forward["ort-optimized"]["seconds"]/forward["ort-reference"]["seconds"]-1),
    "native_training_observed_percent_shorter":100*(1-training["native_after_seconds"]/training["native_before_seconds"])}
(ROOT/"optimization-comparison.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
