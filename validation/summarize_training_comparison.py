import json
import os
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get("DC_DATA_ROOT",str(ROOT)))
manifest=json.loads((DATA/"dataset-manifest.json").read_text())
python=json.loads((DATA/"python-training.json").read_text())
rust=json.loads((DATA/"rust-training.json").read_text())
baseline=json.loads((DATA/"python-baseline.json").read_text())
delta=[abs(a["loss"]-b["loss"]) for a,b in zip(python["steps"],rust["steps"])]
assert len(delta)==200
out={"dataset":{"source":manifest["source"],"splits":{s:{k:v for k,v in d.items() if k in {"examples","molecules","bytes","sha256","uncompressed_records_sha256"}} for s,d in manifest["splits"].items()},"molecule_overlap":False,"training_size_factor_vs_earlier_10_molecules":4.0,"training_window_factor_vs_earlier_1593_windows":6400/1593},"recipe":python["training_recipe"],"runs":{"python":{"epoch_seconds":python["epoch_seconds_including_input_and_checkpoint"],"compile_seconds":python["compile_seconds"],"steady_step_median_seconds":statistics.median(s["seconds"] for s in python["steps"][1:]),"mean_training_loss":statistics.mean(s["loss"] for s in python["steps"]),"evaluation":python["evaluation"]},"rust":{"epoch_seconds":rust["epoch_seconds_including_input_and_checkpoint"],"complete_process_seconds":rust["complete_process_seconds"],"mean_training_loss":rust["mean_training_loss"],"evaluation":rust["evaluation"]}},"python_epoch_divided_by_rust_epoch":python["epoch_seconds_including_input_and_checkpoint"]/rust["epoch_seconds_including_input_and_checkpoint"],"max_paired_training_loss_difference":max(delta),"mean_paired_training_loss_difference":statistics.mean(delta),"held_out_baseline":baseline["evaluation"],"limitations":["Single sequential run per framework; approximately one-percent timing difference is not evidence of a winner","Warm-start one epoch on 6400 examples, not training to convergence","Common Adam/zero-decay/dropout-free recipe; not the upstream default LAMB/distillation trainer or per-step identity metrics","Model tensors F32; Rust CPU dynamic program accumulates in F64, TensorFlow in F32","Separate fine-tuning molecules/splits, but published-checkpoint pretraining overlap is unknown","Prefix-based molecule selection and only six evaluation/three test molecules","Window-level truth metrics; not an end-to-end molecule accuracy or calibrated quality study","CPU macOS ARM64 only; no Linux GPU training measurement"]}
assert all(python["evaluation"][s]["edit_errors"]==rust["evaluation"][s]["edit_errors"] for s in ["eval","test"])
(DATA/"training-comparison.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
