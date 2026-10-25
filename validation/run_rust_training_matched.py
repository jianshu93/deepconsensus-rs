import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(os.environ.get("DC_BENCH_ROOT",str(Path(__file__).resolve().parent)))
DATA=Path(os.environ.get("DC_DATA_ROOT",str(ROOT/"larger-benchmark")))
parser=argparse.ArgumentParser()
parser.add_argument("--binary",type=Path,default=Path(__file__).resolve().parents[1]/"target/release/deepconsensus-rs")
parser.add_argument("--name",default="rust-trained")
parser.add_argument("--report",type=Path,default=DATA/"rust-training.json")
args=parser.parse_args()
binary=args.binary
command=[str(binary),"--cpu-threads","4","train","--data",str(DATA/"train.tfrecord.gz"),"--resume",str(DATA/"warm-start"),"--output",str(DATA/args.name),"--epochs","1","--batch-size","32","--learning-rate","0.000003","--end-learning-rate","0.000003","--weight-decay","0","--shuffle-buffer","1"]
before=time.perf_counter()
lines=[]
with args.report.with_suffix(".log").open("w") as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        lines.append(line.strip())
        if "step=" in line:
            fields=dict(part.split("=",1) for part in line.split())
            if int(fields["step"])%10==0 or fields["step"]=="1":
                print(line.strip(),flush=True)
        else:
            print(line.strip(),flush=True)
    code=process.wait()
    if code:
        raise RuntimeError(f"Rust training exited {code}")
wall=time.perf_counter()-before
last=dict(part.split("=",1) for part in lines[-1].split())
steps=[dict(part.split("=",1) for part in line.split()) for line in lines if "step=" in line]
report={"framework":"Burn 0.22.0 Flex CPU + Rayon analytic soft-alignment VJP","complete_process_seconds":wall,"epoch_seconds_including_input_and_checkpoint":float(last["elapsed_seconds"]),"mean_training_loss":float(last["mean_loss"]),"steps":[{"step":int(s["step"]),"loss":float(s["alignment_loss"])} for s in steps],"evaluation":{}}
for split in ["eval","test"]:
    before=time.perf_counter()
    result=subprocess.run([str(binary),"--cpu-threads","4","evaluate","--model",str(DATA/args.name),"--data",str(DATA/f"{split}.tfrecord.gz"),"--batch-size","64"],capture_output=True,text=True,check=True)
    metrics=json.loads(result.stdout)
    metrics["complete_process_seconds"]=time.perf_counter()-before
    report["evaluation"][split]=metrics
    print(json.dumps({"split":split,**metrics}),flush=True)
args.report.write_text(json.dumps(report,indent=2)+"\n")
