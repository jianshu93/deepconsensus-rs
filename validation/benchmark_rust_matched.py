"""Sequential matched-settings CLI timings; do not run alongside another benchmark."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(os.environ.get("DC_BENCH_ROOT",str(Path(__file__).resolve().parent)))
binary=Path(os.environ.get("DC_RS_BINARY",str(ROOT.parent/"outputs/deepconsensus-rs/target/release/deepconsensus-rs")))
fixture=ROOT/"deepconsensus/deepconsensus/testdata/human_1m"
env=os.environ.copy()
env["ORT_DYLIB_PATH"]=str(ROOT/"validation-deps/onnxruntime/capi/libonnxruntime.1.31.0.dylib")
report={}
for mode in ["burn","ort"]:
    samples=[]
    for repeat in range(3):
        output=ROOT/"python-results"/f"rust-{mode}-{repeat}.fastq"
        model=ROOT/"published-model.onnx" if mode=="ort" else ROOT/"published-model"
        command=[str(binary),"--cpu-threads","4","inference","--subreads",str(fixture/"subreads_to_ccs.bam"),"--ccs",str(fixture/"ccs.bam"),"--model",str(model),"--output",str(output),"--limit","10","--batch-size","64","--block-molecules","2","--threads","4","--skip-above","0","--min-quality","0","--min-length","1","--max-quality","93"]
        if mode=="ort":
            command.extend(["--config",str(ROOT/"published-model/params.json"),"--provider","cpu"])
        before=time.perf_counter()
        result=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
        wall=time.perf_counter()-before
        sample=json.loads(result.stdout)
        sample["process_seconds_including_loading"]=wall
        sample["output"]=str(output)
        samples.append(sample)
        print(json.dumps({"mode":mode,"repeat":repeat,"processing_seconds":sample["total_seconds"],"process_seconds_including_loading":wall}),flush=True)
    report[mode]=samples
(ROOT/"rust-matched.json").write_text(json.dumps(report,indent=2)+"\n")
