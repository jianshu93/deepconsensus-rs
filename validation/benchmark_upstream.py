"""Run original DeepConsensus inference with narrowly scoped import compatibility fixes."""
import argparse
import ast
import contextlib
import dataclasses
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

START=time.perf_counter()
ROOT=Path(os.environ.get("DC_BENCH_ROOT",str(Path(__file__).resolve().parent)))
os.environ.setdefault("TF_USE_LEGACY_KERAS","1")
os.environ.setdefault("KERAS_HOME",str(ROOT/"keras-cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
sys.path[:0]=[str(ROOT/"upstream-deps"),str(ROOT/"validation-deps"),str(ROOT/"python-deps"),str(ROOT/"deepconsensus")]


class CompatLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname):
        tree=ast.parse(self.get_data(self.path),filename=self.path)
        if fullname=="deepconsensus.preprocess.pre_lib":
            read=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Read")
            for node in read.body:
                if isinstance(node,ast.AnnAssign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="np" and node.value.func.attr=="empty":
                    factory=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=node.value)
                    node.value=ast.Call(func=ast.Attribute(value=ast.Name(id="dataclasses",ctx=ast.Load()),attr="field",ctx=ast.Load()),args=[],keywords=[ast.keyword(arg="default_factory",value=factory)])
        elif fullname=="deepconsensus.models.model_utils":
            # The imported training-only optimizer package requires obsolete add-ons.
            tree.body=[node for node in tree.body if not(isinstance(node,ast.ImportFrom) and node.module=="official.modeling" and any(x.name=="optimization" for x in node.names))]
        elif fullname=="official.nlp.modeling":
            tree.body=[node for node in tree.body if not isinstance(node,ast.ImportFrom) or any(x.name=="layers" for x in node.names)]
        elif fullname=="official.nlp.modeling.layers":
            tree.body=[node for node in tree.body if not isinstance(node,ast.ImportFrom) or any(x.name in {"OnDeviceEmbedding","RelativePositionEmbedding"} for x in node.names)]
            for node in tree.body:
                if isinstance(node,ast.ImportFrom):
                    node.names=[x for x in node.names if x.name in {"OnDeviceEmbedding","RelativePositionEmbedding"}]
        return compile(ast.fix_missing_locations(tree),self.path,"exec")


class CompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname not in {"deepconsensus.preprocess.pre_lib","deepconsensus.models.model_utils","official.nlp.modeling","official.nlp.modeling.layers"}:
            return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is not None and spec.origin is not None:
            spec.loader=CompatLoader(fullname,spec.origin)
        return spec


sys.meta_path.insert(0,CompatFinder())
import tensorflow as tf
from deepconsensus.inference import quick_inference as qi
from deepconsensus.models import model_utils

def configure_worker():
    if not qi.FLAGS.is_parsed():
        qi.FLAGS(["preprocess_worker"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["checkpoint","saved"],default="checkpoint")
    parser.add_argument("--cpus",type=int,default=0)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--batch-zmws",type=int,default=2)
    parser.add_argument("--repetitions",type=int,default=1)
    parser.add_argument("--export",action="store_true")
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--limit",type=int,default=10)
    parser.add_argument("--report",type=Path,default=ROOT/"python-benchmark.json")
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([],"GPU")
    fixture=ROOT/"deepconsensus/deepconsensus/testdata/human_1m"
    shutil.copyfile(ROOT/"published-v1.2-params.json",ROOT/"params.json")
    output=ROOT/"python-results"
    output.mkdir(exist_ok=True)
    saved=ROOT/"upstream-saved-model"
    flags=["benchmark",f"--subreads_to_ccs={fixture/'subreads_to_ccs.bam'}",f"--ccs_bam={fixture/'ccs.bam'}",f"--checkpoint={saved if args.mode=='saved' else ROOT/'checkpoint'}",f"--output={output/'initial.fastq'}",f"--batch_size={args.batch_size}",f"--batch_zmws={args.batch_zmws}",f"--cpus={args.cpus}",f"--limit={args.limit}","--skip_windows_above=0","--min_quality=0","--min_length=1","--ins_trim=5","--max_base_quality=93"]
    qi.FLAGS(flags)
    if args.cpus>0:
        executor=qi.concurrent.futures.ProcessPoolExecutor
        def macos_executor(*a,**kw):
            kw["initializer"]=configure_worker
            return executor(*a,**kw)
        qi.concurrent.futures.ProcessPoolExecutor=macos_executor
    if args.export:
        from deepconsensus.quality_calibration.calibration_lib import parse_calibration_string
        p=model_utils.read_params_from_json(str(ROOT/"checkpoint"))
        options=qi.InferenceOptions(max_length=100,example_height=86,max_passes=20,min_quality=0,min_length=1,batch_size=args.batch_size,use_ccs_bq=True,cpus=0,skip_windows_above=0,use_saved_model=False,max_base_quality=93,dc_calibration_values=parse_calibration_string(p.dc_calibration),ccs_calibration_values=parse_calibration_string("skip"))
        with contextlib.redirect_stdout(io.StringIO()):
            model,params=qi.initialize_model(str(ROOT/"checkpoint"),p,options)
        tf.saved_model.save(model,str(saved))
        shutil.copyfile(ROOT/"published-v1.2-params.json",saved/"params.json")
        print(json.dumps({"exported":str(saved),"variables":len(model.variables)}))
        return
    setup=[]
    profile={"formatting_seconds":0.0,"network_seconds":0.0}
    if args.profile:
        original_batches=qi.batch_examples
        def observed_batches(*a,**kw):
            batches=iter(original_batches(*a,**kw))
            while True:
                before=time.perf_counter()
                try:
                    value=next(batches)
                except StopIteration:
                    return
                profile["formatting_seconds"]+=time.perf_counter()-before
                yield value
        qi.batch_examples=observed_batches
    def observed_call(function):
        def call(*a,**kw):
            before=time.perf_counter()
            value=function(*a,**kw)
            profile["network_seconds"]+=time.perf_counter()-before
            return value
        return call
    class ObservedModel:
        def __init__(self,model):
            if args.mode=="saved":
                self.signatures={"serving_default":observed_call(model.signatures["serving_default"])}
            else:
                self.predict=observed_call(model.predict)
    original_init=qi.initialize_model
    def observed_init(*a,**kw):
        before=time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result=original_init(*a,**kw)
        setup.append(time.perf_counter()-before)
        if args.profile:
            result=(ObservedModel(result[0]),result[1])
        return result
    qi.initialize_model=observed_init
    samples=[]
    for repeat in range(args.repetitions):
        qi.timing.clear()
        profile.update(formatting_seconds=0.0,network_seconds=0.0)
        path=output/f"{args.mode}-p{args.cpus}-{repeat}.fastq"
        qi.FLAGS.output=str(path)
        before=time.perf_counter()
        counts=qi.run()
        elapsed=time.perf_counter()-before
        sample={"run_seconds":elapsed,"setup_seconds":setup[-1],"processing_seconds":elapsed-setup[-1],"outcomes":dataclasses.asdict(counts),"stages":list(qi.timing),"output":str(path)}
        if args.profile:
            sample["profile"]=dict(profile)
        samples.append(sample)
        print(json.dumps({k:v for k,v in sample.items() if k!="stages"}),flush=True)
    report={"tensorflow":tf.__version__,"python":sys.version,"mode":args.mode,"preprocessing_processes":args.cpus,"tf_intra_threads":args.threads,"tf_inter_threads":1,"batch_size":args.batch_size,"batch_molecules":args.batch_zmws,"molecules":args.limit,"skip_above":0,"samples":samples,"process_seconds_including_import":time.perf_counter()-START,"compatibility":["tf-keras 2.20.1 legacy API","Read empty ndarray defaults changed to factories by import loader","unused training optimizer import omitted","official NLP package imports restricted to the two original layer implementations used by this model","macOS spawn worker initializer parses default flags before upstream preprocessing"]}
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
