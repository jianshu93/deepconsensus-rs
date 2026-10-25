"""Extract complete original TFRecords from bounded HTTP gzip prefixes."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys

ROOT=Path(os.environ.get("DC_BENCH_ROOT",str(Path(__file__).resolve().parent)))
sys.path.insert(0,str(ROOT/"validation-deps"))
import tensorflow as tf
import numpy as np

OUT=Path(os.environ.get("DC_DATA_ROOT",str(ROOT/"larger-benchmark")))
manifest={"source":"https://storage.googleapis.com/brain-genomics-public/research/deepconsensus/training-tutorial/v1.2/","sampling":"Training: collect first 40 complete molecules in shard 00000, sample up to 160 windows per molecule and fill to 6400 without duplication, then reorder with seed 42. Evaluation/test: first complete records from their separate official shards. Original record payloads and CRCs retained. Prefix sampling is not population-random.","splits":{}}
sets={}
for split,target in [("train",6400),("eval",1600),("test",800)]:
    frames=[]
    names=set()
    positions=set()
    groups={}
    current=None
    with gzip.open(OUT/f"{split}-prefix.gz","rb") as source:
        while split=="train" or len(frames)<target:
            header=source.read(12)
            if len(header)!=12:
                raise RuntimeError(f"{split}: prefix ends before {target} complete records")
            length=struct.unpack("<Q",header[:8])[0]
            payload=source.read(length)
            tail=source.read(4)
            if len(payload)!=length or len(tail)!=4:
                raise RuntimeError("incomplete TFRecord; fetch a larger prefix")
            e=tf.train.Example.FromString(payload).features.feature
            shape=list(e["subreads/shape"].int64_list.value)
            assert shape==[86,100,1],shape
            assert len(e["subreads/encoded"].bytes_list.value[0])==34400
            assert len(e["label/encoded"].bytes_list.value[0])==400
            name=e["name"].bytes_list.value[0].decode()
            if split=="train" and name!=current:
                if len(groups)==40:
                    break
                assert name not in groups,"non-grouped training shard"
                groups[name]=[]
                current=name
            pos=e["window_pos"].int64_list.value[0]
            assert (name,pos) not in positions,"duplicate example"
            positions.add((name,pos))
            names.add(name)
            label=np.frombuffer(e["label/encoded"].bytes_list.value[0],dtype="<f4")
            assert np.all(np.isin(label,np.arange(5)))
            frames.append(header+payload+tail)
            if split=="train":
                groups[name].append(header+payload+tail)
    if split=="train":
        rng=random.Random(42)
        selected=[]
        remainder=[]
        for group in groups.values():
            rng.shuffle(group)
            selected.extend(group[:160])
            remainder.extend(group[160:])
        rng.shuffle(remainder)
        selected.extend(remainder[:target-len(selected)])
        assert len(selected)==target
        frames=selected
        rng.shuffle(frames)
    path=OUT/f"{split}.tfrecord.gz"
    raw_hash=hashlib.sha256()
    with path.open("wb") as output_file:
        with gzip.GzipFile(filename="",fileobj=output_file,mode="wb",compresslevel=1,mtime=0) as target_file:
            for frame in frames:
                target_file.write(frame)
                raw_hash.update(frame)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sets[split]=names
    manifest["splits"][split]={"examples":len(frames),"molecules":len(names),"bytes":path.stat().st_size,"sha256":digest,"uncompressed_records_sha256":raw_hash.hexdigest(),"names":sorted(names),"path":str(path),"source_shard":f"tf-{split}-00000-of-00500.tfrecord.gz"}
    print(json.dumps({k:v for k,v in manifest["splits"][split].items() if k!="names"}),flush=True)
for left,right in [("train","eval"),("train","test"),("eval","test")]:
    assert not(sets[left]&sets[right]),f"molecule leakage {left}/{right}"
manifest["split_molecules_disjoint"]=True
manifest["train_window_size_factor_vs_1593_fixture"]=6400/1593
manifest["train_window_size_factor_vs_1239_labeled_fixture"]=6400/1239
manifest["all_window_size_factor_vs_1593_fixture"]=(6400+1600+800)/1593
(OUT/"dataset-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
