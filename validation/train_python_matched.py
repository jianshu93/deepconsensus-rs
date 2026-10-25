"""Matched warm-start training using the upstream DeepConsensus model and loss."""
import argparse
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import struct
import time
import benchmark_upstream as compat
import numpy as np
tf=compat.tf
model_utils=compat.model_utils
from deepconsensus.models import losses_and_metrics

ROOT=compat.ROOT
DATA=Path(os.environ.get("DC_DATA_ROOT",str(ROOT/"larger-benchmark")))

def records(path):
    with gzip.open(path,"rb") as f:
        while True:
            header=f.read(12)
            if not header:
                return
            if len(header)!=12:
                raise ValueError("truncated TFRecord")
            length=struct.unpack("<Q",header[:8])[0]
            raw=f.read(length)
            crc=f.read(4)
            if len(raw)!=length or len(crc)!=4:
                raise ValueError("truncated TFRecord")
            fields=tf.train.Example.FromString(raw).features.feature
            rows=np.frombuffer(fields["subreads/encoded"].bytes_list.value[0],dtype="<f4").reshape(86,100,1).copy()
            rows[20:40]=np.clip(rows[20:40],0,255)
            rows[40:60]=np.clip(rows[40:60],0,255)
            rows[82:86]=np.clip(rows[82:86],0,500)
            label=np.frombuffer(fields["label/encoded"].bytes_list.value[0],dtype="<f4").astype(np.int32)
            yield rows,label

def batches(path,size):
    rows=[]
    labels=[]
    for x,y in records(path):
        rows.append(x)
        labels.append(y)
        if len(rows)==size:
            yield np.stack(rows),np.stack(labels)
            rows=[]
            labels=[]
    if rows:
        yield np.stack(rows),np.stack(labels)

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        diagonal=row[0]
        row[0]=i+1
        for j,y in enumerate(b):
            old=row[j+1]
            row[j+1]=min(diagonal+(x!=y),row[j]+1,old+1)
            diagonal=old
    return row[-1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-steps",type=int,default=200)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--learning-rate",type=float,default=3e-6)
    parser.add_argument("--report",type=Path,default=DATA/"python-training.json")
    parser.add_argument("--output",type=Path,default=DATA/"python-trained")
    parser.add_argument("--evaluate-only",action="store_true")
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([],"GPU")
    tf.random.set_seed(1)
    p=model_utils.read_params_from_json(str(ROOT/"checkpoint"))
    for key in ["attention_dropout","relu_dropout","layer_postprocess_dropout"]:
        p[key]=type(p[key])(0)
    model_utils.modify_params(p,speedy=True,max_length=100,is_training=False)
    model=model_utils.get_model(p)
    with contextlib.redirect_stdout(io.StringIO()):
        model_utils.print_model_summary(model,(1,86,100,1))
    tf.train.Checkpoint(model=model).restore(str(ROOT/"checkpoint")).expect_partial().assert_existing_objects_matched()
    objective=losses_and_metrics.AlignmentLoss(del_cost=10.0,loss_reg=0.1,width=None,reduction=tf.keras.losses.Reduction.NONE)
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate,beta_1=0.9,beta_2=0.997,epsilon=1e-9)
    @tf.function(input_signature=[tf.TensorSpec([None,86,100,1],tf.float32),tf.TensorSpec([None,100],tf.int32)])
    def forward(rows,labels):
        predictions=model(rows,training=False)
        loss=objective(labels,predictions)
        return predictions,loss
    @tf.function(input_signature=[tf.TensorSpec([args.batch_size,86,100,1],tf.float32),tf.TensorSpec([args.batch_size,100],tf.int32)])
    def step(rows,labels):
        with tf.GradientTape() as tape:
            predictions=model(rows,training=True)
            loss=tf.reduce_mean(objective(labels,predictions))
        gradient=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        return loss
    first=next(batches(DATA/"train.tfrecord.gz",args.batch_size))
    compile_timer=time.perf_counter()
    if not args.evaluate_only:
        step.get_concrete_function()
    forward.get_concrete_function()
    compile_seconds=time.perf_counter()-compile_timer
    # Warm the forward/loss graph without an optimizer update.
    first_output,first_losses=forward(*first)
    initial_first_loss=float(tf.reduce_mean(first_losses).numpy())
    steps=[]
    epoch_start=time.perf_counter()
    if not args.evaluate_only:
        for index,(rows,labels) in enumerate(batches(DATA/"train.tfrecord.gz",args.batch_size)):
            if index>=args.max_steps:
                break
            before=time.perf_counter()
            loss=float(step(rows,labels).numpy())
            elapsed=time.perf_counter()-before
            steps.append({"step":index+1,"loss":loss,"seconds":elapsed})
            if index==0 or (index+1)%10==0:
                print(json.dumps(steps[-1]),flush=True)
        args.output.mkdir(exist_ok=True)
        tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(args.output/"checkpoint"))
    epoch_seconds=time.perf_counter()-epoch_start
    evaluation={}
    for split in ["eval","test"]:
        timer=time.perf_counter()
        count=0
        loss_sum=0.0
        errors=0
        truth_bases=0
        examples=[]
        for rows,labels in batches(DATA/f"{split}.tfrecord.gz",64):
            probabilities,losses=forward(rows,labels)
            probabilities=probabilities.numpy()
            loss_sum+=float(tf.reduce_sum(losses).numpy())
            prediction=np.argmax(probabilities,axis=-1)
            for true,pred in zip(labels,prediction):
                true=true[true!=0].tolist()
                pred=pred[pred!=0].tolist()
                error=edit_distance(true,pred)
                errors+=error
                truth_bases+=len(true)
                count+=1
                examples.append({"truth":true,"prediction":pred,"edits":error})
        evaluation[split]={"examples":count,"mean_alignment_loss":loss_sum/count,"edit_errors":errors,"truth_bases":truth_bases,"error_rate":errors/truth_bases,"seconds":time.perf_counter()-timer}
        (DATA/f"python-{'baseline' if args.evaluate_only else 'trained'}-{split}-predictions.json").write_text(json.dumps(examples))
        print(json.dumps({"split":split,**evaluation[split]}),flush=True)
    report={"framework":"upstream DeepConsensus model and dense differentiable alignment loss on TensorFlow 2.20.0","training_recipe":{"warm_start":"published v1.2","model_layers":5,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"beta1":0.9,"beta2":0.997,"epsilon":1e-9,"weight_decay":0.0,"dropout":0.0,"shuffle":"pre-shuffled identical file; no in-loop reshuffle","loss_reg":0.1,"del_cost":10.0,"band_width":None},"compile_seconds":compile_seconds,"initial_first_batch_loss":initial_first_loss,"epoch_seconds_including_input_and_checkpoint":epoch_seconds,"steps":steps,"evaluation":evaluation,"limitations":["Common controlled optimizer recipe; not upstream default LAMB trainer or its per-step identity metrics","Warm-start short CPU experiment, not converged de novo training","Evaluation/test are separate fine-tuning splits; disjointness from published checkpoint pretraining is not established"]}
    args.report.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
