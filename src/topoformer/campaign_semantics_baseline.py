"""CPU-only matched English training-mixture frequency baseline."""
import argparse,gzip,json,hashlib,time
from pathlib import Path
import torch
from .campaign_semantics_data import load_cache,target
from .semantic_curriculum import pack_graph
from . import semantic_scaling as base
from .thinking_language import KINDS,ROLES


def run(data_dir,output,train_limit=None):
    torch.set_num_threads(2);tick=time.monotonic();root=Path(data_dir);audit=json.loads((root/'audit.json').read_text());vocab=audit['value_vocabulary'];train=load_cache(root/'train.jsonl.gz');dev=load_cache(root/'development.jsonl.gz');capacity=128
    if train_limit is not None:train=train[:train_limit]
    max_tokens=max(r['tokens'] for r in train);counts=dict(presence=torch.zeros(capacity,2,dtype=torch.int32),kind=torch.zeros(capacity,len(KINDS),dtype=torch.int32),value=torch.zeros(capacity,len(vocab)+1,dtype=torch.int32),copy=torch.zeros(capacity,max_tokens+1,dtype=torch.int32),slots=torch.zeros(capacity,capacity,base.MAX_SLOT+1,dtype=torch.int32),edges=torch.zeros(capacity,capacity,len(ROLES),dtype=torch.int32))
    for row in train:
        gold=target(row,vocab)
        for key,count in counts.items():
            if key=='edges':count.add_(gold[key].int());continue
            labels=gold[key].long()+int(key in ('value','copy','slots'))
            count.scatter_add_(-1,labels[...,None],torch.ones_like(labels[...,None],dtype=count.dtype))
    pred={key:(value.argmax(-1)-int(key in ('value','copy','slots'))) for key,value in counts.items() if key!='edges'};pred['presence']=pred['presence'].bool();pred['edges']=counts['edges']>len(train)/2
    rows=[dict(seed=r['seed'],metrics=base.metrics(pred,target(r,vocab))) for r in dev]
    result=dict(scope='Coordinatewise mode over declared TRAIN English prefix. No input text or token features. Fixed canonical output-position/copy-index frequencies remain.',train_examples=len(train),development_examples=len(dev),training_cache_sha256=hashlib.sha256((root/'train.jsonl.gz').read_bytes()).hexdigest(),prediction=pack_graph(pred),rows=rows,cpu_seconds=time.monotonic()-tick)
    with gzip.GzipFile(filename=output,mode='wb',mtime=0) as f:f.write(json.dumps(result,separators=(',',':')).encode())
    print(json.dumps(dict(exact=sum(r['metrics']['semantic_equivalence'] for r in rows),examples=len(rows),cpu_seconds=result['cpu_seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data_dir');p.add_argument('output');p.add_argument('--train-limit',type=int);a=p.parse_args();run(a.data_dir,a.output,a.train_limit)
