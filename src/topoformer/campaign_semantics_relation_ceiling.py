"""Privileged relation replacement counts from archived labels, no actor inference."""
import argparse,gzip,json,time
from pathlib import Path
import torch
from .semantic_curriculum import unpack_graph
from .semantic_scaling import metrics
from .thinking_language import ROLES

p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();torch.set_num_threads(2);start=time.monotonic();x=json.load(gzip.open(a.archive,'rt'));results=[]
for split,rows in [('train',x['train_rows']),('development',x['rows'])]:
    for mode in ['raw','calibrated']:
        counts={r:0 for r in ROLES};counts['refers_to+argument']=0
        for row in rows:
            pred=unpack_graph(row['raw']);gold=unpack_graph(row['target'])
            if mode=='calibrated':pred['edges']=unpack_graph({**row['raw'],'edges':row['calibrated_edges']})['edges']
            for r in counts:
                edges=pred['edges'].clone()
                for role in r.split('+'):i=ROLES.index(role);edges[:,:,i]=gold['edges'][:,:,i]
                counts[r]+=int(metrics({**pred,'edges':edges},gold)['semantic_equivalence'])
        results.append(dict(split=split,mode=mode,examples=len(rows),privileged_replacement_exact=counts))
Path(a.output).write_text(json.dumps(dict(cpu_seconds=time.monotonic()-start,results=results),indent=2)+'\n')
