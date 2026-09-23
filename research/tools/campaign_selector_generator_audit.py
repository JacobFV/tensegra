"""CPU-only public-generator identity/coalescence audit; no learned model."""
import argparse,json
from pathlib import Path
import torch
from topoformer.campaign_attention_selector import generate,oracle_successors,swap_instruction
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);rows=[]
for seed in [938001,948001,958001]:
 for balanced in [False,True]:
  unique={d:[] for d in [1,2,4,8,16,32]};terminal_changed=answer_changed=within_changed=0
  for offset in range(0,512,64):
   b=generate(64,128,32,groups=8,seed=seed+offset,balanced=balanced)
   s=oracle_successors(b);sp=oracle_successors(swap_instruction(b));bi=torch.arange(64)
   pointer=torch.arange(128)[None].expand(64,-1).clone();x=b.starts.clone();y=x.clone()
   # Two distinct starts sharing the first node's public attribute.
   same=(b.attributes==b.attributes[:,0,None]).all(-1);same[:,0]=False
   first=torch.zeros(64,dtype=torch.long);second=same.int().argmax(-1)
   for t in range(32):
    pointer=s[:,t].gather(1,pointer);x=s[bi,t,x];y=sp[bi,t,y]
    first=s[bi,t,first];second=s[bi,t,second]
    if t+1 in unique:unique[t+1].extend(len(torch.unique(z)) for z in pointer)
   terminal_changed+=int((x!=y).sum());answer_changed+=int((b.values[bi,x]!=b.values[bi,y]).sum());within_changed+=int((first!=second).sum())
  rows.append(dict(seed=seed,generator='block_permutation_v2' if balanced else 'independent_v1',graphs=512,nodes=128,groups=8,depth=32,
       distinct_terminals={str(k):dict(mean=sum(v)/len(v),min=min(v),max=max(v)) for k,v in unique.items()},
       first_instruction_changed_terminal=terminal_changed,first_instruction_changed_answer=answer_changed,
       two_same_attribute_starts_still_distinct=within_changed))
a.output.write_text(json.dumps(dict(kind='generator-only implementation audit; no neural acquisition claim',rows=rows),indent=2)+'\n')
