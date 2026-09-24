"""CPU final paired semantic checkpoint bytes, exposure and sampling state."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);roots=[a.root/f's12-{arm}-{a.seed}' for arm in ('constant','decay')];out=[];states=[]
for root in roots:
 m=json.load(gzip.open(root/'manifest.json.gz'));p=root/'model-u24576.pt';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==m['curves'][-1]['checkpoint_sha256'];x=torch.load(p,map_location='cpu',weights_only=False);assert x['update']==24576;assert {int(v['step']) for v in x['optimizer']['state'].values()}=={24576};assert x['visits']==[24]*8192;assert all(v['lr']==m['config']['learning_rate'] for v in x['optimizer']['param_groups']);states.append({k:x[k] for k in ('order','position','generator','schedule','visits')});out.append(dict(arm=root.name,checkpoint_sha256=h,optimizer_steps=24576,presentations=sum(x['visits'])));del x
for k in states[0]:
 x,y=states[0][k],states[1][k];assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y,k
result=dict(seed=a.seed,checkpoints=out,final_sampling_states_exact=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='Independent final checkpoint byte/state/exposure inspection. No model inference; initial parameter/optimizer inheritance audited separately.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
