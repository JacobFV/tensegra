"""CPU immutable optimizer LR-only state and checkpoint-byte verification."""
import argparse,json,gzip,hashlib,time,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);m=json.load(gzip.open(a.root/'manifest.json.gz'));base=a.root.parents[1];cfg=m['config'];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();parent=base/cfg['parent_checkpoint'];assert sha(parent)==cfg['parent_checkpoint_sha256'];receipts=[]
for c in m['curves']:
 path=a.root/f"model-u{c['update']}.pt";got=sha(path);assert got==c['checkpoint_sha256'];receipts.append(dict(update=c['update'],sha256=got))
x=torch.load(parent,map_location='cpu',weights_only=True);y=torch.load(a.root/f"model-u{m['start_update']}.pt",map_location='cpu',weights_only=True)
def eq(x,y):
 if isinstance(x,torch.Tensor):return torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys()and all(eq(v,y[k])for k,v in x.items())
 if isinstance(x,(list,tuple)):return len(x)==len(y)and all(eq(v,w)for v,w in zip(x,y))
 return x==y
for k in x:
 if k!='optimizer':assert eq(x[k],y[k]),k
assert eq(x['optimizer']['state'],y['optimizer']['state'])
for pg,qg in zip(x['optimizer']['param_groups'],y['optimizer']['param_groups']):
 assert pg['lr']==.0001 and qg['lr']==.00001
 assert eq({k:v for k,v in pg.items()if k!='lr'},{k:v for k,v in qg.items()if k!='lr'})
out=dict(checkpoints=receipts,parent_sha256=cfg['parent_checkpoint_sha256'],model_optimizer_moments_sampling_state_exact=True,only_optimizer_group_lr_changed=True,cpu_audit_wall_seconds=time.monotonic()-t);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
