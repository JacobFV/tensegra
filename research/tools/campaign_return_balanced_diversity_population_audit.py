import json,gzip,io,hashlib,time,argparse
from pathlib import Path
import torch
from topoformer.retention_data import make_batch
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2);t=time.monotonic();root=args.root;m=json.load(gzip.open(root/'manifest.json.gz','rt'));cfg=m['config'];cache=torch.load(io.BytesIO(gzip.decompress((root/'cache.pt.gz').read_bytes())),map_location='cpu',weights_only=True)
def digest(tree):
 h=hashlib.sha256()
 def visit(v,name=''):
  if isinstance(v,dict):
   for k in sorted(v):visit(v[k],name+'/'+k)
  else:
   v=v.detach().contiguous().cpu();h.update(name.encode());h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
 visit(tree);return h.hexdigest()
strata=[(typ,label) for typ,labels in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))) for label in labels];checked=0
for split in ['train']+list(cfg['grids']):
 istrain=split=='train';sp=dict(seed=cfg['pool_seed'],pool_size=cfg['pool_size']) if istrain else cfg['grids'][split];pool=make_batch(sp['seed'],sp['pool_size'],distractors=2 if istrain else 8);strat=strata
 if istrain:
  order=torch.randperm(52,generator=torch.Generator().manual_seed(cfg['strata_seed'])).tolist();strat=[strata[i] for i in order]
 arrays=[((pool['targets']['type']==typ)&(pool['targets']['value']==label)).nonzero().flatten().tolist() for typ,label in strat]
 if istrain:ids=[arrays[i%52][i//52] for i in range(max(cfg['fit_sizes']))]
 else:ids=[i for group in arrays for i in group[:sp['per_cell']]]
 saved=cache[split+('/2' if istrain else '/8')];assert ids==saved['event_indices'];ix=torch.tensor(ids);event={k:v[ix] for k,v in pool['public']['event'].items()};assert digest(event)==saved['event_sha256'];assert [digest({k:v[j] for k,v in event.items()}) for j in range(len(ids))]==saved['event_row_hashes']
 for k,v in saved['targets'].items():assert torch.equal(v,pool['targets'][k][ix])
 op=event['operations'][:,0];x,y=event['operand_values'][:,0].unbind(-1);executed=torch.stack((x+y,x-y,x*y,-x,(x<y).float()),1).gather(1,op[:,None]).squeeze(1);assert torch.equal(executed,event['values'][:,0]);checked+=len(ids)
out=dict(public_events_regenerated=checked,independent_nested_and_grid_indices=True,targets_and_exact_arithmetic_witnesses_verified=True,event_and_row_hashes_verified=True,cpu_audit_wall_seconds=time.monotonic()-t);args.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
