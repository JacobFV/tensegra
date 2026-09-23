"""Recreate public P01 populations and CPU replay selected lowering."""
import hashlib,json,time,torch,argparse
from pathlib import Path
from topoformer.campaign_composition import make_lowering_batch,model_inputs,make_model

def digest(public):
 h=hashlib.sha256()
 for k,v in sorted(public.items()):
  v=v.contiguous().cpu();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
start=time.monotonic();torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();base=Path.home()/'topoformer-campaign-01/composition';sets={};replays=[]
for seed in [302,303,304]:
 root=base/f'p01-confirmation-{seed}';m=json.loads((root/'summary.json').read_text());model=make_model().eval();model.load_state_dict(torch.load(root/'selected.pt',map_location='cpu',weights_only=True))
 for split,spec in m['config']['data'].items():
  data=make_lowering_batch(spec['seed'],spec['count']);public=model_inputs(data['public']);assert digest(public)==m['data'][split]['public_tensor_sha256']
  ids=set()
  for i in range(spec['count']):ids.add(digest({k:v[i:i+1] for k,v in public.items()}))
  assert len(ids)==spec['count'];sets[f'{seed}/{split}']=ids
  if split=='validation':
   raw=torch.load(root/'validation.pt',map_location='cpu',weights_only=False)
   assert all(torch.equal(v,raw['public'][k]) for k,v in public.items())
   # Independent exact-address targets from public instruction/query, no generator labels needed.
   for i in range(spec['count']):
    q=public['query_destination'][i];j=(public['instruction_destinations'][i]==q).all(-1).nonzero().flatten();assert len(j)==1;j=int(j)
    role=torch.cat((q[None],public['instruction_arguments'][i,j]));ptr=torch.stack([(public['keys'][i]==r).all(-1).nonzero().flatten()[0] for r in role])
    assert torch.equal(ptr,raw['labels']['targets'][i]);assert int(public['instruction_cues'][i,j].argmax())==int(raw['labels']['primitive'][i])
   with torch.no_grad():
    for j in range(0,spec['count'],256):
     out=model({k:v[j:j+256] for k,v in public.items()})
     for k,v in out.items():assert torch.equal(v.argmax(-1),raw['logits'][k][j:j+256].argmax(-1))
   replays.append(dict(seed=seed,validation_rows=spec['count'],argmax_equal=True,public_exact_address_labels=True))
  del data,public
for i,(name,x) in enumerate(sets.items()):
 for b,y in list(sets.items())[i+1:]:assert not(x&y),(name,b)
result=dict(populations={k:len(v) for k,v in sets.items()},pairwise_disjoint=True,regenerated_public_hashes=True,replays=replays,cpu_audit_wall_seconds=time.monotonic()-start)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
