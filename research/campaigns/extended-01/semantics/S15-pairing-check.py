"""CPU-only sampler/control test. No model construction or checkpoint load."""
import hashlib,json,sys
from pathlib import Path
import torch
from topoformer.campaign_semantics_shape_train import presentation_seed,validate_config
from topoformer.campaign_semantics import Dataset
from topoformer.campaign_semantics_data import load_cache
from topoformer.campaign_semantics_continue import next_indices
from topoformer.semantic_curriculum import sampled_pairs
root=Path(sys.argv[1]);vocab=['<unknown>','"parent"','"unify"','null'];rows={a:load_cache(root/f'train_{a}.jsonl.gz') for a in ('control','mixed')};data={a:Dataset(r,vocab) for a,r in rows.items()}
for a in rows:
 c=json.loads(Path(f'configs/campaign-s15-{a}-profile-prepared-v2.json').read_text())
 try:validate_config(c)
 except ValueError:pass
 else:raise AssertionError('prepared config allowed')
 validate_config({**c,'budget_status':'frozen'})
schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();position=0;visits=[0]*4096;common_checked=0;different_checked=0;hashes={a:hashlib.sha256() for a in rows}
for step in range(4096):
 indices,order,position=next_indices(order,position,schedule,8)
 for offset,index in enumerate(indices):
  visits[index]+=1
  if step>=32:continue
  number=step*8+offset;pairs={a:sampled_pairs(data[a][index][1],torch.Generator().manual_seed(presentation_seed(number)),128) for a in rows}
  if rows['control'][index]['alpha_sha256']==rows['mixed'][index]['alpha_sha256']:
   assert torch.equal(pairs['control'],pairs['mixed']);common_checked+=1
  else:different_checked+=1
  for a,p in pairs.items():hashes[a].update(p.numpy().tobytes())
assert set(visits)=={8} and common_checked>0 and different_checked>0
print(json.dumps(dict(no_model_construction=True,cuda_visible=torch.cuda.is_available(),common_presentations_checked=common_checked,different_presentations_checked=different_checked,each_graph_visits=8,total_presentations=sum(visits),pair_digests={a:h.hexdigest() for a,h in hashes.items()})))
