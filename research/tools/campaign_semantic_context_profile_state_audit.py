"""S18 saved profile states and independent query-stream replay; CPU only."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_data import load_cache,target
from topoformer.semantic_curriculum import sampled_pairs
p=argparse.ArgumentParser();p.add_argument('--bindings',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:torch.load(p,map_location='cpu',weights_only=True);b=json.loads(a.bindings.read_text());m=json.load(gzip.open(a.run/'manifest.json.gz','rt'));assert m['config']['job']=='profile' and m['config']['added_updates']==20;parent=load(b['parent']['path']);assert sha(b['parent']['path'])==b['parent']['sha256'];rows=load_cache(b['caches']['train_mixed']['path']);vocab=json.loads(Path(b['data_audit']['path']).read_text())['value_vocabulary'];hashes={}
def eq(x,y):
 if torch.is_tensor(x):assert torch.equal(x,y)
 elif isinstance(x,dict):assert x.keys()==y.keys();[eq(v,y[k]) for k,v in x.items()]
 elif isinstance(x,(list,tuple)):assert len(x)==len(y);[eq(u,v) for u,v in zip(x,y)]
 else:assert x==y
schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();sequence=hashlib.sha256();pairs=hashlib.sha256();visits=[0]*4096;tokens=nodes=edges=0
for step in range(20):
 ids=order[step*8:(step+1)*8];sequence.update(json.dumps(ids).encode())
 for offset,i in enumerate(ids):
  row=rows[i];q=sampled_pairs(target(row,vocab),torch.Generator().manual_seed(150150000+step*8+offset),128);pairs.update(q.numpy().tobytes());visits[i]+=1;tokens+=row['tokens'];nodes+=len(row['nodes']);edges+=len(row['edges'])
for result in m['results']:
 arm=result['arm'];assert [c['added_update'] for c in result['curves']]==[0,20];assert result['visits']==visits;assert result['construction_sequence_sha256']==sequence.hexdigest() and result['pair_sequence_sha256']==pairs.hexdigest();assert (result['optimizer_tokens'],result['optimizer_nodes'],result['optimizer_edges'])==(tokens,nodes,edges)
 for curve in result['curves']:
  path=a.run/arm/f"model-u{curve['update']}.pt";h=sha(path);assert h==curve['checkpoint_sha256'];hashes[str(path)]=h;x=load(path);step=curve['added_update'];assert x['update']==24576+step and x['added_update']==step and x['presentation_number']==step*8;assert x['order']==order and x['position']==step*8;eq(x['schedule'],schedule.get_state());assert x['visits']==([0]*4096 if step==0 else visits);assert {int(s['step']) for s in x['optimizer']['state'].values()}=={24576+step}
  if step==0:eq(x['model'],parent['model']);eq(x['optimizer'],parent['optimizer'])
  assert all(torch.isfinite(v).all() for v in x['model'].values());assert all(torch.isfinite(v).all() for state in x['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v));del x
out=dict(checkpoint_sha256=hashes,both_initial_model_and_AdamW_exact_parent=True,both160_presentations_and_queries_exact=True,final_AdamW_step=24596,final_visits_exact=True,public_exposure=dict(tokens=tokens,nodes=nodes,edges=edges),cpu_audit_wall_seconds=time.monotonic()-tick,scope='CPU saved tensors and target/query sampling only; no model creation/forward, no profile outcome selection.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
