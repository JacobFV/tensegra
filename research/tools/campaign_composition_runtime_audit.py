"""C01 actual-operand arithmetic and scalar-consumer replay on CPU."""
import argparse,hashlib,json,time,torch
from pathlib import Path
from topoformer.campaign_composition import make_lowering_batch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);m=json.loads((a.root/'summary.json').read_text());cfg=m['config'];spec=cfg['data']['validation'];checks=[]
for name in ('lowerer','backbone','accessor','consumer','oracle_consumer','query_consumer'):
 path=Path(cfg['interfaces'][name]);assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['interfaces'][name+'_sha256']
consumer=torch.nn.Sequential(torch.nn.Linear(35,1024),torch.nn.GELU(),torch.nn.Linear(1024,2));consumer.load_state_dict(torch.load(cfg['interfaces']['consumer'],weights_only=True,map_location='cpu'));consumer.eval()
for dist in cfg['eval_distractors']:
 raw=torch.load(a.root/f'hybrid-{dist}.pt',weights_only=False,map_location='cpu');data=make_lowering_batch(spec['seed'],spec['count'],dist);pub=raw['supplied_public'];proposal=raw['proposal'];n=len(data['public']);assert len(pub['query'])==n
 for i,row in enumerate(data['public']):
  op=int(proposal['primitive'][i]);dest,left,right=proposal['canonical_pointers'][i].tolist();x=row['values'][left];y=row['values'][right]if op!=3 else 0;v=(x+y if op==0 else x-y if op==1 else x*y if op==2 else -x if op==3 else x<y);typ=2 if isinstance(v,bool)else 1 if isinstance(v,float)else 0
  assert float(pub['event']['values'][i,0])==float(v);assert int(pub['event']['types'][i,0])==typ;assert int(pub['event']['operations'][i,0])==op;assert torch.equal(pub['event']['provenance'][i,0],row['keys'][dest]);assert torch.equal(pub['event']['arguments'][i,0,0],row['keys'][left]);assert torch.equal(pub['event']['arguments'][i,0,1],torch.zeros(32)if op==3 else row['keys'][right]);assert raw['reasons'][i]==''
  for field,want in data['reference']['public']['event'].items():assert torch.equal(pub['event'][field][i],want[i])
 for d,cell in raw['result']['cells'].items():
  features=torch.cat((cell['scores'].softmax(-1),pub['query']/torch.tensor([8.,1.])),dim=-1)
  with torch.no_grad():logits=consumer(features)
  assert torch.equal(logits.argmax(-1),cell['predictions']);checks.append(dict(distractors=dist,delay=d,events=n,max_logit_error=float((logits-cell['logits']).abs().max())))
 # Oracle arms must match only because all actual proposals were independently correct.
 for arm in ('oracle_lowering','oracle_return'):
  oracle=torch.load(a.root/f'{arm}-{dist}.pt',weights_only=False,map_location='cpu')
  for d,cell in raw['result']['cells'].items():assert torch.equal(cell['logits'],oracle['result']['cells'][d]['logits'])
out=dict(actual_runtime_events_verified=2*spec['count'],consumer_cells=checks,checkpoint_hashes_verified=6,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU actual-operand arithmetic/type/key provenance reconstructed independently; saved learned scalar scores fed through frozen consumer. This is not a new workspace forward pass. Correct-event comparison is diagnostic only, never repair.');Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out)
