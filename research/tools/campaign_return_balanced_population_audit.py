"""CPU-only independent balanced-event selection/witness and population audit."""
import argparse,gzip,hashlib,io,json,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source/'src'));from topoformer.retention_data import make_batch
start=time.monotonic();torch.set_num_threads(2);populations={};records=[];witness_rows=0

def digest(tree):
 h=hashlib.sha256()
 def visit(v,name=''):
  if isinstance(v,dict):
   for k in sorted(v):visit(v[k],name+'/'+k)
  else:
   v=v.detach().contiguous().cpu();h.update(name.encode());h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
 visit(tree);return h.hexdigest()

for name in ['r01-profile','r01-development',*[f'r01-confirmation/{s}'for s in (10,11,12)],'r02-development',*[f'r04-confirmation/{s}'for s in (10,11,12)]]:
 root=a.root/name;path=root/'manifest.json.gz';m=json.load(gzip.open(path,'rt'))if path.exists()else json.loads((root/'manifest.json').read_text());record=m.get('features',m.get('feature_cache'));packed=(root/record['path']).read_bytes();assert hashlib.sha256(packed).hexdigest()==record['sha256'];cache=torch.load(io.BytesIO(gzip.decompress(packed)),map_location='cpu',weights_only=True)
 for split in m['config']['data']:
  ids=cache[split+'/2']['event_row_hashes'];assert len(ids)==len(set(ids));populations[name+'/'+split]=set(ids)
 if name.startswith('r04'):
  assert hashlib.sha256((a.source/'src/topoformer/retention_data.py').read_bytes()).hexdigest()==m['source']['retention_data.py'];cfg=m['config']['balanced_grid'];pool=make_batch(cfg['seed'],cfg['pool_size'],distractors=8);ix=[]
  for typ,labels in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))):
   for label in labels:
    ids=((pool['targets']['type']==typ)&(pool['targets']['value']==label)).nonzero().flatten();assert len(ids)>=cfg['per_cell'];ix.extend(ids[:cfg['per_cell']].tolist())
  saved=cache['balanced/8'];assert ix==saved['event_indices'];ix=torch.tensor(ix);event={k:v[ix]for k,v in pool['public']['event'].items()};assert digest(event)==saved['event_sha256']
  for k,v in saved['targets'].items():assert torch.equal(v,pool['targets'][k][ix])
  op=event['operations'][:,0];value=event['values'][:,0];x,y=event['operand_values'][:,0].unbind(-1);executed=torch.stack((x+y,x-y,x*y,-x,(x<y).float()),1).gather(1,op[:,None]).squeeze(1);assert torch.equal(executed,value);assert torch.equal(event['argument_mask'][:,0,1],op!=3)
  hashes=[digest({k:v[j]for k,v in event.items()})for j in range(len(ix))];assert hashes==saved['event_row_hashes'];assert len(set(hashes))==len(hashes);populations[name+'/balanced']=set(hashes);witness_rows+=len(hashes);records.append(dict(run=name,events=len(hashes),strata=52,per_stratum=cfg['per_cell'],event_sha256=digest(event)));del pool,event
 del cache,packed
for i,(name,ids)in enumerate(populations.items()):
 for other,them in list(populations.items())[i+1:]:assert not(ids&them),(name,other)
out=dict(cpu_audit_wall_seconds=time.monotonic()-start,balanced_witness_events_verified=witness_rows,balanced_records=records,distinct_populations=len(populations),distinct_events=sum(map(len,populations.values())),all_populations_pairwise_disjoint=True,scope='Unchanged historical generator used only to regenerate public events; independent stratification, target comparison, exact operand execution and byte hashing. No neural inference or GPU work. R03 cache reuse deliberately excluded from disjointness assertion.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
