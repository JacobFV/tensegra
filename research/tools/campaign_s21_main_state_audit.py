"""Closed S21 paired main checkpoint audit; CPU only, no model or outcome calls."""
import argparse,collections,gzip,hashlib,json,math,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_continue import next_indices
from topoformer.campaign_semantics_data import load_cache
from topoformer.campaign_semantics_s19 import learning_rate
from topoformer.thinking_language import state_hash
p=argparse.ArgumentParser()
for n in ('run','receipt','source','data_root','profile_audit','output'):p.add_argument(n,type=Path)
a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
load=lambda p:json.load(gzip.open(p,'rt'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt=json.loads(a.receipt.read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt.get('error') is None and receipt['gpu_processes']=='' and receipt['cap_seconds']==1770
assert receipt['config_sha256']=='5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3'
m=load(a.run/'manifest.json.gz');c=m['config'];assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256']
assert c['job']=='main' and c['updates']==4096 and c['checkpoints']==[0,1024,2048,4096] and [r['arm'] for r in m['results']]==['original','broad']
for n,h in c['source_sha256'].items():assert sha(a.source/n)==h
for x in c['inputs'].values():assert sha(a.data_root/x['path'])==x['sha256']
profile=json.loads(a.profile_audit.read_text());assert profile['status']=='pass';initials={x['model_state_sha256'] for x in profile['states'] if x['update']==0};assert len(initials)==1
states=[]
for result in m['results']:
 arm=result['arm'];folder=a.run/arm;assert load(folder/'manifest.json.gz')==result
 assert result['seed']==2101 and result['inherited_presentations']==0 and result['added_presentations']==32768 and result['initial_state_sha256'] in initials and result['worst_case_timing'] is None
 rows=load_cache(a.data_root/c['inputs']['train_'+arm]['path']);assert len(rows)==4096
 g=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=g).tolist();position=0;visits=[0]*4096;stream=hashlib.sha256();exposure=collections.Counter();previous=0
 assert [x['update'] for x in result['curves']]==[0,1024,2048,4096]
 for curve in result['curves']:
  step=curve['update']
  for _ in range(previous,step):
   ids,order,position=next_indices(order,position,g,8);stream.update(json.dumps(ids).encode())
   for i in ids:
    x=rows[i];visits[i]+=1;exposure.update(tokens=x['tokens'],nodes=len(x['nodes']),edges=len(x['edges']),records=len(x['nodes'])+len(x['edges'])+1)
  previous=step;path=folder/curve['checkpoint'];assert sha(path)==curve['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
  assert ck['seed']==2101 and ck['arm']==arm and ck['update']==step and ck['visits']==visits and ck['order']==order and ck['position']==position and torch.equal(ck['schedule'],g.get_state())
  class Proxy:
   def state_dict(self):return ck['model']
  assert state_hash(Proxy())==curve['model_state_sha256'];assert sum(t.numel() for t in ck['model'].values())==result['parameters']==62677315
  assert all(torch.isfinite(t).all() for t in ck['model'].values());opt=ck['optimizer'];assert len(opt['state'])==(116 if step else 0)
  assert {float(x['step']) for x in opt['state'].values()}==({float(step)} if step else set());assert all(torch.isfinite(x[k]).all() for x in opt['state'].values() for k in ('exp_avg','exp_avg_sq'))
  group=opt['param_groups'][0];assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01 and math.isclose(group['lr'],learning_rate(step) if step else 3e-4)
  assert curve['presentations']==sum(visits)==8*step and curve['exposure']=={k:exposure[k] for k in ('tokens','nodes','edges','records')}
  if step==0:assert curve['model_state_sha256']==result['initial_state_sha256']
  if step==4096:assert curve['model_state_sha256']==result['final_state_sha256']
  states.append(dict(arm=arm,update=step,checkpoint_sha256=sha(path),model_state_sha256=curve['model_state_sha256'],optimizer_states=len(opt['state']),exposure=dict(curve['exposure'])));del ck,opt
 assert result['visits']==visits and set(visits)=={8} and result['construction_sequence_sha256']==stream.hexdigest()==c['expected_construction_sequence_sha256']
 assert result['exposure']==dict(exposure)=={k:v*8 for k,v in c['expected_epoch_exposures'][arm].items()}
 assert [x['update'] for x in result['losses']]==list(range(128,4097,128)) and all(math.isfinite(x['loss']) and math.isclose(x['learning_rate'],learning_rate(x['update'])) for x in result['losses'])
assert not any(p.stat().st_mode&0o222 for p in [a.run,*a.run.rglob('*')])
out=dict(status='pass',states=states,seconds=time.monotonic()-start,manifest_sha256=sha(a.run/'manifest.json.gz'),scope='All eight closed checkpoint/model/AdamW/sampler/exposure/source states; audited profile initial-state equality. TRAIN only, no evaluation or sealed confirmation parsing; no actor/model calls.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='states'})
