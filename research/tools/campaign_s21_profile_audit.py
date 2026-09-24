"""CPU-only S21 mechanical profile audit; no efficacy-based selection."""
import argparse,collections,gzip,hashlib,json,math,time
from pathlib import Path
import torch
from campaign_s19_raw_audit import score
from topoformer.campaign_semantics_s21 import prepare,next_indices,tokens
from topoformer.campaign_semantics_s19 import learning_rate
from topoformer.campaign_semantics_s19_codec import encode_row,NODE,KINDS
from topoformer.thinking_language import state_hash
p=argparse.ArgumentParser()
for name in ('run','receipt','source','data_root','output'):p.add_argument(name,type=Path)
a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
load=lambda p:json.load(gzip.open(p,'rt'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
m=load(a.run/'manifest.json.gz');c=m['config'];receipt=json.loads(a.receipt.read_text())
assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['gpu_processes']=='' and receipt['cap_seconds']==180
assert receipt['config_sha256']=='05b2dcee8c79dc06465bf57169128b6a2776294e390821b6394a47019d72ecea'
assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256']
for n,h in c['source_sha256'].items():assert sha(a.source/n)==h
for x in c['inputs'].values():assert sha(a.data_root/x['path'])==x['sha256']
# prepare uses TRAIN and DEV only; it never opens the sealed confirmation cache.
import os
os.chdir(a.data_root);vocab,arms,dev=prepare(c);assert len(dev)==112
assert [r['arm'] for r in m['results']]==['original','broad'];states=[];graphs=0;stress_reports=[]
for result in m['results']:
 arm=result['arm'];folder=a.run/arm;assert load(folder/'manifest.json.gz')==result
 assert result['seed']==2101 and result['parameters']==62677315 and result['inherited_presentations']==0 and result['added_presentations']==160
 train,panel=arms[arm];g=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=g).tolist();position=0;visits=[0]*4096;stream=hashlib.sha256();exposure=collections.Counter()
 assert [x['update'] for x in result['curves']]==[0,20]
 for curve in result['curves']:
  step=curve['update']
  if step:
   for _ in range(20):
    ids,order,position=next_indices(order,position,g,8);stream.update(json.dumps(ids).encode())
    for i in ids:
     visits[i]+=1;x=train[i];exposure.update(tokens=len(tokens(x['public'])),nodes=len(x['row']['nodes']),edges=len(x['row']['edges']),records=len(x['records']))
  path=folder/curve['checkpoint'];assert sha(path)==curve['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
  assert ck['seed']==2101 and ck['arm']==arm and ck['update']==step and ck['visits']==visits and ck['order']==order and ck['position']==position and torch.equal(ck['schedule'],g.get_state())
  class Proxy:
   def state_dict(self):return ck['model']
  assert state_hash(Proxy())==curve['model_state_sha256'];assert sum(t.numel() for t in ck['model'].values())==62677315
  assert all(torch.isfinite(t).all() for t in ck['model'].values());opt=ck['optimizer'];assert len(opt['state'])==(116 if step else 0)
  assert {float(x['step']) for x in opt['state'].values()}==({20.} if step else set())
  assert all(torch.isfinite(x[k]).all() for x in opt['state'].values() for k in ('exp_avg','exp_avg_sq'))
  group=opt['param_groups'][0];assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01 and math.isclose(group['lr'],learning_rate(20) if step else 3e-4)
  assert curve['presentations']==sum(visits) and curve['exposure']=={k:exposure[k] for k in ('tokens','nodes','edges','records')}
  assert curve['model_state_sha256']==result['initial_state_sha256' if not step else 'final_state_sha256'];states.append(dict(arm=arm,update=step,checkpoint_sha256=sha(path),model_state_sha256=curve['model_state_sha256']));del ck,opt
  for label,pop in [('train',panel),('development',dev)]:
   entry=curve[label];path=folder/entry['artifact'];assert sha(path)==entry['sha256'];d=load(path);assert d['update']==step and d['population']==label and len(d['rows'])==len(pop)==entry['examples'];supports=collections.Counter();invalid=collections.Counter();cells={};valid=exact=0
   for raw,x in zip(d['rows'],pop,strict=True):
    row=x['row'];vi,ex,reason=score(raw,row,vocab);graphs+=1;valid+=vi;exact+=ex;cell=f"{row['arity']}x{row['facts']}";assert raw['cell']==cell;cc=cells.setdefault(cell,dict(examples=0,complete=0));cc['examples']+=1;cc['complete']+=int(ex)
    if not vi:invalid[reason]+=1
    rec=x['records'];n=len(row['nodes']);lex=sum(v[0]==NODE and v[1] in (KINDS.index('ident'),KINDS.index('entity')) for v in rec);ed=len(row['edges']);supports.update(type=len(rec),kind=n,value=n-lex,copy=lex,source=ed,target=ed,role=ed,slot=ed)
   assert entry['valid']==valid and entry['complete']==exact and entry['cells']==cells and entry['invalid_reasons']==dict(invalid) and entry['teacher_forced']==d['teacher_forced']
   for k,x in d['teacher_forced'].items():assert x['count']==supports[k] and 0<=x['correct']<=x['count'] and math.isfinite(x['loss_sum']) and math.isclose(x['mean_loss'],x['loss_sum']/x['count']) and math.isclose(x['accuracy'],x['correct']/x['count'])
   assert math.isclose(d['teacher_forced_loss'],sum(x['mean_loss'] for x in d['teacher_forced'].values())/8) and d['teacher_forced_loss']==entry['teacher_forced_loss']
 assert result['visits']==visits and result['construction_sequence_sha256']==stream.hexdigest() and result['exposure']==dict(exposure)
 assert [x['update'] for x in result['losses']]==[20] and math.isfinite(result['losses'][0]['loss'])
 stress=result['worst_case_timing'];path=folder/stress['artifact'];assert sha(path)==stress['sha256'];d=load(path);public=[x['public'] for x in train+dev];indices=sorted(range(len(public)),key=lambda i:(-len(tokens(public[i])),i))[:32]
 assert stress['public_pool_indices']==d['public_pool_indices']==indices and d['steps']==160 and d['batch_size']==32 and len(d['records'])==32 and all(len(x)==160 for x in d['records'])
 assert d['seconds']==stress['seconds'] and all(len(v)==5 and v[0] in (1,2,3) for rr in d['records'] for v in rr)
 lengths=[len(tokens(public[i])) for i in indices];assert max(lengths)==74;stress_reports.append(dict(arm=arm,lengths=lengths,seconds=d['seconds'],steps=160,batch=32))
assert m['results'][0]['initial_state_sha256']==m['results'][1]['initial_state_sha256'];assert graphs==960
assert not any(q.stat().st_mode&0o222 for q in [a.run,*a.run.rglob('*')])
out=dict(status='pass',seconds=time.monotonic()-start,graphs=graphs,states=states,stress=stress_reports,manifest_sha256=sha(a.run/'manifest.json.gz'),scope='All profile targets/raw codec/metrics/TFsupport, four model/AdamW/sampler states, paired initialization, public-only longest-input160step stress verified. No actor calls or sealed confirmation. No efficacy-based recipe selection.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('states','stress')})
