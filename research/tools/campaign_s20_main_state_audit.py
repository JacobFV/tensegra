"""CPU-only all-nine S20 checkpoint/state audit; does not open confirmation records."""
import argparse,collections,gzip,hashlib,json,math,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_continue import next_indices
from topoformer.campaign_semantics_data import load_cache
from topoformer.thinking_language import state_hash
from topoformer.campaign_semantics_s19 import learning_rate
from topoformer.campaign_semantics_s20_freeze import PAIRS,STREAM

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('receipt',type=Path);p.add_argument('data_root',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2)
load=lambda p:json.load(gzip.open(p,'rt'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
m=load(a.run/'manifest.json.gz');c=m['config'];receipt=json.loads(a.receipt.read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['gpu_processes']=='' and receipt['cap_seconds']==4300;assert receipt['config_sha256']=='4e648a9a274a7cabcbb84c0b2e6e6a2dcbf37925e2146b0758702086245f6c8b';assert hashlib.sha256((json.dumps(c,indent=2)+'\n').encode()).hexdigest()==receipt['config_sha256'];assert [(r['seed'],r['arm']) for r in m['results']]==[(s,arm) for s in (701,702,703) for arm in ('original','context','record')]
for n,h in c['source_sha256'].items():assert sha(a.source/n)==h
for k,v in c['inputs'].items():assert sha(a.data_root/v['path'])==v['sha256']
# Hashing confirmation bytes is allowed; no confirmation content is decoded.
train=load_cache(a.data_root/c['inputs']['train']['path']);g=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=g).tolist();position=0;visits=[0]*4096;stream=hashlib.sha256();exposure=collections.Counter()
for step in range(4096):
 ids,order,position=next_indices(order,position,g,8);stream.update(json.dumps(ids).encode())
 for i in ids:
  row=train[i];visits[i]+=1;exposure.update(tokens=row['tokens'],nodes=len(row['nodes']),edges=len(row['edges']),records=len(row['nodes'])+len(row['edges'])+1)
assert stream.hexdigest()==STREAM and set(visits)=={8};states=[]
for r in m['results']:
 seed,arm=r['seed'],r['arm'];folder=a.run/f'seed-{seed}'/arm;assert r==load(folder/'manifest.json.gz');path=folder/r['checkpoint'];assert sha(path)==r['checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True);record=arm=='record';step=4096 if record else 28672
 assert ck['seed']==seed and ck['arm']==arm and ck['update']==r['update']==step and ck['added_update']==r['added_update']==4096 and ck['presentation_number']==32768
 assert ck['visits']==r['visits']==visits and ck['order']==order and ck['position']==position and torch.equal(ck['schedule'],g.get_state());assert r['added_presentations']==32768 and r['construction_sequence_sha256']==STREAM and r['pair_sequence_sha256']==(None if record else PAIRS)
 for k in ('tokens','nodes','edges'):assert r['optimizer_'+k]==exposure[k]
 assert r['optimizer_records']==(exposure['records'] if record else None)
 class Proxy:
  def state_dict(self):return ck['model']
 assert state_hash(Proxy())==r['final_state_sha256'];assert sum(x.numel() for x in ck['model'].values())==r['parameters']==(62677315 if record else 57853781);assert all(torch.isfinite(x).all() for x in ck['model'].values());opt=ck['optimizer'];assert len(opt['state'])==(116 if record else 98) and {float(x['step']) for x in opt['state'].values()}=={float(step)};assert all(torch.isfinite(x[k]).all() for x in opt['state'].values() for k in ('exp_avg','exp_avg_sq'));group=opt['param_groups'][0];assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==.01 and math.isclose(group['lr'],learning_rate(4096) if record else 1e-5)
 if record:assert r['parent_checkpoint_sha256'] is None and r['inherited_presentations']==0 and r['initial_optimizer_states']==0 and r['inherited_optimizer_steps']==[]
 else:
  parent=c['parents'][str(seed)];assert r['initial_state_sha256']==parent['model_state_sha256'] and r['parent_checkpoint_sha256']==parent['checkpoint']['sha256'] and r['inherited_presentations']==196608 and r['initial_optimizer_states']==98 and r['inherited_optimizer_steps']==[24576.]
 assert [x['added_update'] for x in r['losses']]==list(range(128,4097,128));assert all(math.isfinite(x['loss']) for x in r['losses']);assert r['evaluation_population']=='confirmation' and r['worst_case_timing'] is None
 states.append(dict(seed=seed,arm=arm,checkpoint_sha256=sha(path),final_state_sha256=r['final_state_sha256'],initial_state_sha256=r['initial_state_sha256'],optimizer_step=step,optimizer_slots=len(opt['state'])));del ck,opt
assert not any(q.stat().st_mode&0o222 for q in [a.run,*a.run.rglob('*')]);out=dict(status='pass',seconds=time.monotonic()-tick,states=states,exposure=dict(exposure),construction_sha256=stream.hexdigest(),pair_sha256=PAIRS,readonly=True,manifest_sha256=sha(a.run/'manifest.json.gz'),scope='Allnineclosed checkpoints/optimizer/sampler/exposure/source verified; initialstate restoration reuses audited exactparent/source contract. Fullnegativequery hash reused from independently recomputed S18/S15 sameTRAIN/source/schedule, S20profile query replay confirmed. No model construction/forward or confirmation parsing.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='states'})
