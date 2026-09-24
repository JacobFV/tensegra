"""CPU S13 main checkpoint, inherited state and presentation schedule audit."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arm',choices=('english','mixed'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);load=lambda p:torch.load(p,map_location='cpu',weights_only=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=a.root/f'results/s13-{a.arm}-main';m=json.load(gzip.open(run/'manifest.json.gz','rt'));parentpath=a.root/m['config']['parent_checkpoint'];assert sha(parentpath)==m['config']['parent_checkpoint_sha256'];parent=load(parentpath)
def eq(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):assert a.keys()==b.keys();[eq(v,b[k]) for k,v in a.items()]
 elif isinstance(a,(list,tuple)):assert len(a)==len(b);[eq(x,y) for x,y in zip(a,b)]
 else:assert a==b
hashes={}
for point in m['curves']:
 path=run/f'model-u{point["update"]}.pt';assert sha(path)==point['checkpoint_sha256'];x=load(path);assert x['update']==point['update'];assert {int(s['step']) for s in x['optimizer']['state'].values()}=={point['update']};assert {g['lr'] for g in x['optimizer']['param_groups']}=={1e-5};hashes[str(point['update'])]=sha(path)
 if point['added_update']==0:
  for k in ('model','optimizer','generator','schedule','order','position','visits'):eq(x[k],parent[k])
 assert sum(x['added_visits'])==point['added_update']*8;assert sum(x['visits'])==196608+point['added_update']*8
final=x;assert final['added_visits']==m['added_visits']==[4]*8192;assert final['renderer_visits']==m['renderer_visits'];phases=torch.randint(2,(8192,),generator=torch.Generator().manual_seed(913001)).tolist();assert final['renderer_phases']==m['renderer_phases']==phases;schedule=torch.Generator();schedule.set_state(parent['schedule']);order=parent['order'];position=parent['position'];visits=[0]*8192;sequence=hashlib.sha256()
for _ in range(4096):
 if position>=len(order):order=torch.randperm(8192,generator=schedule).tolist();position=0
 ids=order[position:position+8];position+=8;sequence.update(json.dumps(ids).encode())
 for i in ids:visits[i]+=1
assert sequence.hexdigest()==m['construction_sequence_sha256'];eq(schedule.get_state(),final['schedule']);assert visits==final['added_visits'];assert order==final['order'] and position==final['position'];expected=[[4,0] if a.arm=='english' else [2,2] for _ in range(8192)];assert final['renderer_visits']==expected
out=dict(arm=a.arm,checkpoint_sha256=hashes,initial_model_optimizer_sampler_equal_fixed_S11=True,all_saved_optimizer_steps_verified=True,final_steps=28672,added_presentations=32768,final_presentations=229376,each_construction_visits=4,renderer_visits_per_construction=expected[0],construction_schedule_and_renderer_phases_replayed=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU checkpoint/state/sampler inspection only. Cross-arm final negative RNG and common schedules are a separate pairing check once both main endpoints exist.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
