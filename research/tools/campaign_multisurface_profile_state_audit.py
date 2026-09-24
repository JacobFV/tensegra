import torch,json,gzip,hashlib,time,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2);tick=time.monotonic();root=args.root;load=lambda p:torch.load(p,map_location='cpu',weights_only=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):assert a.keys()==b.keys();[equal(v,b[k]) for k,v in a.items()]
 elif isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
 else:assert a==b
manifests=[json.load(gzip.open(root/f'results/s13-{arm}-profile/manifest.json.gz','rt')) for arm in ('english','mixed')];parentpath=root/manifests[0]['config']['parent_checkpoint'];assert sha(parentpath)==manifests[0]['config']['parent_checkpoint_sha256'];parent=load(parentpath);finals=[];hashes={}
for arm,m in zip(('english','mixed'),manifests):
 for curve in m['curves']:
  p=root/f'results/s13-{arm}-profile/model-u{curve["update"]}.pt';assert sha(p)==curve['checkpoint_sha256'];x=load(p);assert x['update']==curve['update'] and {int(s['step']) for s in x['optimizer']['state'].values()}=={curve['update']};hashes[f'{arm}/{x["update"]}']=sha(p)
  if curve['added_update']==0:
   for k in ('model','optimizer','generator','schedule','order','position','visits'):equal(x[k],parent[k])
  else:finals.append(x);assert sum(x['added_visits'])==160 and sum(x['visits'])==196768;assert x['added_visits']==m['added_visits'] and x['renderer_visits']==m['renderer_visits'];phases=torch.randint(2,(8192,),generator=torch.Generator().manual_seed(913001)).tolist();assert x['renderer_phases']==phases
for k in ('generator','schedule','order','position','visits','added_visits','renderer_phases'):equal(finals[0][k],finals[1][k])
schedule=torch.Generator();schedule.set_state(parent['schedule']);order=parent['order'];position=parent['position'];visits=[0]*8192;sequence=hashlib.sha256()
for _ in range(20):
 if position>=len(order):order=torch.randperm(8192,generator=schedule).tolist();position=0
 ids=order[position:position+8];position+=8;sequence.update(json.dumps(ids).encode())
 for i in ids:visits[i]+=1
assert sequence.hexdigest()==manifests[0]['construction_sequence_sha256'];equal(schedule.get_state(),finals[0]['schedule']);assert visits==finals[0]['added_visits']
out=dict(checkpoint_sha256=hashes,initial_model_optimizer_and_sampler_equal_fixed_S11=True,final_steps=24596,added_presentations=160,final_paired_sampler_exact=True,construction_schedule_and_renderer_phases_replayed=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='CPU state/byte/random-schedule inspection only; no model inference. Negative-pair equality uses manifests and identical final RNG state.');args.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
