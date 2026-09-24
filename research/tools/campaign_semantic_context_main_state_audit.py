"""CPU audit of a closed S18 arm; no model construction or efficacy analysis."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics_data import load_cache,target
from topoformer.semantic_curriculum import sampled_pairs
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',choices=('context','workspace_control'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:torch.load(p,map_location='cpu',weights_only=True);c=json.loads(a.config.read_text());assert sha(a.config)=='cfe0359c80b3ed55a3a643b5dd7f823808eadbf256f50d18610501a5596462e0';assert c['job']=='main' and c['added_updates']==4096;bindings=Path(c['bindings']['path']);assert sha(bindings)==c['bindings']['sha256'];b=json.loads(bindings.read_text());root=Path(c['output_dir'])/a.arm;manifest=root/'manifest.json.gz';manifest_sha=sha(manifest);m=json.load(gzip.open(manifest,'rt'));assert m['arm']==a.arm and [x['added_update'] for x in m['curves']]==[0,1024,2048,4096];parent=load(b['parent']['path']);assert sha(b['parent']['path'])==b['parent']['sha256'];rows=load_cache(b['caches']['train_mixed']['path']);assert sha(b['caches']['train_mixed']['path'])==b['caches']['train_mixed']['sha256'];vocab=json.loads(Path(b['data_audit']['path']).read_text())['value_vocabulary'];hashes={};states={}
def eq(x,y):
 if torch.is_tensor(x):assert torch.equal(x,y)
 elif isinstance(x,dict):assert x.keys()==y.keys();[eq(v,y[k]) for k,v in x.items()]
 elif isinstance(x,(list,tuple)):assert len(x)==len(y);[eq(u,v) for u,v in zip(x,y)]
 else:assert x==y
def tensor_hash(state):
 d=hashlib.sha256()
 for name,v in sorted(state.items()):
  v=v.detach().cpu().contiguous();d.update(name.encode());d.update(str(v.dtype).encode());d.update(str(tuple(v.shape)).encode());d.update(v.numpy().tobytes())
 return d.hexdigest()
for curve in m['curves']:
 u=curve['update'];step=curve['added_update'];assert u==24576+step and curve['added_presentations']==step*8;path=root/f'model-u{u}.pt';h=sha(path);assert h==curve['checkpoint_sha256'];x=load(path);assert x['update']==u and x['added_update']==step and x['presentation_number']==step*8;assert {int(s['step']) for s in x['optimizer']['state'].values()}=={u};assert list(x['model'])==list(parent['model']);assert all(v.shape==parent['model'][k].shape and v.dtype==parent['model'][k].dtype for k,v in x['model'].items());assert len(x['optimizer']['state'])==98;assert sum(v.numel() for v in x['model'].values())==57853781;assert tensor_hash(x['model'])==curve['model_state_sha256'];assert all(torch.isfinite(v).all() for v in x['model'].values());assert all(torch.isfinite(v).all() for s in x['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
 if step==0:eq(x['model'],parent['model']);eq(x['optimizer'],parent['optimizer'])
 states[step]={k:x[k] for k in ('schedule','order','position','visits','presentation_number')};hashes[str(path)]=h
 for policy in ('matched','historical'):
  rec=curve[policy];ep=root/f'{policy}-u{u}'/rec['artifact'];assert sha(ep)==rec['sha256'];d=json.load(gzip.open(ep,'rt'));npz=ep.parent/d['calibration_data_artifact'];assert sha(npz)==d['calibration_data_sha256'];hashes[str(ep)]=sha(ep);hashes[str(npz)]=sha(npz)
 if step==4096:assert tensor_hash(x['model'])==m['final_state_sha256']
 del x
schedule=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=schedule).tolist();position=0;sequence=hashlib.sha256();pairs=hashlib.sha256();visits=[0]*4096;tokens=nodes=edges=0;gold={i:target(row,vocab) for i,row in enumerate(rows)};assert states[0]['order']==order and states[0]['position']==0 and states[0]['visits']==visits;eq(states[0]['schedule'],schedule.get_state())
for step in range(4096):
 if position+8>4096:order=torch.randperm(4096,generator=schedule).tolist();position=0
 ids=order[position:position+8];position+=8;sequence.update(json.dumps(ids).encode())
 for offset,i in enumerate(ids):
  row=rows[i];q=sampled_pairs(gold[i],torch.Generator().manual_seed(150150000+step*8+offset),128);pairs.update(q.numpy().tobytes());visits[i]+=1;tokens+=row['tokens'];nodes+=len(row['nodes']);edges+=len(row['edges'])
 if step+1 in states:
  s=states[step+1];assert s['order']==order and s['position']==position and s['visits']==visits;eq(s['schedule'],schedule.get_state())
assert visits==m['visits']==[8]*4096;streams=dict(construction_sequence_sha256=sequence.hexdigest(),pair_sequence_sha256=pairs.hexdigest(),optimizer_tokens=tokens,optimizer_nodes=nodes,optimizer_edges=edges);assert streams==b['expected_streams'];assert all(m[k]==v for k,v in streams.items());assert sha(manifest)==manifest_sha
out=dict(arm=a.arm,manifest_sha256=manifest_sha,config_sha256=sha(a.config),input_sha256=hashes,initial_model_and_AdamW_exact_parent=True,parameter_count=57853781,optimizer_slots=98,all32768_queries_sampler_visits_exposure_exact=True,final_AdamW_step=28672,all_four_checkpoint_tensor_fingerprints_verified=True,eight_evaluation_and_calibration_artifacts_hash_bound=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Closed arm structural/provenance audit only; no model construction/forward or efficacy comparison. Global completed-job receipt and independent final numerical audit remain required.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='input_sha256'})
