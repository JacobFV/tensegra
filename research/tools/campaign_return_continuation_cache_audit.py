"""CPU frozen-feature replay and optimizer/RNG provenance for R11."""
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=json.load(gzip.open(a.root/'manifest.json.gz','rt'));c=m['config'];old=json.load(gzip.open(c['reference_manifest'],'rt'));fit=next(r for r in old['fits'] if r['arm']==c['reference_arm']);features_path=Path(c['feature_path']);assert sha(features_path)==c['feature_sha256'];f=torch.load(io.BytesIO(gzip.decompress(features_path.read_bytes())),map_location='cpu',weights_only=True);r=m['logits'];packed=(a.root/r['path']).read_bytes();assert hashlib.sha256(packed).hexdigest()==r['sha256'];raw=gzip.decompress(packed);assert hashlib.sha256(raw).hexdigest()==r['uncompressed_sha256'];saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
for k in ('checkpoint','reference_path','reference_manifest','reference_predictions'):assert sha(c[k])==c[k+'_sha256']
reference=torch.load(c['reference_path'],map_location='cpu',weights_only=True);heads={};states={}
for arm,path,hashvalue,steps in [('reference','replayed_reference.pt',m['replay']['checkpoint_sha256'],c['replay_steps'])]+[(r['arm'],r['arm']+'.pt',r['checkpoint_sha256'],c['replay_steps']+c['extra_steps']) for r in m['fits']]:
 assert sha(a.root/path)==hashvalue;s=torch.load(a.root/path,map_location='cpu',weights_only=True);heads[arm]=s;assert s['step']==steps;assert {int(v['step']) for v in s['optimizer']['state'].values()}=={steps};assert len(s['optimizer']['state'])==2;states[arm]=dict(checkpoint_sha256=hashvalue,optimizer_steps=steps,sampled_index_sha256=s['sampled_index_sha256'],rng_sha256=hashlib.sha256(s['rng_state'].numpy().tobytes()).hexdigest())
 for k in ('mean','scale'):assert torch.equal(s[k],reference[k])
 if arm=='reference':
  assert all(torch.equal(v,reference['state'][k]) for k,v in s['state'].items());assert states[arm]['rng_sha256']==m['replay']['final_rng_sha256'];assert s['sampled_index_sha256']==m['replay']['sampled_index_sha256']
 else:
  rec=next(r for r in m['fits'] if r['arm']==arm);assert s['sampled_index_sha256']==rec['sampled_index_sha256'];assert states[arm]['rng_sha256']==rec['final_rng_sha256'];assert s['optimizer']['param_groups'][0]['lr']==c['forks'][arm]
assert torch.equal(heads['constant']['rng_state'],heads['decay']['rng_state']);assert heads['constant']['sampled_index_sha256']==heads['decay']['sampled_index_sha256'];x=torch.cat([f['train/2']['features'][d] for d in c['delays']]);torch.testing.assert_close(reference['mean'],x.mean(0),atol=2e-6,rtol=2e-6);torch.testing.assert_close(reference['scale'],x.std(0).clamp_min(.01),atol=2e-6,rtol=2e-6);checks=[]
for key,batch in f.items():
 for d,x in batch['features'].items():
  for arm,s in heads.items():
   got=torch.nn.functional.linear((x-s['mean'])/s['scale'],s['state']['weight'],s['state']['bias']);wanted=saved[f'{key}/{d}/{arm}'];torch.testing.assert_close(got,wanted,atol=.001,rtol=.0001);diff=int((got.argmax(-1)!=wanted.argmax(-1)).sum());assert diff==0;checks.append(dict(cell=f'{key}/{d}/{arm}',rows=len(x),argmax_disagreements=diff,max_abs_difference=float((got-wanted).abs().max())))
out=dict(states=states,cells=checks,verified_readouts=len(checks),mean_scale_exact_to_historical=True,normalization_recomputed=True,fork_rng_order_equal=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='CPU saved linear-head replay only; no GPU/backbone/optimizer execution. Historical optimizer reconstructed by exact GPU replay, now endpoint moments/steps archived and checked; unavailable historical sampled-order hash not claimed.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cells=len(checks),seconds=out['cpu_audit_wall_seconds']))
