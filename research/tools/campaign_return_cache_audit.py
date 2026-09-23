"""CPU-only readout replay; no producer imports, model inference or optimization."""
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
m=json.loads((a.root/'manifest.json').read_text());cfg=m['config']
def sha(data):return hashlib.sha256(data).hexdigest()
def cache(name):
 r=m[name];packed=(a.root/r['path']).read_bytes();assert sha(packed)==r['sha256'];raw=gzip.decompress(packed);assert sha(raw)==r['uncompressed_sha256'];return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
f=cache('feature_cache');saved=cache('logit_cache')
for name,want in m['readout_hashes'].items():assert sha((a.root/name).read_bytes())==want
ck=Path(cfg['checkpoint']);assert sha(ck.read_bytes())==m['checkpoint_sha256'];original=torch.load(ck,map_location='cpu',weights_only=True)
mean,scale,coef=torch.load(a.root/'ridge.pt',map_location='cpu',weights_only=True);ce=torch.load(a.root/'ce.pt',map_location='cpu',weights_only=True)
x=torch.cat([f['train/2']['features'][d]for d in cfg['delays']]);torch.testing.assert_close(mean,x.mean(0),atol=2e-6,rtol=2e-6);torch.testing.assert_close(scale,x.std(0).clamp_min(.01),atol=2e-6,rtol=2e-6)
assert torch.equal(mean,ce['mean'])and torch.equal(scale,ce['scale'])
sets=[set(f[s+'/2']['event_row_hashes'])for s in cfg['data']];assert all(not(l&r)for i,l in enumerate(sets)for r in sets[i+1:]);checks=[]
for key,batch in f.items():
 for d,x in batch['features'].items():
  z=(x-mean)/scale
  replay=dict(unchanged=torch.nn.functional.linear(x,original['scalar_heads.0.weight'],original['scalar_heads.0.bias']),ridge=torch.cat((z,torch.ones(len(z),1)),1)@coef,ce=torch.nn.functional.linear(z,ce['state']['weight'],ce['state']['bias']))
  for arm,got in replay.items():
   wanted=saved[f'{key}/{d}/{arm}'];torch.testing.assert_close(got,wanted,atol=.001,rtol=.0001)
   checks.append(dict(cell=f'{key}/{d}/{arm}',rows=len(got),max_abs_difference=float((got-wanted).abs().max()),argmax_disagreements=int((got.argmax(-1)!=wanted.argmax(-1)).sum())))
out=dict(cpu_audit_wall_seconds=time.monotonic()-start,cells=checks,verified_readouts=len(checks),checkpoint_sha256=m['checkpoint_sha256'],train_normalization_verified=True,split_event_disjointness_verified=True,scope='CPU replay from frozen saved features; no backbone inference or refitting.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cpu_seconds=out['cpu_audit_wall_seconds'],cells=len(checks),argmax_disagreements=sum(r['argmax_disagreements']for r in checks)))
