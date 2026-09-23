"""CPU-only readout replay; no producer imports, model inference or optimization."""
import argparse,gzip,hashlib,io,json,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
m=json.load(gzip.open(a.root/'manifest.json.gz','rt'));cfg=m['config']
def sha(data):return hashlib.sha256(data).hexdigest()
def cache(name):
 r=m[name];packed=(a.root/r['path']).read_bytes();assert sha(packed)==r['sha256'];raw=gzip.decompress(packed);assert sha(raw)==r['uncompressed_sha256'];return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
f=cache('cache');saved=cache('logits')
ck=Path(cfg['checkpoint']);assert sha(ck.read_bytes())==cfg['checkpoint_sha256'];original=torch.load(ck,map_location='cpu',weights_only=True)
heads={}
for fit in m['fits']:
 n=fit['events'];arm=fit['arm'];path=a.root/f'{arm}.pt';assert sha(path.read_bytes())==fit['checkpoint_sha256'];head=torch.load(path,map_location='cpu',weights_only=True)
 x=torch.cat([f[f'train_{arm}/2']['features'][d][:n]for d in cfg['delays']]);torch.testing.assert_close(head['mean'],x.mean(0),atol=2e-6,rtol=2e-6);torch.testing.assert_close(head['scale'],x.std(0).clamp_min(.01),atol=2e-6,rtol=2e-6)
 assert f[f'train_{arm}/2']['event_row_hashes'][:n]==fit['training_event_hashes'];heads[arm]=head
sets=[set(f[s+'/2']['event_row_hashes'])for s in cfg['data']];
sets.extend(set(f[s+'/8']['event_row_hashes'])for s in cfg['grids'])
for arm in heads:
 train=set(f[f'train_{arm}/2']['event_row_hashes']);assert all(not(train&x)for x in sets)
assert all(not(l&r)for i,l in enumerate(sets)for r in sets[i+1:]);checks=[]
for key,batch in f.items():
 for d,x in batch['features'].items():
  replay=dict(unchanged=torch.nn.functional.linear(x,original['scalar_heads.0.weight'],original['scalar_heads.0.bias']))
  for name,head in heads.items():replay[name]=torch.nn.functional.linear((x-head['mean'])/head['scale'],head['state']['weight'],head['state']['bias'])
  for arm,got in replay.items():
   wanted=saved[f'{key}/{d}/{arm}'];torch.testing.assert_close(got,wanted,atol=.001,rtol=.0001)
   checks.append(dict(cell=f'{key}/{d}/{arm}',rows=len(got),max_abs_difference=float((got-wanted).abs().max()),argmax_disagreements=int((got.argmax(-1)!=wanted.argmax(-1)).sum())))
out=dict(cpu_audit_wall_seconds=time.monotonic()-start,cells=checks,verified_readouts=len(checks),checkpoint_sha256=cfg['checkpoint_sha256'],train_normalization_verified=True,split_event_disjointness_verified=True,scope='CPU replay from frozen saved features; no backbone inference or refitting.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cpu_seconds=out['cpu_audit_wall_seconds'],cells=len(checks),argmax_disagreements=sum(r['argmax_disagreements']for r in checks)))
