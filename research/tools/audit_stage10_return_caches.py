"""CPU-only replay of archived coefficients on frozen captured workspaces."""
import gzip,hashlib,io,json,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());out=[]
for run in m['runs']:
 seed=run['seed'];f=p/run['feature_cache']['path'];payload=f.read_bytes();assert hashlib.sha256(payload).hexdigest()==run['feature_cache']['sha256'];blob=gzip.decompress(payload);assert hashlib.sha256(blob).hexdigest()==run['feature_cache']['uncompressed_sha256'];cache=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
 sets=[set(cache[f'{s}/2']['event_row_hashes']) for s in ('train','calibration','test')];assert all(not(a&b) for i,a in enumerate(sets) for b in sets[i+1:])
 for s in ('calibration','test'):assert cache[f'{s}/2']['event_row_hashes']==cache[f'{s}/8']['event_row_hashes']
 rows=[json.loads(line) for line in gzip.open(p/f'{seed}-predictions.jsonl.gz','rt')];fits={}
 for fit in run['ridge_fit_records']:
  file=p/f'{seed}-{fit["head"]}-fit.pt';assert hashlib.sha256(file.read_bytes()).hexdigest()==fit['fit_sha256'];fits[fit['head']]=torch.load(file,map_location='cpu',weights_only=True)
  size=len(cache['train/2']['labels']);ds=fit['fit_delays'];fitrows=fit['rows'];q,rem=divmod(fitrows,len(ds));perm=torch.randperm(size,generator=torch.Generator().manual_seed(0));pairs=[(d,int(i)) for j,d in enumerate(ds) for i in perm.roll(j*(size//len(ds)))[:q+int(j<rem)]]
  assert hashlib.sha256(json.dumps(pairs).encode()).hexdigest()==fit['selection_sha256']
  x=torch.stack([cache['train/2']['features'][d][i] for d,i in pairs]);mean,scale,_=fits[fit['head']]
  torch.testing.assert_close(mean,x.mean(0),rtol=1e-5,atol=1e-6);torch.testing.assert_close(scale,x.std(0).clamp_min(.01),rtol=1e-5,atol=1e-6)
 count=0
 for row in rows:
  if row['head']=='original':continue
  mean,scale,w=fits[row['head']];x=cache[f'{row["split"]}/{row["distractors"]}']['features'][row['target_delay']]
  z=(x-mean)/scale;z=torch.cat([z,torch.ones(len(z),1)],dim=1);pred=(z@w).argmax(-1).tolist();assert pred==row['predictions']['value'];count+=1
 out.append(dict(seed=seed,feature_hashes_verified=True,disjoint_event_partitions=True,fits_verified=len(fits),train_only_normalization_verified=True,prediction_rows_replayed=count))
print(json.dumps(dict(device='cpu',runs=out),indent=2))
