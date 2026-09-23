"""Cross-seed graph pairing and event-resampled frozen attention intervention."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();seeds=[201,202,203];arms=['soft4','soft8','context','message','hard'];reference=None;deltas=[];result=[]
def wilson(k,n):
 z=1.959963984540054;ph=k/n;den=1+z*z/n;mid=(ph+z*z/(2*n))/den;rad=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den;return[mid-rad,mid+rad]
for seed in seeds:
 for arm in arms:
  root=a.root/f'{arm}-{seed}';raw=np.load(root/'eval-00500.npz');rows=json.loads((root/'eval-00500.json').read_text())['rows'];primary=next(i for i,r in enumerate(rows)if r['condition']==dict(nodes=64,depth=16,data_group=3));draws={f:raw[f'c{primary}_{f}']for f in ('gold','start','relation','successor')}
  if reference is None:reference=draws
  else:
   for f in draws:assert np.array_equal(draws[f],reference[f])
  outcome=raw[f'c{primary}_task'];k=int(outcome.sum());result.append(dict(seed=seed,arm=arm,primary_correct=k,examples=len(outcome),wilson95=wilson(k,len(outcome))))
  if arm=='soft4':
   j=next(i for i,r in enumerate(rows)if r['condition']==dict(nodes=64,depth=16,data_group=3,strength_override=8.));fixed=raw[f'c{j}_task'];deltas.append(fixed.astype(float)-outcome.astype(float));assert fixed.all()
rng=np.random.default_rng(912004);deltas=np.stack(deltas);sample=np.array([deltas[:,rng.integers(1024,size=1024)].mean()for _ in range(4000)])
out=dict(primary=result,unique_primary_graph_draws=1024,seed_replicates=3,frozen_override8_minus_soft4=dict(mean=float(deltas.mean()),paired_event_bootstrap95=np.quantile(sample,[.025,.975]).tolist(),bootstrap_samples=4000),cpu_audit_wall_seconds=time.monotonic()-start,scope='Same1024 primary graph draws across all seeds/arms. Bootstrap resamples graph index jointly across seeds and arms; conditional on these three fitted seeds, not a population guarantee over initialization. Secondary/corruption conditions not pooled as independent draws.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
