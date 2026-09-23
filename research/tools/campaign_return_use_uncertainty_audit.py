"""Event-level paired bootstrap for R05; delays are not independent samples."""
import gzip,json,time,math,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();rng=np.random.default_rng(903011);out=[]
def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return[c-h,c+h]
for seed in (10,11,12):
 rows=json.load(gzip.open(Path(a.root)/str(seed)/'predictions.json.gz','rt'));r={(x['arm'],x['key'],x['delay']):x for x in rows};record={'seed':seed,'clean':[],'causal':[]}
 for x in rows:
  if x['arm']=='learned' and x['key'].startswith(('validation/','test/')):record['clean'].append(dict(key=x['key'],delay=x['delay'],correct=x['original_correct'],total=x['total'],wilson95=wilson(x['original_correct'],x['total'])))
 # One index bootstrap shared across all delays per seed, preserving within-event dependence.
 n=r['learned','intervention_drop/8',0]['total'];ids=rng.integers(n,size=(2000,n))
 for d in (0,1,16):
  x=r['learned','validation/8',d];y=np.array(x['original_targets'][:n]);clean=np.array(x['predictions'][:n])==y;drop=np.array(r['learned','intervention_drop/8',d]['predictions'])==y
  delta=clean.astype(float)-drop;ci=np.quantile(delta[ids].mean(1),[.025,.975]);rec=dict(delay=d,drop_difference=float(delta.mean()),paired_event_bootstrap95=ci.tolist())
  for kind in ('wrong','swap'):
   x=r['learned',f'intervention_{kind}/8',d];rec[kind]=dict(correct=x['changed_supplied_correct'],total=x['changed_total'],wilson95=wilson(x['changed_supplied_correct'],x['changed_total']))
  record['causal'].append(rec)
 out.append(record)
Path(a.output).write_text(json.dumps(dict(seeds=out,bootstrap_replicates=2000,cpu_audit_wall_seconds=time.monotonic()-start,scope='Conditional event-sampling uncertainty per fitted model; three seeds listed separately, correlated delays not pooled.'),indent=2)+'\n')
