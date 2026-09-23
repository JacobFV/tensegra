"""Per-seed marginal intervals and event-paired bootstrap, no gate changes."""
import argparse,gzip,json,math,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--replicates',type=int,default=4000);p.add_argument('--selected',default='ce');p.add_argument('--references',nargs='+',default=['unchanged','ridge']);a=p.parse_args();started=time.monotonic();rng=np.random.default_rng(912003);results=[]
def wilson(k,n):
 z=1.959963984540054;q=k/n;den=1+z*z/n;center=(q+z*z/(2*n))/den;half=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den;return [center-half,center+half]
for seed in (10,11,12):
 rows=json.load(gzip.open(a.root/str(seed)/'predictions.json.gz','rt'))
 for split in ('validation','test'):
  ix={(r['head'],r['target_delay'],r['distractors']):r for r in rows if r['split']==split};conditions=sorted({(d,k)for h,d,k in ix});n=len(next(iter(ix.values()))['targets']['value']);mat={}
  for head in a.references+[a.selected]:
   mat[head]=np.asarray([[int(a==b)for a,b in zip(ix[head,d,k]['predictions']['value'],ix[head,d,k]['targets']['value'])]for d,k in conditions],dtype=float).T
  weights=rng.multinomial(n,np.full(n,1/n),size=a.replicates);comparisons={}
  for reference in a.references:
   delta=mat[a.selected]-mat[reference];rep=weights@delta/n;intervals=np.quantile(rep,[.025,.975],axis=0)
   comparisons[reference]=[dict(delay=d,distractors=k,selected_minus_reference=float(delta[:,i].mean()),percentile95=intervals[:,i].tolist())for i,(d,k)in enumerate(conditions)]
  results.append(dict(seed=seed,split=split,unique_events=n,cells=[dict(arm=h,delay=d,distractors=k,correct=int(mat[h][:,i].sum()),total=n,wilson95=wilson(int(mat[h][:,i].sum()),n))for h in mat for i,(d,k)in enumerate(conditions)],paired_bootstrap=comparisons))
out=dict(selected_arm=a.selected,reference_arms=a.references,cpu_audit_wall_seconds=time.monotonic()-started,replicates=a.replicates,bootstrap_seed=912003,results=results,scope='Each seed/split resamples the same event indices jointly across arms/delays/distractors. Marginal Wilson95 and descriptive percentile intervals, not simultaneous guarantees. Different seeds remain separate. Empirical gate decisions are unchanged.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cpu_seconds=out['cpu_audit_wall_seconds'],seed_split_populations=len(results)))
