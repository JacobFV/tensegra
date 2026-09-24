import gzip,json,hashlib,time,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('repository',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();t=time.monotonic();base=args.repository;root=base/'research/results/campaign-01/semantics';summary=root/'s12-confirmation-summary.json';report=json.loads(summary.read_text());rows={};hashes={};identities=None
for s in (701,702,703):
 for arm in ('constant','decay'):
  p=root/f's12-{arm}-{s}/evaluation-u24576.json.gz';a=json.load(gzip.open(p,'rt'));assert a['update']==24576 and len(a['rows'])==1024;keys=[(r['seed'],r['semantic_sha256'],r['target']) for r in a['rows']]
  if identities is None:identities=keys
  else:assert keys==identities
  rows[s,arm]=a['rows'];hashes[str(p.relative_to(base))]=hashlib.sha256(p.read_bytes()).hexdigest()
results={}
for policy in ('raw_metrics','calibrated_metrics'):
 delta=np.array([[int(d[policy]['semantic_equivalence'])-int(c[policy]['semantic_equivalence']) for c,d in zip(rows[s,'constant'],rows[s,'decay'])] for s in (701,702,703)],float);rng=np.random.default_rng(12012);samples=[]
 for start in range(0,10000,128):
  ids=rng.integers(0,1024,(min(128,10000-start),1024));weights=np.array([np.bincount(i,minlength=1024) for i in ids],float);samples.append(weights@delta.T/1024)
 samples=np.concatenate(samples);actual=report['results'][policy];iv=actual['interval'];assert np.allclose(iv['per_seed_mean'],delta.mean(1),rtol=0,atol=1e-12);assert np.allclose(iv['per_seed_interval'],np.quantile(samples,[.025,.975],axis=0).T,rtol=0,atol=1e-12);assert np.allclose(iv['shared_event_mean_interval'],np.quantile(samples.mean(1),[.025,.975]),rtol=0,atol=1e-12);assert iv['mean_paired_gain']==delta.mean();assert actual['all_decayed_competent']==all(p['decay']>=103 for p in actual['pairs']);assert actual['replicated_directional_advantage']==bool((delta.mean(1)>0).all() and np.quantile(samples.mean(1),.025)>0)
 for p,s in zip(actual['pairs'],(701,702,703)):
  c=np.array([r[policy]['semantic_equivalence'] for r in rows[s,'constant']],bool);d=np.array([r[policy]['semantic_equivalence'] for r in rows[s,'decay']],bool)
  for key,v in dict(constant=c.sum(),decay=d.sum(),both_correct=(c&d).sum(),constant_only=(c&~d).sum(),decay_only=(~c&d).sum(),neither=(~c&~d).sum(),support=1024).items():assert p[key]==v
 results[policy]=actual
assert not results['calibrated_metrics']['all_decayed_competent'];hashes[str(summary.relative_to(base))]=hashlib.sha256(summary.read_bytes()).hexdigest();out=dict(status='independently_audited',input_sha256=hashes,results=results,all_six_actual_targets_and_order_exact=True,independent_bootstrap='Multinomial event multiplicities multiplied by3seed difference matrix; same events jointly resampled across fixedseeds;10000draws seed12012',cpu_audit_wall_seconds=time.monotonic()-t,scope='All six immutable raw endpoint audits precede this aggregate. Directional advantage passes; competence fails retained seed701. No threshold/seed/schema-policy selection.')
args.output.write_text(json.dumps(out,indent=2)+'\n');print(out['cpu_audit_wall_seconds'])
