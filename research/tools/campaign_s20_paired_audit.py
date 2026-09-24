"""Independent S20 paired audit from independently reconstructed exact flags."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
for name in ('workspace','record','analysis','output'): p.add_argument(name,type=Path)
a=p.parse_args(); start=time.monotonic()
w=json.loads(a.workspace.read_text());r=json.loads(a.record.read_text());reported=json.loads(a.analysis.read_text())
assert w['status']==r['status']=='pass' and w['graphs']==13056 and r['graphs']==6528 and w['global_cutoffs']==78
seeds=('701','702','703');cells=('3x3','3x4','4x3','4x4');known=('3x3','4x3','4x4')
flags={s:{**w['flags'][s],**r['flags'][s]} for s in seeds};identities={}
for s in seeds:
 assert set(flags[s])=={'original','context','record'}
 for arm,policies in flags[s].items():
  assert set(policies)==({'categorical'} if arm=='record' else {'raw','matched'})
  for policy,cc in policies.items():
   assert set(cc)==set(cells)
   for cell,events in cc.items():
    assert len(events)==512 and all(type(v)is bool for v in events.values())
    ids=sorted(events);assert ids==identities.setdefault(cell,ids)
assert len(set().union(*(set(ids) for ids in identities.values())))==2048
# Bootstrap sample multiplicities: same draws reused across every comparison/seed.
rng=np.random.default_rng(20020);weights=[]
for cell in known:
 draws=rng.integers(0,512,size=(10000,512));offset=512*np.arange(10000)[:,None]
 weights.append(np.bincount((draws+offset).ravel(),minlength=5120000).reshape(10000,512).astype(float))
def array(seed,arm,policy,cell):
 return np.array([flags[seed][arm][policy][cell][i] for i in identities[cell]],dtype=float)
counts={};paired={};distributions={};per_seed={}
for s in seeds:
 counts[s]={cell:int(array(s,'record','categorical',cell).sum()) for cell in cells};paired[s]={};superiority=True
 for arm in ('original','context'):
  for policy in ('raw','matched'):
   key=arm+'/'+policy;cell_results={};distribution=np.zeros(10000);delta=0
   for cell in cells:
    new=array(s,'record','categorical',cell);old=array(s,arm,policy,cell);difference=new-old
    cell_results[cell]=dict(examples=512,record_complete=int(new.sum()),baseline_complete=int(old.sum()),delta_complete=int(difference.sum()),transitions_baseline_to_record={f'{i}->{j}':int(((old==i)&(new==j)).sum()) for i in (0,1) for j in (0,1)})
    if cell in known:
     delta+=int(difference.sum());distribution+=weights[known.index(cell)]@difference/1536
   ci=(np.quantile(distribution,[.025,.975])*100).tolist();distributions.setdefault(key,[]).append(distribution)
   paired[s][key]=dict(cells=cell_results,known_delta_complete=delta,known_macro_delta_percentage_points=100*delta/1536,conditional_event_ci95_percentage_points=ci)
   if policy=='matched':superiority &= delta>0 and ci[0]>0
   given=reported['paired'][s][key];assert given['cells']==cell_results and given['known_delta_complete']==delta
   assert np.allclose(given['conditional_event_ci95_percentage_points'],ci,rtol=0,atol=1e-10)
   assert abs(given['known_macro_delta_percentage_points']-100*delta/1536)<1e-10
 g=dict(known_macro_competence=sum(counts[s][c] for c in known)>=1383,each_known_cell_competence=min(counts[s][c] for c in known)>=410,heldout_competence=counts[s]['3x4']>=52,superiority_vs_both=bool(superiority))
 g['known_claim']=g['known_macro_competence'] and g['each_known_cell_competence'] and g['superiority_vs_both'];per_seed[s]=g
average={key:(np.quantile(np.mean(ds,axis=0),[.025,.975])*100).tolist() for key,ds in distributions.items()}
for key,ci in average.items():assert np.allclose(ci,reported['three_fixed_lineage_mean_conditional_ci95_percentage_points'][key],rtol=0,atol=1e-10)
decisions=dict(per_seed=per_seed,replicated_known_claim=all(g['known_claim'] for g in per_seed.values()),heldout_claim=all(g['heldout_competence'] for g in per_seed.values()),automatic_extension=False)
assert counts==reported['counts'] and decisions==reported['decisions']
assert reported['bootstrap']==dict(seed=20020,draws=10000,strata=list(known),events_per_stratum=512)
out=dict(status='pass',counts=counts,paired=paired,decisions=decisions,three_fixed_lineage_mean_conditional_ci95_percentage_points=average,seconds=time.monotonic()-start,analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),scope='All independent raw exact flags, paired transitions and multiplicity-weighted stratified shared-event bootstrap. Fixed lineages; no population-of-seeds interval or heldout competence promotion.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('paired',)})
