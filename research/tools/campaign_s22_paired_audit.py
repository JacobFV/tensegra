"""Independent complete S22 paired tables/conditional bootstrap from audited flags."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ('raw_audit','analysis','output'):p.add_argument(k,type=Path)
a=p.parse_args();start=time.monotonic();raw=json.loads(a.raw_audit.read_text());d=json.loads(a.analysis.read_text())
assert raw['status']=='PASS' and raw['graphs']==21504 and raw['policies']==6
cells=('2x4','3x3','3x4','4x3','4x4','5x3','5x4');policies=('oracle_node_count','oracle_node_kinds','oracle_node_prefix');rng=np.random.default_rng(22023);verified={}
for cell in cells:
 draws=rng.integers(0,512,size=(10000,512));weights=np.zeros((10000,512),dtype=np.int16)
 for i,indices in enumerate(draws):weights[i]=np.bincount(indices,minlength=512)
 for arm in ('original','broad'):
  ids=sorted(raw['baseline'][arm][cell]);assert len(ids)==512
  for policy in policies:
   f=raw['flags'][arm][policy][cell];assert sorted(f)==ids;new=np.array([f[i] for i in ids],dtype=np.int16)
   summary=raw['summaries'][arm][policy]
   for k in ('examples','valid','complete','controller_stats','invalid_reasons'):assert summary[k]==d['results'][arm][policy][k]
   for k,v in summary['cells'][cell].items():assert v==d['results'][arm][policy]['cells'][cell][k]
   for comparator in ('public',)+policies[:policies.index(policy)]:
    oldf=raw['baseline'][arm][cell] if comparator=='public' else raw['flags'][arm][comparator][cell];assert sorted(oldf)==ids;old=np.array([oldf[i] for i in ids],dtype=np.int16)
    table={f'{x}->{y}':int(np.logical_and(old==x,new==y).sum()) for x in (0,1) for y in (0,1)};delta=new-old;key=f'{arm}/{cell}/{policy}-minus-{comparator}';assert d['paired'][key]==dict(transitions=table,delta_complete=int(delta.sum()))
    ci=np.percentile((weights@delta)/512*100,[2.5,97.5]).tolist();assert np.allclose(ci,d['conditional_event_delta_pp_ci95'][key],rtol=0,atol=1e-12);verified[key]=dict(transitions=table,ci95=ci)
assert len(verified)==84 and set(verified)==set(d['paired'])==set(d['conditional_event_delta_pp_ci95']);assert d['no_acquisition_gate'] and d['no_automatic_next_run']
out=dict(status='PASS',paired_comparisons=84,cpu_wall_seconds=time.monotonic()-start,analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),comparisons=verified,scope='Shared sorted event bootstrap draws per cell, conditional on two fixed models and three explicitly privileged policies; no learned-public or promotion inference.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='comparisons'})
