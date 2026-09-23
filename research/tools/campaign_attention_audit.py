"""Independent A01 compact task/suffix/path reconstruction; no producer imports."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
cfg=json.loads((a.root/'config.json').read_text());manifest=json.loads((a.root/'manifest.json').read_text());sources=json.loads((a.root/'source.json').read_text())
assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==manifest['config_sha256'];assert cfg['width']==manifest['width']==1024
for name,want in sources.items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{a.source_ref}:src/topoformer/{name}'])).hexdigest()==want
cells=[]
for f in sorted(a.root.glob('eval-*.npz')):
 raw=np.load(f);reported=json.loads(f.with_suffix('.json').read_text())
 for ci,row in enumerate(reported['rows']):
  get=lambda k:raw[f'c{ci}_{k}'];pred,gold=get('pred'),get('gold');route,start,rel,succ=get('route'),get('start'),get('relation'),get('successor');b,d,n=pred.shape;bi=np.arange(b)
  assert b==row['examples']==cfg['eval_examples'];assert rel.shape==(b,d)and succ.shape==(b,3,n)
  correct=pred==gold;values=dict(task=correct[:,-1][bi,start],all_node=correct.mean((1,2)),suffix_value_trajectory=correct.all((1,2)))
  # Gold suffix state t must follow its current public relation from state t-1.
  for t in range(1,d):assert np.array_equal(gold[:,t],np.take_along_axis(gold[:,t-1],succ[bi,rel[:,d-1-t]],axis=1))
  chosen=start.copy();actual=start.copy();path=np.ones(b,dtype=bool)
  for t in reversed(range(d)):
   chosen=route[bi,t,chosen];actual=succ[bi,rel[:,d-1-t],actual];path &= chosen==actual
  values['exact_pointer_path']=path
  for key,v in values.items():np.testing.assert_allclose(v,get(key),atol=1e-7,rtol=1e-6);assert abs(float(v.mean())-row[key])<1e-6
  mass=get('edge_mass');assert np.all((mass>=-1e-6)&(mass<=1+1e-6));assert abs(float(mass.mean())-row['edge_mass'])<1e-6
  cells.append(dict(artifact=f.name,condition=row['condition'],examples=b,task_correct=int(values['task'].sum()),pointer_correct=int(path.sum()),suffix_correct=int(values['suffix_value_trajectory'].sum())))
out=dict(cpu_audit_wall_seconds=time.monotonic()-started,source_ref=a.source_ref,mode=cfg['mode'],cells=cells,scope='Task, all-node, suffix and mean-head-argmax path reconstructed. Edge-mass mean/bounds checked only; compact archive does not contain attention weights. No inference rerun.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cells=len(cells),cpu_seconds=out['cpu_audit_wall_seconds']))
