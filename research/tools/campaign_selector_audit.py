"""Independent content-selector task/path and counterfactual reconstruction."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();cfg=json.loads((a.root/'config.json').read_text());m=json.loads((a.root/'manifest.json').read_text());assert cfg['width']==m['width']==1024;assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['config_sha256']
for name,want in json.loads((a.root/'source.json').read_text()).items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{a.source_ref}:src/topoformer/{name}'])).hexdigest()==want
cells=[]
for f in sorted(a.root.glob('eval-*.npz')):
 step=int(f.stem.split('-')[1]);curve=step!=cfg['steps'] and 'curve_eval_seed' in cfg
 raw=np.load(f);report=json.loads(f.with_suffix('.json').read_text());assert report['eval_seed']==cfg['curve_eval_seed' if curve else 'eval_seed'];rows=report['rows'];assert [r['condition']for r in rows]==cfg['curve_conditions' if curve else 'conditions']
 for ci,row in enumerate(rows):
  get=lambda k:raw[f'c{ci}_{k}'];pred,gold,route,succ,startnode=get('pred'),get('gold'),get('route'),get('successor'),get('start');b,d,n=pred.shape;assert b==cfg['curve_examples' if curve else 'eval_examples'];bi=np.arange(b);ok=pred==gold
  actual=startnode.copy();proposed=actual.copy();path=np.ones(b,dtype=bool)
  for t in range(d):actual=succ[bi,t,actual];proposed=route[bi,d-1-t,proposed];path&=actual==proposed
  assert np.array_equal(actual,get('oracle_terminal'))
  for t in range(1,d):assert np.array_equal(gold[:,t],np.take_along_axis(gold[:,t-1],succ[:,d-1-t],axis=1))
  values=dict(task=ok[:,-1][bi,startnode],all_node=ok.mean((1,2)),suffix_value_trajectory=ok.all((1,2)),exact_pointer_path=path,agreement_supplied_task=pred[:,-1][bi,startnode]==get('supplied_final_target'),changed_terminal=actual!=get('original_terminal'))
  condition=row['condition'];base=next(j for j,r in enumerate(rows)if r['condition']=={k:v for k,v in condition.items()if k in ('nodes','depth','groups','composition','data_group')})
  original_gold=raw[f'c{base}_gold'][:,-1][bi,raw[f'c{base}_start']];values['changed_answer']=gold[:,-1][bi,startnode]!=original_gold
  for key,v in values.items():np.testing.assert_allclose(v,get(key),atol=1e-7,rtol=1e-6);assert abs(v.mean()-row[key])<1e-6
  for name in ('selected_mass','edge_mass'):
   v=get(name);assert np.all((v>=-1e-6)&(v<=1+1e-6));assert abs(v.mean()-row[name])<1e-6
  for name in ('terminal','answer'):
   mask=values['changed_'+name];assert int(mask.sum())==row['changed_'+name+'_count'];expected=float(values['task'][mask].mean())if mask.any()else None;assert expected==row['task_given_changed_'+name]
  if condition.get('wrong_instruction'):
   swap=next(j for j,r in enumerate(rows)if r['condition'].get('instruction_swap')and r['condition']['data_group']==condition['data_group']);assert np.array_equal(get('supplied_final_target'),raw[f'c{swap}_gold'][:,-1][bi,raw[f'c{swap}_start']])
  cells.append(dict(step=int(f.stem.split('-')[1]),condition=condition,examples=b,task_correct=int(values['task'].sum()),path_correct=int(path.sum()),supplied_correct=int(values['agreement_supplied_task'].sum()),changed_answer_count=int(values['changed_answer'].sum()),changed_terminal_count=int(values['changed_terminal'].sum())))
out=dict(mode=cfg['mode'],source_ref=a.source_ref,cells=cells,cpu_audit_wall_seconds=time.monotonic()-start,scope='Task/path/gold-transition/supplied-target and changed-outcome metrics independently reconstructed. Selected/edge mass means and bounds only; no saved full attention weights. Per-cell archived metrics; curve and final populations verified separately when configured.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(mode=cfg['mode'],cells=len(cells),seconds=out['cpu_audit_wall_seconds']))
