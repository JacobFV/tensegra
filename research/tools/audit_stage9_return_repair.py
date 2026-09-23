"""Independent exact-field, intervention and selection audit for readout repair."""
import argparse,gzip,json,hashlib
from pathlib import Path
FIELDS=('value','type','operation','argument0','argument1','provenance')
def counts(pred,target):
 m={k:[a==b for a,b in zip(pred[k],target[k])]for k in FIELDS}
 for name,fields in [('joint',FIELDS),('scalar_joint',FIELDS[:3]),('identity_joint',FIELDS[3:])]:m[name]=[all(v)for v in zip(*(m[k]for k in fields))]
 result={k:dict(correct=sum(v),total=len(v))for k,v in m.items()};v=[x for x,t in zip(m['argument1'],target['argument1'])if t!=6];result['argument1_required']=dict(correct=sum(v),total=len(v));return result

def audit(root):
 manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config'];errors=[];total=0;gates=[];cells=[]
 for run in manifest['runs']:
  seed=run['seed']
  with gzip.open(root/f'{seed}-predictions.json.gz','rt') as f:rows=json.load(f)
  if max(run['ridge_selection'],key=lambda x:x['correct'])['alpha']!=run['chosen_alpha']:errors.append([seed,'ridge_selection'])
  if max(run['ce_curve'],key=lambda x:x['correct'])['step']!=run['chosen_ce_step']:errors.append([seed,'ce_selection'])
  for r in rows:
   total+=1;cells.append((seed,r['split'],r['arm'],r['distractors'],r['steps'],r['intervention']));calculated=counts(r['predictions'],r['targets'])
   if calculated!=r['counts']:errors.append([seed,r['split'],r['arm'],r['steps'],'counts'])
   if 'supplied_targets'in r:
    expected={k:v[:]for k,v in r['targets'].items()}
    if r['intervention']=='wrong_value':expected['value']=[32-v if v!=16 else 18 for v in expected['value']]
    if r['intervention']=='wrong_type':expected['type']=[(v+1)%3 for v in expected['type']]
    if r['intervention']=='wrong_provenance':expected['provenance']=[(v+1)%4 for v in expected['provenance']]
    if expected!=r['supplied_targets']:errors.append([seed,'supplied_targets'])
    if counts(r['predictions'],expected)!=r['supplied_fact_counts']:errors.append([seed,'supplied_counts'])
  for arm in ('unchanged','ridge','ce_refit'):
   failed=[]
   for d in cfg['eval_distractors']:
    selected=[r for r in rows if r['split']=='validation' and r['steps']==16 and r['distractors']==d and r['intervention']=='none' and r['arm']==arm]
    if len(selected)!=1:failed.append([d,'missing']);continue
    c=counts(selected[0]['predictions'],selected[0]['targets'])
    for k in FIELDS:
     v=c['argument1_required' if k=='argument1'else k]
     if c['joint']['total']<512 or not v['total'] or v['correct']/v['total']<=(.99 if k in ('type','operation')else.98):failed.append([d,k])
   gates.append(dict(seed=seed,arm=arm,passed=not failed,failed=failed))
  for r in rows:
   if r['arm']!='unchanged':
    baseline=next(x for x in rows if x['arm']=='unchanged' and all(x[k]==r[k]for k in ('split','steps','distractors','intervention')))
    if baseline['targets']!=r['targets'] or baseline['event_sha256']!=r['event_sha256']:errors.append([seed,'pairing'])
    if any(baseline['predictions'][k]!=r['predictions'][k]for k in FIELDS[1:]):errors.append([seed,'nonvalue_head_changed'])
 expected={(seed,split,arm,d,step,intervention) for seed in cfg['seeds'] for split in ('validation','test') for arm in ('unchanged','ridge','ce_refit') for d in cfg['eval_distractors'] for step,intervention in [(step,'none')for step in cfg['eval_delays']]+[(16,i)for i in cfg.get('interventions',[])]}
 if set(cells)!=expected or len(cells)!=len(expected):errors.append(['incomplete_or_duplicate_matrix'])
 return dict(rows=total,complete_matrix=set(cells)==expected,runs=len(manifest['runs']),gates=gates,complete_seeds=sorted(r['seed']for r in manifest['runs'])==[10,11,12],errors=errors,passed=bool(total)and not errors)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='gates'}));raise SystemExit(0 if r['passed']else 1)
