"""Independent raw reconstruction of frozen cross-delay return readouts."""
import argparse,gzip,hashlib,json
from pathlib import Path
FIELDS=('value','type','operation','argument0','argument1','provenance')
def counts(p,t):
 m={f:[a==b for a,b in zip(p[f],t[f])] for f in FIELDS}
 m['joint']=[all(v) for v in zip(*(m[f] for f in FIELDS))]
 m['nonvalue_joint']=[all(v) for v in zip(*(m[f] for f in FIELDS[1:]))]
 m['argument1_required']=[ok for ok,v in zip(m['argument1'],t['argument1']) if v!=6]
 return {k:dict(correct=sum(v),total=len(v)) for k,v in m.items()}
def audit(root):
 manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config'];total=0;matrix=[];gates=[];retention=[]
 assert [r['seed'] for r in manifest['runs']]==[10,11,12]
 for run in manifest['runs']:
  seed=run['seed'];path=root/f'{seed}-predictions.jsonl.gz'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==run['prediction_sha256']
  rows=[json.loads(line) for line in gzip.open(path,'rt')];idx={}
  for r in rows:
   key=(r['head'],r['split'],r['target_delay'],r['distractors']);assert key not in idx;idx[key]=r
   assert counts(r['predictions'],r['targets'])==r['counts'];total+=1
  heads=['original']+[r['head'] for r in run['ridge_fit_records']]
  expected={(h,s,d,k) for h in heads for s in (['calibration','test'] if h=='original' else ['train','calibration','test']) for d in cfg['delays'] for k in ([2] if s=='train' else cfg['eval_distractors'])}
  assert set(idx)==expected
  for fit in run['ridge_fit_records']:
   chosen=max(fit['validation_grid'],key=lambda v:v['correct']);assert chosen['alpha']==fit['chosen_alpha']
   assert fit['input_dimensions']==1024 and fit['unique_events']==cfg['data']['train']['size']
   for v in fit['validation_grid']:
    assert sum(c['correct'] for c in v['cells'])==v['correct']
    assert {(c['delay'],c['distractors']) for c in v['cells']}=={(d,k) for d in fit['fit_delays'] for k in cfg['eval_distractors']}
   actual=sum(idx[(fit['head'],'calibration',d,k)]['counts']['value']['correct'] for d in fit['fit_delays'] for k in cfg['eval_distractors'])
   assert actual==chosen['correct']
  for r in rows:
   s=r['split'];d=r['target_delay'];k=r['distractors']
   if s=='train':continue
   original=idx[('original',s,d,k)]
   assert r['event_sha256']==original['event_sha256'] and r['targets']==original['targets']
   assert all(r['predictions'][f]==original['predictions'][f] for f in FIELDS[1:])
   reference=idx[('original',s,0,2)]
   assert r['event_sha256']==reference['event_sha256'] and r['targets']==reference['targets']
   if s=='test':matrix.append(dict(seed=seed,head=r['head'],delay=d,distractors=k,**r['counts']))
  for h in heads:
   for split in ('calibration','test'):
    failures=[]
    for k in cfg['eval_distractors']:
     c=idx[(h,split,16,k)]['counts']
     for f in FIELDS:
      v=c['argument1_required' if f=='argument1' else f]
      if not v['total'] or v['correct']/v['total']<=(.99 if f in ('type','operation') else .98):failures.append([k,f])
    retention.append(dict(seed=seed,head=h,split=split,passed=not failures,failed_fields=failures))
   for scope,delays in [('covered',[0,1,2,4,8,16]),('extrapolation',[32])]:
    selected=[idx[(h,'test',d,k)]['counts']['value'] for d in delays for k in cfg['eval_distractors']]
    gates.append(dict(seed=seed,head=h,scope=scope,passed=all(c['total']>=512 and c['correct']/c['total']>.98 for c in selected)))
 return dict(rows_verified=total,complete_matrix=True,seeds=[10,11,12],selection='Source-fit delays only; pooled head has no32 training/calibration',gates=gates,inherited_retention=retention,test_matrix=matrix)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=audit(a.root);a.output.write_text(json.dumps(out,indent=2)+'\n');print('Verified',out['rows_verified'],'rows')
