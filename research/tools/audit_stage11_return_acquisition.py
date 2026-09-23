"""Independent six-field fixed-set return acquisition audit, no inference."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
FIELDS=('value','type','operation','argument0','argument1','provenance')
def counts(p,t):
 m={f:[a==b for a,b in zip(p[f],t[f])]for f in FIELDS}
 m['joint']=[all(v)for v in zip(*(m[f]for f in FIELDS))];m['nonvalue_joint']=[all(v)for v in zip(*(m[f]for f in FIELDS[1:]))]
 m['argument1_required']=[ok for ok,y in zip(m['argument1'],t['argument1'])if y!=6]
 return {k:dict(correct=sum(v),total=len(v))for k,v in m.items()}
def audit(root):
 x=json.loads((root/'manifest.json').read_text());cfg=x['config'];repo=Path(__file__).resolve().parents[2]
 assert x['width']==1024 and cfg['mode']=='acquisition'
 assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==x['config_sha256']
 for name,h in x['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show','1b2860b:src/topoformer/'+name],cwd=repo)).hexdigest()==h
 assert x['checkpoint_sha256']==cfg['checkpoint_sha256'];summary=[];allrows={};raw_count=0;hashes=[]
 for run in x['runs']:
  arm=run['arm'];schedule=cfg['schedules'][arm];assert max(schedule)<=16
  file=root/f'{arm}-predictions.json.gz';assert hashlib.sha256(file.read_bytes()).hexdigest()==run['predictions_sha256']
  with gzip.open(file,'rt')as f:rows=json.load(f)
  expected={(s,u,d)for s,us in [('fixed',range(0,cfg['updates']+1,cfg['check_every'])),('fresh',[None])]for u in us for d in set(schedule)};seen=set()
  fixedhashes=None;freshhashes=None
  for row in rows:
   key=(row['split'],row.get('update'),row['target_delay']);assert key not in seen;seen.add(key)
   assert row['counts']==counts(row['predictions'],row['targets']);raw_count+=1
   assert row['distractors']==2 and row['counts']['joint']['total']==(cfg['batch_size']if row['split']=='fixed'else cfg['fresh_size'])
   if row['split']=='fixed':
    if fixedhashes is None:fixedhashes=row['event_row_hashes']
    assert row['event_row_hashes']==fixedhashes
   else:
    if freshhashes is None:freshhashes=row['event_row_hashes']
    assert row['event_row_hashes']==freshhashes
  assert seen==expected and not set(fixedhashes)&set(freshhashes)
  counts_delay={str(d):sum(schedule[i%len(schedule)]==d for i in range(cfg['updates']))for d in set(schedule)}
  assert counts_delay==run['schedule_counts']and run['presentations']==cfg['updates']*cfg['batch_size']
  final=[r for r in rows if r.get('update')==cfg['updates']];passed=all(r['counts']['joint']['correct']==cfg['batch_size']for r in final)
  assert passed==run['passed_fixed_gate'];hashes.append((run['initial_state_sha256'],run['event_sha256']));allrows[arm]=rows
  summary.append(dict(arm=arm,passed_fixed_gate=passed,final=[dict(delay=r['target_delay'],counts=r['counts'])for r in final],fresh=[dict(delay=r['target_delay'],counts=r['counts'])for r in rows if r['split']=='fresh']))
 assert len(set(hashes))==1 and set(allrows)==set(cfg['schedules'])
 common=set(cfg['schedules']['short'])&set(cfg['schedules']['wide'])
 for d in common:
  a=next(r for r in allrows['short']if r.get('update')==0 and r['target_delay']==d);b=next(r for r in allrows['wide']if r.get('update')==0 and r['target_delay']==d)
  assert a['predictions']==b['predictions'] and a['targets']==b['targets'] and a['event_row_hashes']==b['event_row_hashes']
 return dict(rows_verified=raw_count,paired_initial_predictions=True,disjoint_fixed_fresh_events=True,arms=summary,dependent_main_allowed=all(r['passed_fixed_gate']for r in summary),scope='Single independent development checkpoint; fixed32 fitting gate and fresh512 diagnostics, not three-seed main generalization.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=audit(a.root);a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='arms'})
