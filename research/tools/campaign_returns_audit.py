"""Independent R01 raw fields, grouped errors and calibration selection audit."""
import argparse,gzip,hashlib,json,subprocess,time
from pathlib import Path
from audit_stage11_return_main import counts,FIELDS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def groups(row):
 result={}
 for i,(pred,y) in enumerate(zip(row['predictions']['value'],row['targets']['value'])):
  v=(y-16)/2;keys={f:str(row['targets'][f][i])for f in FIELDS}
  keys.update(value_type_operation=f"{y}/{row['targets']['type'][i]}/{row['targets']['operation'][i]}",sign=str(-1 if v<0 else int(v>0)),magnitude=str(abs(v)))
  for group,key in keys.items():
   r=result.setdefault(group,{}).setdefault(key,dict(correct=0,total=0,signed_error_sum=0.,absolute_error_sum=0.));e=(pred-y)/2
   r['correct']+=pred==y;r['total']+=1;r['signed_error_sum']+=e;r['absolute_error_sum']+=abs(e)
 return result

def audit(root,ref):
 started=time.monotonic();m=json.loads((root/'manifest.json').read_text());cfg=m['config'];assert 32 not in cfg['delays']
 assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['config_sha256']
 for name,want in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{ref}:src/topoformer/{name}'])).hexdigest()==want
 p=root/'predictions.json.gz';assert sha(p)==m['predictions_sha256'];rows=json.load(gzip.open(p,'rt'));index={}
 for r in rows:
  key=r['split'],r['distractors'],r['target_delay'],r['head'];assert key not in index;index[key]=r
  assert counts(r)==r['counts']and groups(r)==r['groups'];assert r['counts']['value']['total']==cfg['data'][r['split']]['size']
 expected={(s,k,d,a)for s in cfg['data']for k in ([2]if s=='train'else cfg['eval_distractors'])for d in cfg['delays']for a in ('unchanged','ridge','ce')};assert set(index)==expected
 for key,r in index.items():
  s,k,d,a=key;reference=index[s,k,d,'unchanged'];assert r['targets']==reference['targets']and r['event_sha256']==reference['event_sha256']
  assert all(r['predictions'][f]==reference['predictions'][f]for f in FIELDS[1:])
  base=index[s,2,cfg['delays'][0],'unchanged'];assert r['targets']==base['targets']and r['event_sha256']==base['event_sha256']
 def score(c):return min(x['correct']for x in c['cells']),sum(x['correct']for x in c['cells'])
 for name,label,key in [('fit_candidates','ridge','alpha'),('ce_curve','ce','step')]:
  candidates=m[name]
  for c in candidates:assert list(score(c))==c['score']
  winner=max(candidates,key=score);assert winner[key]==m['selected_alpha'if label=='ridge'else'selected_ce_step']
  for c in winner['cells']:
   r=index['calibration',c['distractors'],c['delay'],label];assert c['correct']==r['counts']['value']['correct']and c['total']==r['counts']['value']['total']
 assert m['fit_rows']==cfg['data']['train']['size']*len(cfg['delays']);assert sum(m['label_counts'])==cfg['data']['train']['size']
 if cfg['mode']!='profile':assert min(m['label_counts'])>0
 return dict(cpu_audit_wall_seconds=time.monotonic()-started,source_ref=ref,rows_verified=len(rows),selected_alpha=m['selected_alpha'],selected_ce_step=m['selected_ce_step'],mode=cfg['mode'],scope='Archived metric and selection reconstruction; feature/checkpoint byte replay is separate.',validation=[dict(arm=a,minimum_correct=min(r['counts']['value']['correct']for r in rows if r['split']=='validation'and r['head']==a),support=cfg['data']['validation']['size'])for a in ('unchanged','ridge','ce')])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=audit(a.root,a.source_ref);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
