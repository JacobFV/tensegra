"""Independent paired continuation metrics, exposure and heldout-data audit."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
FIELDS=('value','type','operation','argument0','argument1','provenance')
def matches(r,key):
 fields=FIELDS if key=='joint' else (key,)
 return [all(r['predictions'][f][i]==r['targets'][f][i]for f in fields)for i in range(len(r['targets']['value']))]
def counts(r):
 m={f:matches(r,f)for f in FIELDS};m['joint']=matches(r,'joint');m['nonvalue_joint']=[all(v)for v in zip(*(m[f]for f in FIELDS[1:]))];m['argument1_required']=[ok for ok,y in zip(m['argument1'],r['targets']['argument1'])if y!=6]
 return {k:dict(correct=sum(v),total=len(v))for k,v in m.items()}
def load(p):
 with (gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root):
 x=load(root/'manifest.json');cfg=x['config'];repo=Path(__file__).resolve().parents[2]
 assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==x['config_sha256']
 for name,h in x['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show','9ccadae:src/topoformer/'+name],cwd=repo)).hexdigest()==h
 assert [r['backbone_seed']for r in x['runs']]==cfg['backbone_seeds']==[10,11,12]
 totals=curvecount=0;gates=[];paired=[];training=[]
 for run in x['runs']:
  seed=run['backbone_seed'];p=root/f'{seed}-predictions.json.gz';c=root/f'{seed}-curves.json.gz';assert sha(p)==run['predictions_sha256']and sha(c)==run['curves_sha256'];rows=load(p);curves=load(c)
  ix={};expected={(arm,s,d,k)for arm in ['frozen_original','frozen_shared_readout',*cfg['schedules']]for s in ('validation','test')for d in cfg['eval_delays']for k in cfg['eval_distractors']}
  for r in rows:
   key=(r['head'],r['split'],r['target_delay'],r['distractors']);assert key not in ix;ix[key]=r;assert counts(r)==r['counts'];totals+=1
  assert set(ix)==expected
  for r in rows:
   ref=ix[('frozen_original',r['split'],0,2)];assert r['targets']==ref['targets']and r['event_sha256']==ref['event_sha256']
   if r['head']=='frozen_shared_readout':assert all(r['predictions'][f]==ix[('frozen_original',r['split'],r['target_delay'],r['distractors'])]['predictions'][f]for f in FIELDS[1:])
  curvekeys=set()
  for r in curves:
   key=(r['head'],r['update'],r['target_delay']);assert key not in curvekeys;curvekeys.add(key);assert r['split']=='validation_curve'and r['target_delay']!=32 and counts(r)==r['counts'];curvecount+=1
  assert curvekeys=={(a,u,d)for a in cfg['schedules']for u in range(0,cfg['updates']+1,cfg['curve_every'])for d in cfg['curve_delays']}
  armrecords=run['arms'];assert len(armrecords)==2 and armrecords[0]['training_batches']==armrecords[1]['training_batches']
  for a in armrecords:
   assert a['initial_state_sha256']==run['initial_state_sha256']and a['optimizer_reset']and a['width']==1024
   schedule=cfg['schedules'][a['arm']];assert max(schedule)<=16
   dc={str(d):sum(schedule[s%len(schedule)]==d for s in range(cfg['updates']))for d in set(schedule)};assert dc==a['delay_updates']
   events=[h for b in a['training_batches']for h in b['event_row_hashes']];assert len(events)==a['presentations']==cfg['updates']*cfg['batch_size']and len(set(events))==a['unique_training_events']
   assert a['recurrent_example_microsteps']==sum(int(d)*v*cfg['batch_size']for d,v in dc.items())
   assert all(not(set(events)&set(h))for h in run['event_row_hashes'].values())
   training.append(dict(seed=seed,arm=a['arm'],presentations=len(events),unique=len(set(events)),recurrent_example_microsteps=a['recurrent_example_microsteps']))
  assert not(set(run['event_row_hashes']['validation'])&set(run['event_row_hashes']['test']))
  for arm in ['frozen_original','frozen_shared_readout',*cfg['schedules']]:
   failed=[]
   for k in cfg['eval_distractors']:
    c=ix[(arm,'validation',16,k)]['counts']
    for f in FIELDS:
     v=c['argument1_required'if f=='argument1'else f]
     if c['joint']['total']<512 or not v['total']or v['correct']/v['total']<=(.99 if f in ('type','operation')else.98):failed.append([k,f])
   gates.append(dict(seed=seed,arm=arm,passed=not failed,failed=failed))
  for split in ('validation','test'):
   for delay in (0,16,32):
    for k in cfg['eval_distractors']:
     for field in ('value','joint'):
      a=matches(ix[('short_continue',split,delay,k)],field);b=matches(ix[('wide_continue',split,delay,k)],field)
      paired.append(dict(seed=seed,split=split,delay=delay,distractors=k,field=field,correct_correct=sum(v and w for v,w in zip(a,b)),correct_wrong=sum(v and not w for v,w in zip(a,b)),wrong_correct=sum(not v and w for v,w in zip(a,b)),wrong_wrong=sum(not v and not w for v,w in zip(a,b))))
 return dict(rows_verified=totals,curve_rows_verified=curvecount,complete_matrix=True,training=training,retention_gates=gates,paired_short_to_wide=paired)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=audit(a.root);a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k not in ('training','retention_gates','paired_short_to_wide')})
