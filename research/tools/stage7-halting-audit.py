import json,gzip,hashlib,subprocess
from pathlib import Path
from collections import Counter
x=json.loads(gzip.decompress(Path('research/stages/stage-07/stage7-halt-main-raw-metrics.json.gz').read_bytes()));c=json.loads(x['config.json']);m=json.loads(x['manifest.json']);g=json.loads(x['gate.json']);assert hashlib.sha256(x['config.json'].encode()).hexdigest()==m['config_sha256']
for n,h in m['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show','b49d53d:src/topoformer/'+n])).hexdigest()==h
assert not any('/test-' in k and not k.endswith('-2000.json') for k in x)
checks=[];records=0
for seed in c['seeds']:
 curves=json.loads(x[f'seed-{seed}/curves.json']);final=curves[-1]['metrics']
 for step in curves:
  for split,controls in step['metrics'].items():
   for control,metrics in controls.items():
    rows=json.loads(x[f'seed-{seed}/{split}-{control}-{step["step"]}.json']);n=len(rows);assert n==512
    for r in rows:
     assert r['correct']==(r['prediction']==r['target']);assert r['exact']==(r['stop']==r['arrival']);assert r['premature']==(r['stop']<r['arrival']);assert r['updates']==r['stop'];assert r['extra']==max(0,r['stop']-r['arrival'])
    arrivals=Counter(str(r['arrival']) for r in rows);assert arrivals=={'3':128,'5':128,'7':128,'8':128}
    calc=dict(task_accuracy=sum(r['correct'] for r in rows)/n,halt_exact_accuracy=sum(r['exact'] for r in rows)/n,stop_reject_accuracy=sum(r['exact'] and r['correct'] for r in rows)/n,premature_rate=sum(r['premature'] for r in rows)/n,mean_updates=sum(r['updates'] for r in rows)/n,extra_steps=sum(r['extra'] for r in rows)/n,reject_accuracy=sum(r['correct'] and r['exact'] for r in rows if r['reject'])/128)
    for k,v in calc.items():assert abs(v-metrics[k])<1e-10
    for a,v in metrics['arrival_accuracy'].items():assert v==sum(r['exact'] and r['correct'] for r in rows if str(r['arrival'])==a)/128
    for a,dist in metrics['microsteps_by_arrival'].items():assert dist==dict(Counter(str(r['stop']) for r in rows if str(r['arrival'])==a))
    records+=n
 l=final['validation']['learned'];minimum=final['validation']['minimum'];check=dict(stop_reject=l['stop_reject_accuracy']>.95,premature=l['premature_rate']<.01,rejection=l['reject_accuracy']>.95,variable_arrivals=sum(v>.95 for v in l['arrival_accuracy'].values())>=3,task_advantage=l['task_accuracy']-minimum['task_accuracy']>.20);checks.append(check)
 print('seed',seed,'validation pure/joint',l['halt_exact_accuracy'],l['stop_reject_accuracy'],'test pure/joint',final['test']['learned']['halt_exact_accuracy'],final['test']['learned']['stop_reject_accuracy'])
assert checks==g['validation_seed_checks'];assert g['passed']==all(all(v.values()) for v in checks);assert not g['passed'];assert not g['composition_allowed']
print('PASS',records,'raw episode records exact metrics/arrival histograms/final gate/source config hashes/test isolation')
