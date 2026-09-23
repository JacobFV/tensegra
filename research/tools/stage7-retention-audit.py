import tarfile,json,hashlib,subprocess
from collections import defaultdict
p='research/results/stage7-c/stage7-c-main-artifacts.tar.gz'
with tarfile.open(p) as a:
 def get(n):return a.extractfile(n).read()
 names=a.getnames();m=json.loads(get('c-main/manifest.json'));c=json.loads(get('c-main/config.json'));print('sources',[n for n in names if 'source' in n]);print('checkpoints',[n for n in names if n.endswith('.pt')])
 for name,h in m['source'].items():assert hashlib.sha256(get('source-main/'+name)).hexdigest()==h,name
 assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==m['config']
 assert set(c['validation_seeds']).isdisjoint(c['test_seeds'])
 assert {(r['seed'],r['mode']) for r in m['runs']}=={(seed,mode) for seed in c['seeds'] for mode in c['modes']}
 pair={}; checked=0; gates=[]
 for run in m['runs']:
  seed,mode=run['seed'],run['mode'];rows=[json.loads(l) for l in get(f'c-main/seed{seed}-{mode}.jsonl').splitlines()];evals=[r for r in rows if 'predictions' in r and r.get('mode')!='persistent']; fields=('value','type','operation','argument0','argument1','provenance')
  pair.setdefault(seed,[]).append((run['initial_hash'],run['parameters']));support=set()
  for row in [r for r in rows if 'predictions' in r]:
   t=row['targets'];pr=row['predictions'];matches={k:[x==y for x,y in zip(pr[k],t[k])] for k in fields};matches['joint']=[all(v) for v in zip(*matches.values())];task=[1-x for x in t['task']] if row['intervention']=='query_counterfactual' else t['task'];matches['task']=[x==y for x,y in zip(row['task_predictions'],task)]
   for k,v in matches.items():assert row['counts'][k]==dict(correct=sum(v),total=len(v));assert len(v)==512
   assert row['counts']['task_given_retained']==dict(correct=sum(x and y for x,y in zip(matches['task'],matches['joint'])),total=sum(matches['joint']))
   assert row['split']==('validation' if row['seed'] in c['validation_seeds'] else 'test');checked+=1
  for row in evals:support.add((row['seed'],row['distractors'],row['delay'],row['intervention']))
  expected={(s,d,delay,'none') for s in c['validation_seeds']+c['test_seeds'] for d in c['eval_distractors'] for delay in c['eval_delays']}|{(s,d,16,i) for s in c['validation_seeds']+c['test_seeds'] for d in c['eval_distractors'] for i in c['interventions']}
  assert support==expected,(len(support),len(expected))
  clean=[r for r in evals if r['seed'] in c['validation_seeds'] and r['delay']==16 and r['intervention']=='none'];assert len(clean)==len(c['validation_seeds'])*len(c['eval_distractors'])
  gate=all(r['counts'][k]['correct']/512>(.99 if k in ('type','operation') else .98) for r in clean for k in fields);assert gate==run['retention_gate'];assert not gate
  assert run['optimizer_steps']==1000 and run['examples_seen']==32000;assert run['dependent_experiments']=='blocked';gates.append((seed,mode,gate))
 for seed,values in pair.items():assert len(set(values))==1
 print('PASS C rawrows',checked,'strictgates',gates,'allcondition supports/source/config hashes/pairedinitializations match')
