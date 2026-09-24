"""Independent archived-data reconstruction; no model inference or torch import."""
import collections,gzip,hashlib,json,pathlib,sys,time
start=time.process_time();root=pathlib.Path(sys.argv[1]);out={};bind={};allrows={}
for arm in ['lightweight','recurrent']:
 base=root/f'research/results/campaign-02/e03-acquire-{arm}'
 paths={n:base/n for n in ['summary.json','curves.json','development-600.jsonl.gz']}
 for p in paths.values():bind[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 s=json.loads(paths['summary.json'].read_text());curves=json.loads(paths['curves.json'].read_text())
 with gzip.open(paths['development-600.jsonl.gz'],'rt') as f:rows=[json.loads(x) for x in f]
 allrows[arm]=rows;actions=collections.Counter();counts=collections.Counter();cost=utility=0.
 for row in rows:
  o=row['outcome'];tr=row['trace'];h=o['history'];assert len(tr)==len(h)==o['steps']
  assert all(t['action']==e['action'] and t['feedback']==e['feedback'] for t,e in zip(tr,h))
  p=tr[0]['observation']['prices'];c=o['steps']*p['action']+o['observations']*p['observation']+o['work_units']*p['work']+o['travel_distance']*p['travel']+o['modeled_compute_cost']
  assert abs(c-o['cost'])<1e-12;assert abs(o['utility']-(int(o['verified_success'])-c))<1e-12
  assert sum(t['neural_work_units'] for t in tr)==o['compute_units']
  assert sum(e['action']['kind']=='inspect' for e in h)==o['observations']
  for t, red in zip([t for t in tr if t['action']['kind']=='call'],o['reductions']):
   obs=t['observation']; entry=obs['problems'][t['action']['arguments']['problem']]; problem=entry['problem']
   if entry['primitive']=='constrained_subset':
    hs=problem['handles']; expected={h:[r['category'],r['weight'],r['price'],1] for h,r in obs['known_items'].items()}
    ok=(set(hs)=={r['handle'] for r in obs['item_inventory']} and len(hs)==len(set(hs)) and all(expected[h]==r for h,r in zip(hs,problem['items'])) and problem.get('capacity')==obs['goal']['capacity'] and problem.get('max_cost')==obs['goal']['funds'] and {frozenset(x) for x in problem.get('forbidden_pairs',[])}=={frozenset((hs.index(a),hs.index(b))) for a,b in obs['incompatible']})
   else:ok=(problem['edges']==obs['known_edges'] and problem['start']==obs['position'] and problem['goal']==obs['goal']['destination'])
   assert ok==red['correct']
  last=tr[-1]['observation'];goal=last['goal'];items=[last['known_items'][v] for v in last['selected']]
  public_valid=(sorted(x['category'] for x in items)==sorted(goal['categories']) and sum(x['weight'] for x in items)<=goal['capacity'] and sum(x['price'] for x in items)<=goal['funds'] and not any(a in last['selected'] and b in last['selected'] for a,b in last['incompatible']) and last['position']==goal['destination'])
  if o['verified_success']:assert public_valid and h[-1]['action']['kind']=='verify' and h[-1]['feedback']['verified']
  counts['verified']+=bool(o['verified_success']);counts['correct_reductions']+=sum(r['correct'] for r in o['reductions']);counts['reductions']+=len(o['reductions']);counts['truncated']+=row['truncated']
  actions.update(t['action']['kind'] for t in tr);cost+=c;utility+=o['utility']
 assert len(rows)==len({r['seed'] for r in rows})==128
 assert abs(utility/128-s['validation']['utility'])<1e-12
 assert counts['verified']/128==s['validation']['success']
 assert s['training_seed_interval_this_invocation'][1]<=min(r['seed'] for r in rows)
 assert len({r['seeds_hash'] for r in s['development']})==1
 out[arm]={'rows':len(rows),'unique_world_seeds':len({r['seed'] for r in rows}),'seed_range':[min(r['seed'] for r in rows),max(r['seed'] for r in rows)],'counts':dict(counts),'actions':dict(actions),'mean_cost':cost/128,'mean_utility':utility/128,'reported_development_success_by_update':[(r['update'],r['success']) for r in s['development']],'curve_count':len(curves),'teacher_training_success_mean':sum(r['training_success'] for r in curves)/len(curves),'parameter_count':s['parameter_count'],'width':s['workspace_width'],'training_episodes':s['unique_training_episodes'],'decision_presentations':s['decision_presentations']}
a,b=allrows.values();assert [r['seed'] for r in a]==[r['seed'] for r in b]
out['paired']={'identical_environmental_action_traces':sum([x['action'] for x in aa['trace']]==[x['action'] for x in bb['trace']] for aa,bb in zip(a,b)),'shared_unique_episodes':128,'independent_lineages_per_architecture':1}
out['source_artifact_hashes']=bind;out['cpu_core_seconds']=time.process_time()-start
out['scope']='Archived metrics/traces only; no checkpoint inference or hidden-world replay; positive completion checked from full inspected public facts.'
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='source_artifact_hashes'},indent=2))
