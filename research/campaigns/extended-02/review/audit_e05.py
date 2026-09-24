import pathlib,json,gzip,collections,hashlib,time,sys
start=time.process_time();root=pathlib.Path(sys.argv[1]);out={'arms':{},'inputs':{}};final={}
for arm in ['lightweight','recurrent']:
 base=root/f'e05-acquire-{arm}';state=json.loads((base/'state.json').read_text());protocol=json.loads((base/'protocol.json').read_text());assert protocol['mode']=='single' and protocol['rounds']==1 and state['status']=='completed'
 curves=[];support=[];denom=collections.Counter();failtypes=collections.Counter();examples={}
 for allocation in state['allocations']:
  p=base/allocation['development_predictions'];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==allocation['development_sha256'];out['inputs'][str(p.relative_to(root))]=hashlib.sha256(b).hexdigest()
  with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
  assert len(rows)==128 and len({r['seed'] for r in rows})==128
  support.append([r['seed'] for r in rows]);success=sum(r['outcome']['verified_success'] for r in rows);utility=sum(r['outcome']['utility'] for r in rows)/128;cost=sum(r['outcome']['cost'] for r in rows)/128
  assert success/128==allocation['success'] and abs(utility-allocation['utility'])<1e-12 and abs(cost-allocation['cost'])<1e-12
  for r in rows:
   o=r['outcome'];assert abs(o['utility']-(float(o['verified_success'])-o['cost']))<1e-10;assert len(o['history'])==o['steps']
  curves.append({'update':allocation['cumulative_slot_updates'],'success_count':success,'utility':utility,'cost':cost})
  if allocation['slot']==5:
   final[arm]=[(r['seed'],r['outcome']['verified_success']) for r in rows]
   for r in rows:
    o=r['outcome'];h=o['history'];denom['incorrect_reduction_episodes']+=any(not x['correct'] for x in o['reductions']);denom['no_reductions']+=not o['reductions'];denom['truncated']+=r['truncated']
    if not o['verified_success']:
     key=str((h[-1]['action']['kind'],h[-1]['feedback'].get('status'),h[-1]['feedback'].get('reason')));failtypes[key]+=1
     if key not in examples:examples[key]={'seed':r['seed'],'last_six':[{'kind':e['action']['kind'],'feedback':e['feedback'].get('status'),'reason':e['feedback'].get('reason')} for e in h[-6:]],'steps':o['steps'],'remaining_work':r['trace'][-1]['remaining_work']}
 assert all(x==support[0] for x in support)
 intervals=[a['training_seed_interval'] for a in state['allocations']];assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]));assert intervals[-1][1]-intervals[0][0]==4800
 assert state['finalist']['slot']==5
 for name in ['state.json','protocol.json']:out['inputs'][f'e05-acquire-{arm}/{name}']=hashlib.sha256((base/name).read_bytes()).hexdigest()
 out['arms'][arm]={'curves':curves,'unique_development_support':128,'training_episodes':4800,'decision_presentations':state['allocations'][-1]['cumulative_decision_presentations'],'final_failure_endings':dict(failtypes),'episode_diagnostics':dict(denom),'failure_examples':examples,'mode':'single learner, six sequential100-update slots; not evolution'}
assert [s for s,v in final['lightweight']]==[s for s,v in final['recurrent']]
out['paired_final_recurrent_to_lightweight']=dict(collections.Counter(f'{int(b)}->{int(a)}' for (_,a),(_,b) in zip(final['lightweight'],final['recurrent'])))
out['cpu_core_seconds']=time.process_time()-start;out['scope']='Archived outcome reconstruction, not inference; does not attribute observed failures to constructive encoder collisions.'
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,separators=(',',':'))+'\n')
for arm,d in out['arms'].items():print(arm,d['curves'],d['final_failure_endings'],d['episode_diagnostics'])
print(out['paired_final_recurrent_to_lightweight'],'CPU',out['cpu_core_seconds'])
