import pathlib,json,gzip,collections,time,hashlib,sys
start=time.process_time();root=pathlib.Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text());out={'results':[],'inputs':{}};cached={}
for item in summary['results']:
 p=root/item['artifact'];sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==item['artifact_sha256'];out['inputs'][item['artifact']]=sha
 with gzip.open(p,'rt') as f:rows=[json.loads(x) for x in f]
 n=len(rows);assert n==item['examples']==512 and len({r['seed'] for r in rows})==n
 totals=collections.Counter();ends=collections.Counter();patterns=collections.Counter();failactions=collections.Counter();cost=utility=0.;success=0;truncated=0;examples=[]
 for r in rows:
  o=r['outcome'];success+=o['verified_success'];cost+=o['cost'];utility+=o['utility'];truncated+=r['truncated'];totals.update(r['counts'])
  assert abs(o['utility']-(int(o['verified_success'])-o['cost']))<1e-10
  h=o['history'];assert len(h)==o['steps'];assert sum(e['action']['kind']=='inspect' for e in h)==o['observations']
  if not o['verified_success']:
   key=(h[-1]['action']['kind'],h[-1]['feedback'].get('status'),h[-1]['feedback'].get('reason'));ends[str(key)]+=1
   patterns[' -> '.join(e['action']['kind'] for e in h[-6:])]+=1
   failactions.update(e['action']['kind'] for e in h)
   if len(examples)<2:examples.append({'seed':r['seed'],'last_events':h[-6:],'steps':o['steps'],'work':o['work_units'],'reductions':o['reductions']})
 assert success==item['success_count'];assert dict(totals)==item['counts'];assert abs(cost/n-item['means']['cost'])<1e-10;assert abs(utility/n-item['means']['utility'])<1e-10;assert truncated==item['truncated']
 out['results'].append({'condition':item['condition'],'arm':item['arm'],'success_count':success,'mean_cost':cost/n,'mean_utility':utility/n,'truncated':truncated,'counts':dict(totals),'failure_endings':dict(ends),'failure_last_six':dict(patterns.most_common(8)),'failure_actions':dict(failactions),'failure_examples':examples})
 cached[item['condition'],item['arm']]=rows
for pair in summary['paired']:
 a=cached[pair['condition'],pair['learned']];b=cached[pair['condition'],pair['comparator']]
 assert [r['seed'] for r in a]==[r['seed'] for r in b];assert [r['spec_hash'] for r in a]==[r['spec_hash'] for r in b]
 counts=collections.Counter(f"{int(x['outcome']['verified_success'])}->{int(y['outcome']['verified_success'])}" for x,y in zip(b,a));assert dict(counts)==pair['success_transitions_comparator_to_learned']
 assert abs(sum(x['outcome']['utility']-y['outcome']['utility'] for x,y in zip(a,b))/len(a)-pair['mean_utility_difference'])<1e-10
out['paired_comparisons']=len(summary['paired']);out['rows']=sum(len(v) for v in cached.values());out['cpu_core_seconds']=time.process_time()-start;out['scope']='Independent archived metric reconstruction; no model inference. Producer action-count fields summed but nontrivial address metric not independently reimplemented.'
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n')
for r in out['results']:print(r['condition'],r['arm'],r['success_count'],round(r['mean_utility'],4),r['failure_endings'],r['failure_last_six'])
print('cpu',out['cpu_core_seconds'])
