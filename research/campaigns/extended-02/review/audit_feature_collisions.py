"""Independent compact-record/source replay and semantic witness checks."""
import json,pathlib,sys,time,resource,hashlib,subprocess,tempfile,importlib.util
start=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN);p=pathlib.Path(sys.argv[1]);raw=json.loads(p.read_text());checks={}
with tempfile.TemporaryDirectory() as d:
 source=subprocess.check_output(['git','show',raw['source']+':src/topoformer/campaign02_world.py'])
 assert hashlib.sha256(source).hexdigest()==raw['source_hashes']['campaign02_world.py']
 f=pathlib.Path(d)/'world.py';f.write_bytes(source);spec=importlib.util.spec_from_file_location('collision_audit_world',f);w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
 for name,pair in raw['pairs'].items():
  left,right=pair['left'],pair['right'];assert left['encoded']==right['encoded'];assert left['actions']==right['actions'];assert left['teacher']['kind']!=right['teacher']['kind']
  for side in [left,right]:
   o=w.Observation(**side['observation']);assert w.encode_observation(o)==side['encoded']['observation'];assert [w.encode_action(o,a) for a in w.action_catalog(o)]==side['encoded']['candidates']
  checks[name]={'exact_full_feature_collision':True,'matching_action_catalog':True,'teacher_kinds':[left['teacher']['kind'],right['teacher']['kind']]}
 a,b=(raw['pairs']['nonpending_incompatibility'][k]['observation'] for k in ['left','right'])
 assert a['incompatible']==[] and b['incompatible']==[['a','c']]
 assert a['known_items']==b['known_items'] and not a['pending'] and not b['pending']
 feasible=[]
 for obs in [a,b]:
  hs=list(obs['known_items']);options=[]
  import itertools
  for chosen in itertools.combinations(hs,2):
   rs=[obs['known_items'][h] for h in chosen]
   if sorted(r['category'] for r in rs)==[0,1] and sum(r['weight'] for r in rs)<=4 and sum(r['price'] for r in rs)<=4 and not any(x in chosen and y in chosen for x,y in obs['incompatible']):options.append(chosen)
  feasible.append(options)
 assert feasible[0]==[('a','c'),('b','c')] and feasible[1]==[('b','c')]
 checks['nonpending_incompatibility']['independent_feasible_subsets']=feasible
 routes=[]
 for side in ['left','right']:
  o=raw['pairs']['downstream_route_weight'][side]['observation'];e={(u,v):d for u,v,d in o['known_edges']};routes.append({'via1':e[0,1]+e[1,2],'direct':e[0,2]})
 assert routes==[{'via1':2,'direct':3},{'via1':5,'direct':3}];checks['downstream_route_weight']['independent_route_lengths']=routes
end=resource.getrusage(resource.RUSAGE_CHILDREN);out={'status':'PASS_CONSTRUCTIVE_COLLISIONS','artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':raw['source'],'checks':checks,'cpu_core_seconds':time.process_time()-start+end.ru_utime+end.ru_stime-before.ru_utime-before.ru_stime,'scope':'No model inference; independent semantic witnesses and frozen encoder replay; no prevalence claim.'}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
