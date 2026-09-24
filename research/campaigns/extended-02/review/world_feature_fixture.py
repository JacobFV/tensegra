"""Public-action fixture: mutable draft provenance collapses unequal returns."""
import importlib.util,sys,itertools,time,json,pathlib,hashlib
start=time.process_time();p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('review_world',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
def executor(kind,problem,budget):
 rows=problem['items'];groups=[[i for i,r in enumerate(rows) if r[0]==c] for c in sorted({r[0] for r in rows})]
 for ids in itertools.product(*groups):
  weight=sum(rows[i][1] for i in ids);cost=sum(rows[i][2] for i in ids)
  if weight<=problem.get('capacity',100) and cost<=problem.get('max_cost',100):return dict(status='success',payload=(ids,len(ids),weight,cost),work_units=1)
 return dict(status='infeasible',payload=None,work_units=1)
spec=m.WorldSpec((m.Item('a',0,5,1),m.Item('b',0,1,1),m.Item('c',1,5,1),m.Item('d',1,1,1)),(0,1),2,3,(),((0,1,1),(1,2,1)),0,2)
w=m.Workshop(spec,executor)
for item in spec.items:w.step(m.Action('inspect',{'target':item.handle}))
w.step(m.Action('start_subset',{'handle':'p'}));o=w.step(m.Action('call',{'problem':'p','budget':16}));old=o.feedback['return']
w.step(m.Action('add_constraint',{'problem':'p','constraint':'capacity'}));o=w.step(m.Action('call',{'problem':'p','budget':16}));new=o.feedback['return']
for h in [old,new]:o=w.step(m.Action('retrieve',{'handle':h}))
features=[m.encode_action(o,m.Action('use_return',{'handle':h,'as':'subset'})) for h in [old,new]]
report={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'old_payload':o.retrieved[old]['payload'],'new_payload':o.retrieved[new]['payload'],'different_payloads':o.retrieved[old]['payload']!=o.retrieved[new]['payload'],'identical_encoded_actions':features[0]==features[1],'public_current_draft':o.problems['p'],'cpu_core_seconds':time.process_time()-start,'scope':'Mechanical public-action fixture, supplied small independent enumeration; not a learned-policy run.'}
pathlib.Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
