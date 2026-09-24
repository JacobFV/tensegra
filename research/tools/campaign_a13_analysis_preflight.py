import importlib.util,json,hashlib,tempfile,time,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.root;source=root/'research/tools/campaign_a13_analysis.py';spec=importlib.util.spec_from_file_location('a13',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tick=time.monotonic();run=root/'research/results/campaign-01/attention/a13-profile';result=m.analyze(run);checked=0
for policy in m.POLICIES:
 a=np.load(run/'results'/f'{policy}.npz');r=result['conditions'][0]['policy'][policy];tasks=a['c0_task'];trace=a['c0_diagnostic_query_mean_route_correct'];joint=[0]*4
 for i in range(len(tasks)):
  route=all(bool(trace[i,j,0]) for j in range(trace.shape[1]));task=bool(tasks[i]);joint[(0 if route else 2)+(0 if task else 1)]+=1
 assert list(r['task_route_joint_counts'].values())==joint and sum(joint)==32
 assert r['task_correct_given_queried_mean_route_correct']==joint[0]/sum(joint[:2]);assert r['query_mean_trace_equals_existing_path_metric'];assert r['mean_trace_is_repeated_across_heads']
 if policy!='oracle_common':assert r['all_node_used_routes_equal_unchanged']
 for kind,h in r['head_paths'].items():
  array=a['c0_diagnostic_query_'+kind+'_head_correct'];independent=sum(all(bool(array[i,j,k]) for j in range(array.shape[1])) for i in range(32) for k in range(8));assert h['correct_head_paths']==independent and h['head_path_denominator']==256 and h['event_denominator']==32 and h['headpath_denominator_given_meanpath_correct']==sum(joint[:2])*8
 checked+=1
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'results';p.mkdir();(p/'config.json').write_bytes((run/'results/config.json').read_bytes())
 for policy in m.POLICIES:
  original=np.load(run/'results'/f'{policy}.npz');a={k:original[k].copy() for k in original.files}
  for key in ['c0_diagnostic_query_mean_route_correct','c0_exact_pointer_path','c0_diagnostic_payload_argmax_energy','c0_diagnostic_correct_destination_count']:a[key][...]=0
  np.savez(p/f'{policy}.npz',**a)
 z=m.analyze(Path(tmp))
 for r in z['conditions'][0]['policy'].values():
  assert r['task_correct_given_queried_mean_route_correct'] is None and r['suffix_correct_given_queried_mean_route_correct'] is None
  assert r['all_node_local_payload_metrics']['relative_mse'] is None and r['all_node_local_payload_metrics']['conditional_mse_given_correct_used_destination'] is None
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),profile_cells_checked=checked,independent_checks=['actual task-route joint counts and conditional intersections','32 event and256 head-path denominators','full-step head conjunctions by explicit loop','unchanged/shared policy exact all-node route equality','zero route, energy and correct-destination support produce null'],input_sha256={str((run/'results'/k).relative_to(root)):v for k,v in result['input_sha256'].items()},cpu_audit_wall_seconds=time.monotonic()-tick,scope='Profile and synthetic array fixtures only; no policy selection, independence products or main outcomes. Oracle arm privileged, schema and local distortion metrics separate.')
args.output.write_text(json.dumps(out,indent=2)+'\n');print(out['cpu_audit_wall_seconds'])
