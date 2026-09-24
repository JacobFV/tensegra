"""Stdlib-only immutable freeze/launch failure checks; no real GPU or model child."""
import importlib.util,json,os,pathlib,subprocess,sys,tempfile
source=pathlib.Path('src/topoformer');spec=importlib.util.spec_from_file_location('shape_freeze',source/'campaign_semantics_shape_freeze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as temporary:
 root=pathlib.Path(temporary);data=root/'data';data.mkdir();sources=root/'source';sources.mkdir()
 for name in ('campaign_semantics_shape_freeze.py','campaign_semantics_shape_launch.py'):(sources/name).write_bytes((source/name).read_bytes())
 (sources/'actor.py').write_text('fixture');(data/'audit.json').write_text('{}');(data/'train.jsonl.gz').write_text('fixture')
 c=dict(budget_status='prepared',job='profile',source_sha256={'actor.py':m.digest(sources/'actor.py')},data_dir=str(data),audit_sha256=m.digest(data/'audit.json'),cache_sha256={'train':m.digest(data/'train.jsonl.gz')},output_dir=str(root/'unused'))
 for name in ('parent_checkpoint','parent_evaluation','calibration_cache','data_audit'):
  p=root/name;p.write_text(name);c[name]=str(p);c[name+'_sha256']=m.digest(p)
 prepared=root/'prepared.json';prepared.write_text(json.dumps(c));frozen=root/'frozen.json';f=m.freeze(prepared,frozen,sources,60);m.verify(f,sources,60)
 try:m.freeze(prepared,frozen,sources,60)
 except FileExistsError:pass
 else:raise AssertionError('overwrite accepted')
 try:m.verify(f,sources,61)
 except ValueError:pass
 else:raise AssertionError('cap mutation accepted')
 # Exercise the real wrapper's durable failure path, with a fake idle query.
 binary=root/'bin';binary.mkdir();idle=binary/'nvidia-smi';idle.write_text('#!/bin/sh\nexit 0\n');idle.chmod(0o755)
 (sources/'actor.py').write_text('changed');prefix=root/'failure'
 result=subprocess.run([sys.executable,str(sources/'campaign_semantics_shape_launch.py'),str(frozen),'--cap','60','--python','/nonexistent/child','--prefix',str(prefix)],env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']},capture_output=True,text=True)
 receipt=json.loads(pathlib.Path(str(prefix)+'.occupancy.json').read_text());assert result.returncode==1 and receipt['exit_code']==1 and not receipt['timed_out'] and 'source changed' in receipt['error'];assert pathlib.Path(str(prefix)+'.started.json').exists();assert receipt['process_occupancy_seconds']>0
 print(json.dumps(dict(immutable_freeze=True,cap_guard=True,source_guard=True,durable_failure_receipt=True,child_not_launched=True,real_gpu_query=False)))
