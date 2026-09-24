"""Bounded independent accounting fixtures. Run against an explicit frozen file."""
import importlib.util,json,os,pathlib,resource,signal,subprocess,sys,tempfile,time,hashlib
source=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
start=time.process_time(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
spec=importlib.util.spec_from_file_location('reviewed_job',source); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records={}
with tempfile.TemporaryDirectory(prefix='e02-accounting-review-') as root:
 root=pathlib.Path(root)
 child="import time; s=time.process_time();\nwhile time.process_time()-s<0.25: pass"
 parent="import subprocess,sys,time; subprocess.run([sys.executable,'-c',"+repr(child)+"],check=True); s=time.process_time();\nwhile time.process_time()-s<0.15: pass"
 module.run([sys.executable,'-c',parent],root/'normal',5,4)
 records['reaped_child']=json.loads((root/'normal/occupancy.json').read_text())
 module.run([sys.executable,'-c','import time;time.sleep(20)'],root/'wall',.3,4)
 records['wall']=json.loads((root/'wall/occupancy.json').read_text())
 for kind in ['leader_exits','child_ignores_term']:
  pidfile=root/(kind+'.pid')
  orphan="import os,pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(20)"
  parent="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',"+repr(orphan)+"]);\nwhile not pathlib.Path("+repr(str(pidfile))+").exists(): time.sleep(.01)\n"+('time.sleep(20)' if kind=='child_ignores_term' else '')
  module.run([sys.executable,'-c',parent],root/kind,.4,4)
  pid=int(pidfile.read_text()); alive=pathlib.Path('/proc')/str(pid)
  records[kind]={'receipt':json.loads((root/kind/'occupancy.json').read_text()),'child_alive_after_wrapper':alive.exists()}
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 records['audit_cpu_core_seconds']=time.process_time()-start+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
 records['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
 records['assertions']={'normal_includes_child_cpu':records['reaped_child']['cpu_core_seconds']>=.4,'wall_cap_reason':records['wall']['stop_reason']=='wall_cap'}
 output.write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps(records,indent=2))
