"""Independent temporary-ledger budget/reserve fixtures; never touches live ledger."""
import argparse,importlib.util,json,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();spec=importlib.util.spec_from_file_location('queue_audit_target',a.source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);checks=[]
with tempfile.TemporaryDirectory()as d:
 root=Path(d);q=dict(running=None,ready=[dict(id='x'),'x','keep'],completed=[]);b=dict(deadline_utc='2099-01-01T00:00:00Z',gpu_seconds_ceiling=1000,confirmation_reserve_seconds=200,charged_seconds=100,jobs=[dict(kind='confirmation',seconds=50),dict(kind='exploration',seconds=50)])
 def reset():
  (root/'queue.json').write_text(json.dumps(q));(root/'budget.json').write_text(json.dumps(b))
 reset()
 try:m.record(root,'start','x',751)
 except ValueError:checks.append('remaining reserve rejects overbudget')
 else:raise AssertionError
 m.record(root,'start','x',750);assert json.load(open(root/'queue.json'))['ready']==['keep'];checks.append('structured and string ready ids removed')
 try:m.record(root,'start','y',1)
 except ValueError:checks.append('overlapping release rejected')
 else:raise AssertionError
 m.record(root,'completed','x',750);assert json.load(open(root/'budget.json'))['charged_seconds']==850;checks.append('actual occupancy recorded')
 m.record(root,'start','confirm',150,'confirmation');checks.append('confirmation may use reserve through ceiling')
 b['jobs']=[dict(kind='confirmation',seconds=250),dict(kind='exploration',seconds=50)];b['charged_seconds']=300;reset();m.record(root,'start','x',700);checks.append('spent reserve not withheld twice')
out=dict(checks=checks,cpu_audit_wall_seconds=time.monotonic()-start,scope='Temporary fixture only; actual campaign ledger untouched.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
