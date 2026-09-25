"""S17 stdlib immutable source/input/cap binding; no model import."""
import argparse,hashlib,json,os
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(c,source,cap):
 if c['budget_status']!='frozen' or c['cap_seconds']!=cap:raise ValueError('frozen cap mismatch')
 for group in ('source_sha256','launch_source_sha256'):
  for name,expected in c[group].items():
   if digest(source/name)!=expected:raise ValueError('source changed: '+name)
 for name in ('selection','selection_audit','data_audit'):
  if digest(c[name])!=c[name+'_sha256']:raise ValueError('input changed: '+name)
 data=Path(c['data_dir'])
 for split,expected in c['cache_sha256'].items():
  if digest(data/(split+'.jsonl.gz'))!=expected:raise ValueError('cache changed: '+split)
 for run in c['runs']:
  for name in ('checkpoint','evaluation','manifest'):
   if digest(run[name])!=run[name+'_sha256']:raise ValueError('endpoint changed: '+name)
 if Path(c['output_dir']).exists():raise ValueError('output exists')
 return dict(selection_sha256=c['selection_sha256'],endpoints=[dict(arm=r['arm'],checkpoint_sha256=r['checkpoint_sha256'],evaluation_sha256=r['evaluation_sha256']) for r in c['runs']])
def freeze(prepared,output,source,cap):
 c=json.loads(prepared.read_text())
 if c['budget_status']!='prepared' or not 0<cap<=3600:raise ValueError('invalid preparation/cap')
 c.update(budget_status='frozen',cap_seconds=cap,prepared_sha256=digest(prepared),launch_source_sha256={n:digest(source/n) for n in ('campaign_semantics_recalibrate_freeze.py','campaign_semantics_recalibrate_launch.py')},launch_authority='Separate explicit root GPU release required')
 verify(c,source,cap)
 with output.open('x') as f:f.write(json.dumps(c,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,default=Path(__file__).parent);p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
