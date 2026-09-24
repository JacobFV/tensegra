"""S15 immutable CPU prepared→frozen binding and launch guards; no model imports."""
import argparse,hashlib,json,os
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(c,source,cap=None):
    if c['budget_status']!='frozen' or c['job'] not in ('profile','main'):raise ValueError('not a frozen S15 job')
    if cap is not None and cap!=c['cap_seconds']:raise ValueError('cap differs from frozen allocation')
    for group in ('source_sha256','launch_source_sha256'):
        for name,sha in c[group].items():
            if digest(source/name)!=sha:raise ValueError('source changed: '+name)
    data=Path(c['data_dir'])
    if digest(data/'audit.json')!=c['audit_sha256']:raise ValueError('cache audit changed')
    for split,sha in c['cache_sha256'].items():
        if digest(data/(split+'.jsonl.gz'))!=sha:raise ValueError('cache changed: '+split)
    for name in ('parent_checkpoint','parent_evaluation','calibration_cache','data_audit'):
        if digest(c[name])!=c[name+'_sha256']:raise ValueError('binding changed: '+name)
    if Path(c['output_dir']).exists():raise ValueError('output already exists')
    return dict(parent_checkpoint_sha256=c['parent_checkpoint_sha256'],audit_sha256=c['audit_sha256'],cache_sha256=c['cache_sha256'])

def freeze(prepared,output,source,cap):
    c=json.loads(prepared.read_text())
    if c['budget_status']!='prepared' or not 0<cap<=3600:raise ValueError('invalid preparation/allocation')
    for name,sha in c['source_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('prepared source changed')
    c.update(budget_status='frozen',prepared_sha256=digest(prepared),cap_seconds=cap,launch_source_sha256={n:digest(source/n) for n in ('campaign_semantics_shape_freeze.py','campaign_semantics_shape_launch.py')},launch_authority='Requires separate explicit root GPU release; freeze is not launch permission')
    verify(c,source,cap)
    with output.open('x') as f:f.write(json.dumps(c,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    return c
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prepared',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,default=Path(__file__).parent);p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
