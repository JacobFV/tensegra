"""S22 immutable no-training diagnostic guards; no model or target parsing."""
import argparse,gzip,hashlib,json,os
from pathlib import Path
from .campaign_semantics_s19_freeze import SOURCES as BASE_SOURCES
SOURCES=tuple(sorted(set(BASE_SOURCES+('campaign_semantics_s22.py','campaign_semantics_s22_controller.py','campaign_semantics_s22_freeze.py','campaign_semantics_s22_launch.py'))))
CELLS=((2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4))
POLICIES=('oracle_node_count','oracle_node_kinds','oracle_node_prefix')
INPUTS={'development':'07df44cdadcb9a697b723b9292dd9b1234f7311680e20074b5aea624a7648b29','vocabulary_audit':'2582514c2a31e51dc32071fcd4177d0e50d8bf710dfa48d73c66a50b81b3d1d1'}
ENDPOINTS={'original':('b1b9445f21f7d6e3856e13e6edad0aa1265114098d8c622b744666ed6efe609b','3d81794fa49cae0f41e09c3e799f7f5085e54e4b7594d07a891c220b97e53cfc','2446f1bb72e537510fd9132407d3233dcd40204bce0288b49b3f0a8167868afe','a74aa350c6790ad12fa431f51ae6fbdfc1a253712875956beb5ed1963945e2f3'),'broad':('a5ad0f6d4fba9804a0ae3ed1ef8dd7b399ed2037dc89acd7b28c07f3bfc56af8','2d1267174e4f7fbd93ebd77da36bec0748e74f5c412cc0144b56cf64925d683e','face2f00331e1b2cddcccc2f64b8fdf6f9355bcd9f5955b861ea38a498783a8e','15777b9714c648144304e36041ed419eb11bceeffe92ef0063da9a2229573502')}
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(c):
 if c['job'] not in ('profile','main'):raise ValueError('unregistered S22 job')
 profile=c['job']=='profile';fixed=dict(arms=['original','broad'],policies=list(POLICIES),dev_per_cell=16 if profile else 512,width=1024,heads=8,capacity=128,max_records=160,max_slot=32,batch_size=32,autocast_dtype='bfloat16',device='cuda',worst_case_profile=profile,no_optimizer_or_training=True,calibration='none',privilege='explicit_named_NODE_boundary_only',checkpoint_update=4096)
 for k,v in fixed.items():
  if c[k]!=v:raise ValueError('fixed diagnostic changed: '+k)
 if set(c['inputs'])!=set(INPUTS):raise ValueError('only DEV/vocabulary inputs; no confirmation')
 for k,h in INPUTS.items():
  if c['inputs'][k]['sha256']!=h:raise ValueError('input changed: '+k)
 if set(c['checkpoints'])!=set(ENDPOINTS):raise ValueError('both frozen backbones required')
 for arm,(checkpoint,state,manifest,reference) in ENDPOINTS.items():
  r=c['checkpoints'][arm]
  if (r['checkpoint']['sha256'],r['model_state_sha256'],r['manifest']['sha256'],r['public_reference']['sha256'])!=(checkpoint,state,manifest,reference):raise ValueError('frozen endpoint changed: '+arm)
def verify(c,source,cap):
 validate(c)
 if c['budget_status']!='frozen' or c['cap_seconds']!=cap or not isinstance(cap,(float,int)) or not 0<cap<=1200:raise ValueError('positive allocated frozen cap required')
 if set(c['source_sha256'])!=set(SOURCES):raise ValueError('source inventory changed')
 for n,h in c['source_sha256'].items():
  if digest(Path(source)/n)!=h:raise ValueError('source changed: '+n)
 for r in list(c['inputs'].values())+[r[k] for r in c['checkpoints'].values() for k in ('checkpoint','manifest','public_reference')]:
  if digest(r['path'])!=r['sha256']:raise ValueError('input/endpoint hash mismatch: '+r['path'])
 for arm,bind in c['checkpoints'].items():
  m=json.load(gzip.open(bind['manifest']['path'],'rt'))
  if m['arm']!=arm or m['final_state_sha256']!=bind['model_state_sha256'] or m['curves'][-1]['update']!=4096 or m['curves'][-1]['checkpoint_sha256']!=bind['checkpoint']['sha256']:raise ValueError('endpoint provenance')
 if Path(c['output_dir']).exists():raise ValueError('immutable output already exists')
 return dict(job=c['job'],inputs={k:r['sha256'] for k,r in c['inputs'].items()})
def freeze(prepared,output,source,cap):
 c=json.loads(Path(prepared).read_text());validate(c)
 if c['budget_status']!='prepared':raise ValueError('fresh prepared config required')
 c.update(budget_status='frozen',cap_seconds=cap,prepared_sha256=digest(prepared),source_sha256={n:digest(Path(source)/n) for n in SOURCES});verify(c,source,cap)
 with Path(output).open('x') as f:json.dump(c,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output');p.add_argument('--source',default='src/topoformer');p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
