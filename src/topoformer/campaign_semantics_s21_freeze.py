"""S21 immutable preparation/launch guards. No actor or sealed-cache access."""
import argparse,hashlib,json,os
from pathlib import Path
from .campaign_semantics_s19_freeze import SOURCES as S19_SOURCES
SOURCES=tuple(sorted(set(S19_SOURCES+('campaign_semantics_s21.py','campaign_semantics_s21_freeze.py','campaign_semantics_s21_launch.py'))))
CELLS=((2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4))
DATA_BINDINGS=None # Set after independently audited S21 cache freeze, before any inference.
STREAM='85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9'
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(c):
 if c['job'] not in ('profile','main'):raise ValueError('unregistered S21 job')
 profile=c['job']=='profile'
 constants=dict(seed=2101,arms=['original','broad'],updates=20 if profile else 4096,checkpoints=[0,20] if profile else [0,1024,2048,4096],dev_per_cell=16 if profile else 512,schedule_seed=15115,width=1024,encoder_layers=2,decoder_layers=2,heads=8,ffn_width=4096,dropout=0,capacity=128,max_records=160,max_slot=32,batch_size=8,evaluation_batch_size=32,train_panel_count=128,learning_rate=3e-4,final_learning_rate=3e-5,warmup_updates=128,weight_decay=.01,gradient_clip=1.,loss_policy='equal_mean_of_eight_applicable_field_means_absent_zero',autocast_dtype='bfloat16',device='cuda',initialization='scratch_no_inherited_weights',calibration='none',optimizer_betas=[.9,.999],optimizer_eps=1e-8,worst_case_profile=profile,expected_construction_sequence_sha256=STREAM)
 for k,v in constants.items():
  if c[k]!=v:raise ValueError('fixed S21 recipe changed: '+k)
 if set(c['inputs'])!={'train_original','train_broad','development','panels','audit','vocabulary_audit'}:raise ValueError('input boundary; sealed confirmation forbidden')
 for r in c['inputs'].values():
  h=r['sha256']
  if not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h):raise ValueError('missing immutable data binding')
 if set(c['expected_epoch_exposures'])!={'original','broad'}:raise ValueError('both exposure bindings required')
 for d in c['expected_epoch_exposures'].values():
  if set(d)!={'tokens','nodes','edges','records'} or any(type(v)is not int or v<=0 for v in d.values()):raise ValueError('positive exact data-derived exposures required')
def verify(c,source,cap):
 validate(c)
 if DATA_BINDINGS is None or {k:r['sha256'] for k,r in c['inputs'].items()}!=DATA_BINDINGS:raise ValueError('independently audited S21 data pin required')
 if c['budget_status']!='frozen' or c['cap_seconds']!=cap or not isinstance(cap,(int,float)) or not 0<cap<=1800:raise ValueError('frozen positive allocated cap required')
 if set(c['source_sha256'])!=set(SOURCES):raise ValueError('source inventory changed')
 for n,h in c['source_sha256'].items():
  if digest(Path(source)/n)!=h:raise ValueError('source changed: '+n)
 for r in c['inputs'].values():
  if digest(r['path'])!=r['sha256']:raise ValueError('input changed: '+r['path'])
 if Path(c['output_dir']).exists():raise ValueError('output prefix already exists')
 return dict(inputs={k:r['sha256'] for k,r in c['inputs'].items()},job=c['job'])
def freeze(prepared,output,source,cap):
 c=json.loads(Path(prepared).read_text());validate(c)
 if c['budget_status']!='prepared':raise ValueError('fresh prepared config required')
 c.update(budget_status='frozen',cap_seconds=cap,prepared_sha256=digest(prepared),source_sha256={n:digest(Path(source)/n) for n in SOURCES})
 verify(c,source,cap)
 with Path(output).open('x') as f:json.dump(c,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output');p.add_argument('--source',default='src/topoformer');p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
