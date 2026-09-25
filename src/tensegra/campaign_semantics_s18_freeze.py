"""S18 source/recipe/baseline guards; stdlib-only and no launch authority."""
import argparse,gzip,hashlib,json
from pathlib import Path
SOURCES=('campaign_semantics_s18.py','campaign_semantics_s18_freeze.py','campaign_semantics_s18_launch.py',
 'campaign_semantics_s18_actor.py','campaign_semantics_s18_compute.py','campaign_semantics_grounded_actor.py',
 'campaign_semantics_shape_train.py','campaign_semantics.py','campaign_semantics_data.py','campaign_semantics_continue.py',
 'campaign_semantics_lr.py','semantic_curriculum.py','semantic_text_acquisition.py','semantic_scaling.py',
 'semantic_contracts.py','thinking.py','thinking_language.py','semantic_graph.py','tcn_data.py')
PARENT='3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e'
MIXED='8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2'
SELECTION='5b8091e73320c5ef87bee39189933e6f3aa856c98ac07d93c6cb90d8d1c13172'
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(c):
 if c['job'] not in ('profile','main','reference'):raise ValueError('unregistered job')
 expected={'profile':(20,[0,20],16),'main':(4096,[0,1024,2048,4096],512),'reference':(0,[0,1024,2048],512)}[c['job']]
 if (c['added_updates'],c['checkpoints'],c['dev_per_cell'])!=expected:raise ValueError('fixed exposure/evaluation changed')
 if c['arms']!=(['original'] if c['job']=='reference' else ['context','workspace_control']) or c['control_microsteps']!=10:raise ValueError('fixed arms changed')
 if (c['width'],c['capacity'],c['batch_size'],c['learning_rate'],c['schedule_seed'],c['negative_pairs'])!=(1024,128,8,1e-5,15115,128):raise ValueError('fixed recipe changed')
 if c['device']!='cuda' or c['calibration_count']!=128 or c['baseline_policy']!='reuse_S15_mixed_S17_endpoint_plus_frozen_early_curves':raise ValueError('registered reference policy changed')
 if c['calibration_policy']!='same_S17_mixed128_matched_primary_historical_secondary':raise ValueError('policy changed')
def check(record):
 if digest(record['path'])!=record['sha256']:raise ValueError('immutable reference changed: '+record['path'])
def references(b,source):
 if b['parent']['sha256']!=PARENT or b['caches']['train_mixed']['sha256']!=MIXED or b['selection']['sha256']!=SELECTION:raise ValueError('fixed parent/data/selection changed')
 for k in ('original_manifest','s17_manifest','matched_endpoint','matched_endpoint_calibration','selection','selection_audit','parent','data_audit','historical_train','shape_audit'):check(b[k])
 for r in b['caches'].values():check(r)
 if [(r['added_update'],r['update']) for r in b['checkpoints']]!=[(0,24576),(1024,25600),(2048,26624),(4096,28672)]:raise ValueError('baseline curves changed')
 for r in b['checkpoints']:
  check(r['checkpoint']);check(r['historical_evaluation'])
 with gzip.open(b['original_manifest']['path']) as f:m=json.load(f)
 if m['config']!=b['original_config']:raise ValueError('original config mismatch')
 c=m['config']
 if (c['arm'],c['added_updates'],c['learning_rate'],c['batch_size'],c['negative_pairs'],c['schedule_seed'])!=('mixed',4096,1e-5,8,128,15115):raise ValueError('baseline science mismatch')
 if c['parent_checkpoint_sha256']!=PARENT or c['cache_sha256']['train_mixed']!=MIXED:raise ValueError('baseline parent/cache mismatch')
 if set(m['visits'])!={8} or len(m['visits'])!=4096 or m['added_presentations']!=32768:raise ValueError('baseline exposure mismatch')
 for n,h in c['source_sha256'].items():
  if digest(Path(source)/n)!=h:raise ValueError('inherited training source changed: '+n)
 for k,v in b['expected_streams'].items():
  if m[k]!=v:raise ValueError('baseline stream mismatch')
 for curve,r in zip(m['curves'],b['checkpoints']):
  if curve['checkpoint_sha256']!=r['checkpoint']['sha256'] or curve['evaluation']['sha256']!=r['historical_evaluation']['sha256']:raise ValueError('curve binding mismatch')
 with gzip.open(b['s17_manifest']['path']) as f:s=json.load(f)
 entry=next(r for r in s['artifacts'] if r['arm']=='mixed')
 if entry['artifact_sha256']!=b['matched_endpoint']['sha256'] or entry['checkpoint_sha256']!=b['checkpoints'][-1]['checkpoint']['sha256'] or not entry['raw_target_replay_exact']:raise ValueError('S17 reference mismatch')
 if entry['model_state_sha256']!=m['final_state_sha256']:raise ValueError('S17/S15 state mismatch')
 return b

def verify(c,source,cap=None):
 validate(c)
 if c['budget_status']!='frozen' or not isinstance(c['proposed_cap_seconds'],(float,int)) or not 0<c['proposed_cap_seconds']<=5400:raise ValueError('positive prospective cap required')
 if cap is not None and cap!=c['proposed_cap_seconds']:raise ValueError('cap mismatch')
 if set(c['source_sha256'])!=set(SOURCES):raise ValueError('incomplete source freeze')
 for n,h in c['source_sha256'].items():
  if digest(Path(source)/n)!=h:raise ValueError('source changed: '+n)
 check(c['bindings']);return references(json.loads(Path(c['bindings']['path']).read_text()),source)
def freeze(prepared,output,source,cap):
 c=json.loads(Path(prepared).read_text());validate(c)
 if c['budget_status']!='prepared' or not 0<cap<=5400:raise ValueError('fresh prepared recipe/cap needed')
 c.update(budget_status='frozen',proposed_cap_seconds=cap,prepared_sha256=digest(prepared),source_sha256={n:digest(Path(source)/n) for n in SOURCES})
 with Path(output).open('x') as f:json.dump(c,f,indent=2);f.write('\n')
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output');p.add_argument('--cap',type=float,required=True);p.add_argument('--source',default='src/topoformer');a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
