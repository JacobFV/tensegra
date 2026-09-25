"""S20 stdlib guards: fixed nine-arm study, final-only confirmation, no actor calls."""
import argparse,gzip,hashlib,json,os
from pathlib import Path
from .campaign_semantics_s18_freeze import SOURCES as S18_SOURCES
from .campaign_semantics_s19_freeze import SOURCES as S19_SOURCES
SOURCES=tuple(sorted(set(S18_SOURCES+S19_SOURCES+('campaign_semantics_s20.py','campaign_semantics_s20_freeze.py','campaign_semantics_s20_launch.py'))))
INPUTS={'train':'8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2','selection':'5b8091e73320c5ef87bee39189933e6f3aa856c98ac07d93c6cb90d8d1c13172','vocabulary_audit':'2582514c2a31e51dc32071fcd4177d0e50d8bf710dfa48d73c66a50b81b3d1d1','development':'25522313035e646cfca4bf0efb1ae550c295cb55ddf6057558d03c42f2239dc3','confirmation':'2710e5e1b6fffab6fa33d56105de13e49d46d64c63d81f61c6ea331347229d32'}
PARENTS={701:('f419fe818626fc23b98f8e0bb7a89491ff5e015e529b5facb68ee5281d2e808a','bfd24b660c3a4a28577fc1d6cc7bcbdcf3f6ba138907d2660298b89e78c6112a','0ce7b767ec460dca9ed9004c2379483603a70919e0a18651af3ff74782db7785'),702:('64b3b035c386982d3f9563ff7e24bbd459fba8432cadaa60ffc89bf4c5d8261f','c3501de19812fa454ed2015df65728fe442b7268dea52d1ec2b1e5a3a0eaf3b2','d630372e839ece322d967ea06f1cc3c0404966906e95ad5646160d6ab2f5b0c1'),703:('be5c324a24d5a0741b4c66159e46a92f32b8d1691c51df887a17402ccb3267a2','9917ac55ece4df83970bc46474bd62938e3e549fe8322d2d8cd782c41a3b6416','9e7dfa80e845ccda88d7cba737295a58ce1b1fdb841535937f578e7e09829c2c')}
STREAM='85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9'
PAIRS='e2fa3128d6d68928310e2b49d3d29d31274eca131c8607ffb9c6483df6a846ed'
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(c):
 if c['job'] not in ('profile','main'):raise ValueError('unregistered S20 job')
 profile=c['job']=='profile'
 fixed=dict(arms=['original','context','record'],seeds=[701] if profile else [701,702,703],updates=20 if profile else 4096,evaluation_per_cell=16 if profile else 512,worst_case_profile=profile,width=1024,capacity=128,batch_size=8,schedule_seed=15115,device='cuda',autocast_dtype='bfloat16',baseline_learning_rate=1e-5,negative_pairs=128,baseline_presentation_seed_base=150150000,calibration_count=128,record_evaluation_batch_size=32,record_learning_rate=3e-4,record_final_learning_rate=3e-5,record_warmup_updates=128,record_max_records=160,record_max_slot=32,final_only=True,record_initialization='scratch_same_numeric_lineage_seed',baseline_policy='same_S17_mixed128_global_primary_raw_retained',construction_sequence_sha256=STREAM,pair_sequence_sha256=PAIRS)
 for k,v in fixed.items():
  if c[k]!=v:raise ValueError('fixed S20 recipe changed: '+k)
 expected={'train','selection','vocabulary_audit','development' if profile else 'confirmation'}
 if set(c['inputs'])!=expected:raise ValueError('profile/main input boundary changed')
 for k,r in c['inputs'].items():
  if r['sha256']!=INPUTS[k]:raise ValueError('input identity changed: '+k)
 if set(c['parents'])!={'701','702','703'}:raise ValueError('all three parents must remain bound')
 for seed,(checkpoint,manifest,state) in PARENTS.items():
  p=c['parents'][str(seed)]
  if (p['checkpoint']['sha256'],p['manifest']['sha256'],p['model_state_sha256'])!=(checkpoint,manifest,state):raise ValueError('parent lineage changed')
def check(r):
 if digest(r['path'])!=r['sha256']:raise ValueError('immutable input changed: '+r['path'])
def verify(c,source,cap):
 validate(c)
 if c['budget_status']!='frozen' or c['cap_seconds']!=cap or not isinstance(cap,(int,float)) or not 0<cap<=5000:raise ValueError('frozen positive cap required')
 if set(c['source_sha256'])!=set(SOURCES):raise ValueError('source inventory changed')
 for n,h in c['source_sha256'].items():
  if digest(Path(source)/n)!=h:raise ValueError('source changed: '+n)
 # Hash-only access to input bytes; no confirmation parsing or fitting here.
 for r in c['inputs'].values():check(r)
 for seed in c['seeds']:
  p=c['parents'][str(seed)];check(p['checkpoint']);check(p['manifest'])
  m=json.load(gzip.open(p['manifest']['path'],'rt'))
  if m['config']['seed']!=seed or m['final_state_sha256']!=p['model_state_sha256']:raise ValueError('parent manifest lineage mismatch')
 if Path(c['output_dir']).exists():raise ValueError('output prefix already used')
 return {'job':c['job'],'seeds':c['seeds'],'inputs':{k:r['sha256'] for k,r in c['inputs'].items()}}
def freeze(prepared,output,source,cap):
 c=json.loads(Path(prepared).read_text());validate(c)
 if c['budget_status']!='prepared':raise ValueError('fresh prepared config required')
 c.update(budget_status='frozen',cap_seconds=cap,prepared_sha256=digest(prepared),source_sha256={n:digest(Path(source)/n) for n in SOURCES})
 verify(c,source,cap)
 with Path(output).open('x') as f:json.dump(c,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output');p.add_argument('--source',default='src/topoformer');p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
