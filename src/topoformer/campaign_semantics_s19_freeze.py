"""S19 stdlib immutable scientific/config/input guards. No model imports."""
import argparse,hashlib,json,os
from pathlib import Path
SOURCES=('campaign_semantics_s19.py','campaign_semantics_s19_actor.py','campaign_semantics_s19_codec.py','campaign_semantics_s19_freeze.py','campaign_semantics_s19_launch.py','campaign_semantics.py','campaign_semantics_data.py','campaign_semantics_continue.py','semantic_curriculum.py','semantic_scaling.py','semantic_text_acquisition.py','semantic_contracts.py','thinking_language.py','thinking.py','semantic_graph.py','tcn_data.py')
INPUT_HASHES={'train':'8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2','development':'25522313035e646cfca4bf0efb1ae550c295cb55ddf6057558d03c42f2239dc3','selection':'5b8091e73320c5ef87bee39189933e6f3aa856c98ac07d93c6cb90d8d1c13172','vocabulary_audit':'2582514c2a31e51dc32071fcd4177d0e50d8bf710dfa48d73c66a50b81b3d1d1'}
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(c):
 if c['job'] not in ('profile','main'):raise ValueError('unregistered S19 job')
 expected=(20,[0,20],16) if c['job']=='profile' else (4096,[0,1024,2048,4096],512)
 if (c['updates'],c['checkpoints'],c['dev_per_cell'])!=expected:raise ValueError('fixed exposure or evaluation changed')
 constants={'seed':1901,'schedule_seed':15115,'width':1024,'encoder_layers':2,'decoder_layers':2,'heads':8,'ffn_width':4096,'dropout':0,'capacity':128,'max_records':160,'max_slot':32,'batch_size':8,'evaluation_batch_size':32,'train_panel_count':128,'learning_rate':3e-4,'final_learning_rate':3e-5,'warmup_updates':128,'weight_decay':.01,'gradient_clip':1.,'loss_policy':'equal_mean_of_eight_applicable_field_means_absent_zero','autocast_dtype':'bfloat16','device':'cuda','expected_construction_sequence_sha256':'85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9'}
 for k,v in constants.items():
  if c[k]!=v:raise ValueError('frozen recipe changed: '+k)
 if c['optimizer_betas']!=[.9,.999] or c['optimizer_eps']!=1e-8:raise ValueError('AdamW changed')
 if c['worst_case_profile']!=(c['job']=='profile'):raise ValueError('stress only belongs in mechanical profile')
 if set(c['inputs'])!=set(INPUT_HASHES):raise ValueError('input inventory changed')
 for k,h in INPUT_HASHES.items():
  if c['inputs'][k]['sha256']!=h:raise ValueError('input binding changed: '+k)
 if c['initialization']!='scratch_no_inherited_weights' or c['calibration']!='none':raise ValueError('history or decoding changed')
def verify(c,source,cap):
 validate(c)
 if c['budget_status']!='frozen' or c['cap_seconds']!=cap or not isinstance(cap,(int,float)) or not 0<cap<=3600:raise ValueError('separate positive cap freeze required')
 if set(c['source_sha256'])!=set(SOURCES):raise ValueError('incomplete source binding')
 for name,h in c['source_sha256'].items():
  if digest(Path(source)/name)!=h:raise ValueError('source changed: '+name)
 for k,r in c['inputs'].items():
  if digest(r['path'])!=r['sha256']:raise ValueError('input changed: '+k)
 if Path(c['output_dir']).exists():raise ValueError('output prefix already used')
 return {'inputs':{k:r['sha256'] for k,r in c['inputs'].items()},'job':c['job']}
def freeze(prepared,output,source,cap):
 c=json.loads(Path(prepared).read_text());validate(c)
 if c['budget_status']!='prepared' or not 0<cap<=3600:raise ValueError('invalid fresh freeze')
 c.update(budget_status='frozen',cap_seconds=cap,prepared_sha256=digest(prepared),source_sha256={n:digest(Path(source)/n) for n in SOURCES},launch_authority='Separate explicit root GPU release required')
 verify(c,source,cap)
 with Path(output).open('x') as f:json.dump(c,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('output');p.add_argument('--source',default='src/topoformer');p.add_argument('--cap',type=float,required=True);a=p.parse_args();freeze(a.prepared,a.output,a.source,a.cap)
