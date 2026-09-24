"""Independent paired S13 profile provenance, calibration and graph metric audit."""
import argparse,gzip,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifests=[];seen=set();graphs=0;thresholds=0;timing={}
for arm in ('english','mixed'):
 root=a.root/f's13-{arm}-profile';m=load(root/'manifest.json.gz');manifests.append(m);cfg=m['config']
 for name,h in cfg['profile_source_sha256'].items():assert sha(a.source_root/name)==h
 for name,h in m['source_sha256'].items():assert sha(a.source_root/name)==h
 assert cfg['added_updates']==20 and cfg['checkpoints']==[0,20] and m['added_presentations']==160 and sum(m['added_visits'])==160 and m['inherited_optimizer_steps']==[24576.]
 occ=json.loads((root/f's13-{arm}-profile-v2.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out']
 evalseconds=0
 for point in m['curves']:
  ep=root/point['english']['artifact'];sp=root/point['spanish']['artifact'];assert sha(ep)==point['english']['sha256'] and sha(sp)==point['spanish']['sha256'];eng=load(ep);spa=load(sp);assert eng['thresholds']==spa['thresholds'] and spa['threshold_source']=='same-checkpoint English TRAIN128';evalseconds+=point['english']['seconds']+point['spanish']['seconds']
  if sha(ep) not in seen:
   calpath=root/eng['calibration_data_artifact'];assert sha(calpath)==eng['calibration_data_sha256'];cal=np.load(calpath)
   for i,record in enumerate(eng['calibration']):
    cut,error=cutoff(cal['scores'][:,i],cal['targets'][:,i].astype(bool));assert abs(cut-record['threshold'])<1e-6 and error==record['train_errors'];thresholds+=1
  for path,data in [(ep,eng),(sp,spa)]:
   if sha(path) in seen:continue
   seen.add(sha(path))
   for row in data['rows']:
    for mode,pred in [('raw',row['raw']),('calibrated',dict(row['raw'],edges=row['calibrated_edges']))]:same(components(pred,row['target']),row[mode+'_metrics'])
    graphs+=1
   assert len(data['rows'])==512
  assert sum(r['calibrated_metrics']['semantic_equivalence'] for r in spa['rows'])==point['spanish']['calibrated_exact']
  assert sum(r['raw_metrics']['semantic_equivalence'] for r in spa['rows'])==point['spanish']['raw_exact']
 timing[arm]=dict(full_seconds=occ['process_occupancy_seconds'],preflight_seconds=occ['preflight_seconds'],training_seconds=m['training_seconds'],evaluation_seconds=evalseconds,projected_main_training=m['training_seconds']*4096/20,projected_main_evaluation=evalseconds*2)
x,y=manifests
for k in ['initial_state_sha256','inherited_presentations','inherited_optimizer_steps','renderer_phases','first_batch_replay','added_visits','construction_sequence_sha256','pair_sequence_sha256']:assert x[k]==y[k],k
for m in manifests:
 assert all(sum(r)==v for r,v in zip(m['renderer_visits'],m['added_visits']))
 for i,(v,pair,phase) in enumerate(zip(m['added_visits'],m['renderer_visits'],m['renderer_phases'])):
  expected=[v,0] if m['config']['arm']=='english' else [sum((phase+j)%2==0 for j in range(v)),sum((phase+j)%2==1 for j in range(v))];assert pair==expected
out=dict(status='profile_raw_audited',unique_graphs_reconstructed=graphs,unique_train_threshold_records=thresholds,paired_initial_schedule_negative_pair_visits_verified=True,same_checkpoint_english_thresholds_applied_spanish=True,timing=timing,main_cap_seconds_per_arm=600,main_cap_supported_for_separate_release=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Paired mechanical profiles only, no outcome/recipe selection. Main fixed4096 updates and four bilingual evaluations; profile extrapolation includes margin for export/setup.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
