"""Independent S13 main arm provenance, calibration and graph metric audit."""
import argparse,gzip,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--arm',choices=('english','mixed'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifests=[];seen=set();graphs=0;thresholds=0;timing={}
for arm in (a.arm,):
 root=a.root/f's13-{arm}-main';m=load(root/'manifest.json.gz');manifests.append(m);cfg=m['config']
 for name,h in cfg['main_source_sha256'].items():assert sha(a.source_root/name)==h
 for name,h in m['source_sha256'].items():assert sha(a.source_root/name)==h
 assert cfg['added_updates']==4096 and cfg['checkpoints']==[0,1024,2048,4096] and m['added_presentations']==32768 and m['added_visits']==[4]*8192 and m['inherited_optimizer_steps']==[24576.]
 occ=json.loads((root/f's13-{arm}-main-v1.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out']
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
 timing[arm]=dict(full_seconds=occ['process_occupancy_seconds'],preflight_seconds=occ['preflight_seconds'],training_seconds=m['training_seconds'],evaluation_seconds=evalseconds,fixed_main_training=m['training_seconds'],fixed_main_evaluation=evalseconds)
for m in manifests:
 assert all(sum(r)==v for r,v in zip(m['renderer_visits'],m['added_visits']))
 for i,(v,pair,phase) in enumerate(zip(m['added_visits'],m['renderer_visits'],m['renderer_phases'])):
  expected=[v,0] if m['config']['arm']=='english' else [sum((phase+j)%2==0 for j in range(v)),sum((phase+j)%2==1 for j in range(v))];assert pair==expected
out=dict(status='main_arm_raw_audited',arm=a.arm,unique_graphs_reconstructed=graphs,unique_train_threshold_records=thresholds,renderer_visits_verified=True,same_checkpoint_english_thresholds_applied_spanish=True,timing=timing,input_sha256={str(p.relative_to(a.root.parents[3])):sha(p) for p in root.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-tick,scope='Fixed main arm4096updates/four bilingual curves. Same English TRAIN128 threshold policy applies Spanish; paired state/sampler and cross-arm comparison are separate.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
