"""Independent S15 mechanical profile raw graphs, pairing and timing contracts."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));dev=[json.loads(s) for s in gzip.open(base/'s15-shape-cache-v2/development.jsonl.gz','rt')];expected=[r for arity in (3,4) for facts in (3,4) for r in [x for x in dev if (x['arity'],x['facts'])==(arity,facts)][:16]];parent=load(base/'s13-english-main/evaluation-u24576.json.gz');ms={};times={};bindings={};graphs=thresholds=0
for arm in ('control','mixed'):
 root=base/f's15-{arm}-profile-v2';m=load(root/'manifest.json.gz');ms[arm]=m;cfg=m['config'];assert cfg['job']=='profile' and cfg['added_updates']==20 and cfg['checkpoints']==[0,20];assert m['added_presentations']==160 and sum(m['visits'])==160 and m['common_presentations']==74;assert m['inherited_optimizer_steps']==[24576.]
 for group in ('source_sha256','launch_source_sha256'):
  for name,h in cfg[group].items():assert sha(a.repo/'src/topoformer'/name)==h
 occ=json.loads((root/f's15-{arm}-profile-v2.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out'];assert occ['config_sha256']==sha(a.repo/f'configs/campaign-s15-{arm}-profile-frozen-v2.json');evaluation=0
 for point in m['curves']:
  ep=root/point['evaluation']['artifact'];assert sha(ep)==point['evaluation']['sha256'];e=load(ep);assert e['thresholds']==[x['threshold'] for x in e['calibration']];assert [r['seed'] for r in e['rows']]==[r['seed'] for r in expected];assert len(e['rows'])==64;evaluation+=point['evaluation']['seconds']
  if point['added_update']==0:assert e['thresholds']==parent['thresholds'] and e['train_rows']==parent['train_rows']
  cal=np.load(root/e['calibration_data_artifact']);assert sha(root/e['calibration_data_artifact'])==e['calibration_data_sha256']
  for j,r in enumerate(e['calibration']):
   cut,err=cutoff(cal['scores'][:,j],cal['targets'][:,j].astype(bool));assert abs(cut-r['threshold'])<1e-6 and err==r['train_errors'];thresholds+=1
  for row in e['rows']:
   for mode,pred in [('raw',row['raw']),('calibrated',{**row['raw'],'edges':row['calibrated_edges']})]:same(components(pred,row['target']),row[mode+'_metrics'])
   graphs+=1
  for arity in (3,4):
   for facts in (3,4):
    chosen=[r for r,d in zip(e['rows'],expected) if (d['arity'],d['facts'])==(arity,facts)];assert point['cells'][f'{arity}x{facts}']==dict(examples=16,raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in chosen),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in chosen))
 training=m['training_seconds'];other=occ['process_occupancy_seconds']-training-evaluation;times[arm]=dict(occupancy=occ['process_occupancy_seconds'],training=training,combined_evaluation=evaluation,residual=other,training_projection=training*4096/20,average_per_example_evaluation_projection=evaluation*2*2176/192,residual_projection=2*other,projected_main=training*4096/20+evaluation*2*2176/192+2*other,all_evaluation_as_DEV_upper_sensitivity=training*4096/20+evaluation*64+2*other)
 for path in root.iterdir():
  if path.is_file():bindings[str(path.relative_to(a.repo))]=sha(path)
for k in ('initial_state_sha256','inherited_optimizer_steps','inherited_presentations','construction_sequence_sha256','common_pair_sequence_sha256','common_presentations','visits'):assert ms['control'][k]==ms['mixed'][k]
assert ms['control']['pair_sequence_sha256']!=ms['mixed']['pair_sequence_sha256']
reported=json.loads((base/'s15-paired-profile-timing.json').read_text())
for arm,v in times.items():assert abs(v['projected_main']-reported['timing'][arm]['projected_main_seconds'])<1e-9
out=dict(profile_graphs_reconstructed=graphs,train_threshold_records_verified=thresholds,initial_parent_TRAIN128_predictions_thresholds_exact=True,paired_initial_schedule_visits_common_pair_hashes_equal=True,timing=times,input_sha256=bindings,cpu_audit_wall_seconds=time.monotonic()-t,scope='Mechanical timing/contracts only; no policy or recipe selection. Projection assumes combined per-example evaluation average because calibration and DEV were not separately timed. Cap decisions require separate freeze and root release; neither forecast is a rigorous upper bound. Main restarts original S11.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='input_sha256'})
