"""CPU-only immutable S15/S17 baseline map and calibration-overlap TRAIN panel."""
import collections,gzip,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('research/results/campaign-01/semantics');HERE=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.load(gzip.open(p,'rt'))
p=HERE/'S17-analyze.py';spec=importlib.util.spec_from_file_location('s17',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
manifest_path=ROOT/'s15-mixed-main-v3/manifest.json.gz';manifest=load(manifest_path)
eval_path=ROOT/'s17-calibration-main-v1/mixed/evaluation-u28672.json.gz';evaluation=load(eval_path)
selection_path=ROOT/'s17-calibration-selection/selection.json';selection=json.loads(selection_path.read_text())['mixed']
assert sha(manifest_path)=='8e4ab8105d5e8bfc0212c6a95a09e65cbb1dbd98f9c8b33362ea5c710278a67b'
assert sha(eval_path)=='f5ad28c8d1dbee1b973baf00d2be3a56700b7b0c650c0798ae99b019fb337d4c'
assert sha(selection_path)=='5b8091e73320c5ef87bee39189933e6f3aa856c98ac07d93c6cb90d8d1c13172'
assert len(evaluation['train_rows'])==len(evaluation['train_metrics'])==len(selection)==128
panel={}
for row,metrics,item in zip(evaluation['train_rows'],evaluation['train_metrics'],selection,strict=True):
 assert row['seed']==item['seed'];cell=panel.setdefault(item['cell'],{'examples':0,'raw':{'complete':0,'exact_components':collections.Counter()},'matched_calibrated':{'complete':0,'exact_components':collections.Counter()}});cell['examples']+=1
 for label,key in [('raw','raw'),('matched_calibrated','calibrated')]:
  g=a.unpack(row['target']);pred=a.unpack(row['raw'])
  if label!='raw':pred['edges']=a.edge(row['calibrated_edges'])
  counts=a.components(pred,g);assert all(counts.values())==bool(metrics[key]['semantic_equivalence']);cell[label]['complete']+=int(all(counts.values()));cell[label]['exact_components'].update({k:int(v) for k,v in counts.items()})
assert {k:v['examples'] for k,v in panel.items()}=={'3x3':64,'4x3':32,'4x4':32}
curves=[]
for c in manifest['curves']:
 ep=ROOT/f"s15-mixed-main-v3/evaluation-u{c['update']}.json.gz";assert sha(ep)==c['evaluation']['sha256']
 curves.append({'update':c['update'],'added_update':c['added_update'],'checkpoint_remote':f"results/s15-mixed-main-v3/model-u{c['update']}.pt",'checkpoint_sha256':c['checkpoint_sha256'],'historical_evaluation_sha256':c['evaluation']['sha256'],'raw_historical_cells':c['cells'],'matched_calibration_available':c['added_update']==4096})
out={'scope':'Read-only reference reuse, not a new independent baseline or model run. TRAIN128 is identical to fitted calibration population: optimistic fit panel, not independent TRAIN generalization. No heldout3x4 TRAIN cell exists.','source_sha256':sha(Path(__file__)),'component_helper_sha256':sha(p),'manifest_sha256':sha(manifest_path),'s17_mixed_evaluation_sha256':sha(eval_path),'selection_sha256':sha(selection_path),'training_source_sha256':manifest['config']['source_sha256'],'recipe':manifest['config'],'training_hashes':{k:manifest[k] for k in ('initial_state_sha256','final_state_sha256','construction_sequence_sha256','pair_sequence_sha256','common_pair_sequence_sha256')},'exposure':{k:manifest[k] for k in ('added_presentations','optimizer_tokens','optimizer_nodes','optimizer_edges','training_seconds')},'original_whole_process_seconds':484.14140757126734,'curves':curves,'actual_train128_panel':panel}
(HERE/'S18-reference-baseline.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(panel,indent=2))
