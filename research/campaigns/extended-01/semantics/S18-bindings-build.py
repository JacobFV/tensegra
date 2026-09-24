"""CPU immutable provenance binding only; no checkpoints loaded or model calls."""
import gzip,hashlib,json
from pathlib import Path
BASE=Path('/home/brandonin/topoformer-campaign01-semantics')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def gz(p):return json.load(gzip.open(p))
original=BASE/'results/s15-mixed-main-v3';m=gz(original/'manifest.json.gz')
s17=BASE/'results/s17-train-calibration-main-v1';s=gz(s17/'manifest.json.gz');r=next(x for x in s['artifacts'] if x['arm']=='mixed')
records=[]
for curve in m['curves']:
 u=curve['update'];records.append(dict(added_update=curve['added_update'],update=u,
  checkpoint=bind(original/f'model-u{u}.pt'),historical_evaluation=bind(original/curve['evaluation']['artifact'])))
result=dict(original_manifest=bind(original/'manifest.json.gz'),original_config=m['config'],
 original_initial_state_sha256=m['initial_state_sha256'],original_final_state_sha256=m['final_state_sha256'],
 expected_streams={k:m[k] for k in ('construction_sequence_sha256','pair_sequence_sha256','optimizer_tokens','optimizer_nodes','optimizer_edges')},
 checkpoints=records,s17_manifest=bind(s17/'manifest.json.gz'),
 matched_endpoint=bind(s17/r['artifact']),matched_endpoint_calibration=bind(s17/r['calibration_npz']),
 selection=bind(BASE/'data/s17-calibration-selection-v1/selection.json'),
 selection_audit=bind(BASE/'data/s17-calibration-selection-v1/audit.json'),
 parent=bind(BASE/'results/s11-lr-decay-n8192-196k-dev201/model-u24576.pt'),
 data_audit=bind(BASE/'data/s01/audit.json'),historical_train=bind(BASE/'data/s01/train.jsonl.gz'),
 shape_audit=bind(BASE/'data/s15-shape-v2/audit.json'),
 caches={k:bind(BASE/f'data/s15-shape-v2/{k}.jsonl.gz') for k in ('train_mixed','development','confirmation')})
with Path('/tmp/s18-bindings.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
