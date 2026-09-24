"""CPU deterministic selection from actual TRAIN, before any S17 model scores."""
import collections,gzip,hashlib,json
from pathlib import Path
root=Path('research/results/campaign-01/semantics');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'))
out=root/'s17-calibration-selection';out.mkdir(exist_ok=False);selections={};audit={}
for arm,quotas in [('control',{'4x3':64,'4x4':64}),('mixed',{'3x3':64,'4x3':32,'4x4':32})]:
 path=root/'s15-shape-cache-v2'/f'train_{arm}.jsonl.gz';rows=[json.loads(line) for line in gzip.open(path,'rt')];counts=collections.Counter();selected=[];roles=collections.Counter()
 for index,row in enumerate(rows):
  cell=f"{row['arity']}x{row['facts']}"
  if counts[cell]>=quotas.get(cell,0):continue
  counts[cell]+=1;selected.append(dict(index=index,seed=row['seed'],semantic_sha256=row['semantic_sha256'],alpha_sha256=row['alpha_sha256'],cell=cell));roles.update(edge[2] for edge in row['edges'])
 assert dict(counts)==quotas and len(selected)==128
 m=load(root/f's15-{arm}-main-v3/manifest.json.gz');assert all(m['visits'][r['index']]==8 for r in selected)
 selections[arm]=selected;audit[arm]=dict(train_cache_sha256=sha(path),counts=dict(counts),gold_relation_counts=dict(roles),manifest_sha256=sha(root/f's15-{arm}-main-v3/manifest.json.gz'),checkpoint_sha256=m['curves'][-1]['checkpoint_sha256'],evaluation_sha256=m['curves'][-1]['evaluation']['sha256'],model_state_sha256=m['final_state_sha256'],all_selected_visited_eight_times=True)
p=out/'selection.json';p.write_text(json.dumps(selections,indent=2)+'\n');(out/'audit.json').write_text(json.dumps(dict(source_sha256=sha(__file__),protocol_sha256=sha('research/campaigns/extended-01/semantics/S17-protocol.md'),selection_sha256=sha(p),arms=audit,no_model_import_or_inference=True),indent=2)+'\n');print(json.dumps(audit,indent=2))
