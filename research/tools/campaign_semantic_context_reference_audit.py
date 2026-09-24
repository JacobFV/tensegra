"""Independent sparse TRAIN fit-panel and immutable S18 baseline-reference audit."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));base=a.repo/'research/results/campaign-01/semantics';here=a.repo/'research/campaigns/extended-01/semantics';ref=json.loads((here/'S18-reference-baseline.json').read_text());paths=[]
for name in ('S15-mixed-main-audit.json','S17-main-audit.json'):
 receipt=json.loads((a.review/name).read_text())
 for path,h in receipt['input_sha256'].items():assert sha(a.repo/path)==h
for key,path in [('manifest_sha256',base/'s15-mixed-main-v3/manifest.json.gz'),('s17_mixed_evaluation_sha256',base/'s17-calibration-main-v1/mixed/evaluation-u28672.json.gz'),('selection_sha256',base/'s17-calibration-selection/selection.json'),('source_sha256',here/'S18-reference-extract.py'),('component_helper_sha256',here/'S17-analyze.py')]:assert sha(path)==ref[key];paths.append(path)
m=load(paths[0]);d=load(paths[1]);selection=json.loads(paths[2].read_text())['mixed'];assert ref['recipe']==m['config'];assert ref['training_source_sha256']==m['config']['source_sha256']
for group in ('training_hashes','exposure'):
 for k,v in ref[group].items():assert v==m[k]
assert ref['original_whole_process_seconds']==json.loads((a.review/'S15-mixed-main-audit.json').read_text())['full_occupancy_seconds']
state=json.loads((a.review/'S15-mixed-main-state-audit.json').read_text());assert len(ref['curves'])==4
for rc,c in zip(ref['curves'],m['curves'],strict=True):
 ep=base/f"s15-mixed-main-v3/evaluation-u{c['update']}.json.gz";assert sha(ep)==rc['historical_evaluation_sha256']==c['evaluation']['sha256'];assert rc['checkpoint_sha256']==c['checkpoint_sha256'];assert rc['raw_historical_cells']==c['cells'];assert rc['matched_calibration_available']==(c['added_update']==4096);assert rc['update']==c['update'] and rc['added_update']==c['added_update'];paths.append(ep)
assert state['initial_model_adamw_exact_parent'] and state['all32768_presentation_negative_queries_replayed'];assert state['added_presentations']==32768
panel={};failures=collections.Counter();assert len(d['train_rows'])==len(d['train_metrics'])==len(selection)==128
for row,metric,item in zip(d['train_rows'],d['train_metrics'],selection,strict=True):
 assert row['seed']==item['seed'];out=panel.setdefault(item['cell'],dict(examples=0,raw=dict(complete=0,exact_components=collections.Counter()),matched_calibrated=dict(complete=0,exact_components=collections.Counter())));out['examples']+=1;g=row['target'];p=row['raw'];P={i for i,v in enumerate(p['presence']) if v};G={i for i,v in enumerate(g['presence']) if v};ge=edge_set(g)
 for label,mkey in [('raw','raw'),('matched_calibrated','calibrated')]:
  pe={e for e in edge_set({**p,'edges':p['edges'] if label=='raw' else row['calibrated_edges']}) if e[0] in P and e[1] in P};flags=dict(presence=P==G,kind=all(p['kind'][i]==g['kind'][i] for i in G),value=all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0),copy=all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0),edges=pe==ge,slots=all(p['slots'][i][j]==g['slots'][i][j] for i,j,r in ge));assert all(flags.values())==bool(metric[mkey]['semantic_equivalence']);out[label]['complete']+=all(flags.values());out[label]['exact_components'].update({k:int(v) for k,v in flags.items()});failures[label+':'+','.join(k for k,v in flags.items() if not v)]+=1
assert json.loads(json.dumps(panel))==ref['actual_train128_panel'];assert {k:v['examples'] for k,v in panel.items()}=={'3x3':64,'4x3':32,'4x4':32};paths.extend([here/'S18-reference-baseline.json',here/'S18-reference-reuse.md']);out=dict(actual_train128_panel=panel,failure_patterns=failures,immutable_reference_bindings_verified=True,prior_raw_and_optimizer_query_audits_reused=True,original_whole_process_seconds=ref['original_whole_process_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-t,scope='Retrospective inspected-DEV baseline reuse, conditional on separately audited new runner identity. TRAIN128 overlaps threshold fitting and is optimistic, not independent generalization. No new model calls; original training cost retained; matched early curves require separately frozen evaluation.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
