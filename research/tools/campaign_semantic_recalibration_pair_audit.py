"""Independent sparse policy comparisons and fixed profile/main replay for S17."""
import argparse,ast,collections,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import edge_set
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';base=root/'s17-calibration-main-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));summary=json.loads((root/'s17-paired-analysis.json').read_text());m=load(base/'manifest.json.gz');cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(root/'s15-shape-cache-v2/development.jsonl.gz','rt'))};roles=next(ast.literal_eval(n.value) for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES');receipt=json.loads((a.review/'S17-main-audit.json').read_text())
for path,h in receipt['input_sha256'].items():assert sha(a.repo/path)==h
assert summary['source_sha256']==sha(a.repo/'research/campaigns/extended-01/semantics/S17-analyze.py');assert summary['calibration_metadata']==[{k:v for k,v in r.items() if k!='evaluation'} for r in m['artifacts']];results={};reportcounts={};profile_replay=0
for rec in m['artifacts']:
 arm=rec['arm'];newpath=base/rec['artifact'];oldpath=root/f's15-{arm}-main-v3/evaluation-u28672.json.gz';assert summary['endpoint_artifacts'][arm]==dict(new=sha(newpath),old=sha(oldpath));d=load(newpath);old={r['semantic_sha256']:r for r in load(oldpath)['rows']};profile_root=root/'s17-calibration-profile-v1'/arm;pr=load(profile_root/'evaluation-u28672.json.gz');assert d['thresholds']==pr['thresholds'] and d['train_rows']==pr['train_rows'] and d['calibration']==pr['calibration'];pn=np.load(profile_root/'calibration-u28672.npz');mn=np.load(base/rec['calibration_npz']);assert pn.files==mn.files
 for k in pn.files:assert np.array_equal(pn[k],mn[k])
 byid={r['semantic_sha256']:r for r in d['rows']}
 for row in pr['rows']:assert row==byid[row['semantic_sha256']];profile_replay+=1
 cells={}
 for cell in ('3x3','3x4','4x3','4x4'):
  rows=[r for r in d['rows'] if f"{cache[r['semantic_sha256']]['arity']}x{cache[r['semantic_sha256']]['facts']}"==cell];assert len(rows)==512;out=dict(examples=512,policies={},historical_to_matched_complete=collections.Counter())
  for label in ('raw','historical','matched'):
   counts=dict(complete=0,exact_components=collections.Counter(),mean_metrics=collections.Counter(),relations={role:dict(false_positive=0,false_negative=0,wrong_slots_on_present_gold=0) for role in roles})
   for row in rows:
    oldrow=old[row['semantic_sha256']];assert row['raw']==oldrow['raw'] and row['target']==oldrow['target'];g=row['target'];p=row['raw'];P={i for i,v in enumerate(p['presence']) if v};G={i for i,v in enumerate(g['presence']) if v};ge=edge_set(g);edge=p['edges'] if label=='raw' else (oldrow if label=='historical' else row)['calibrated_edges'];pe={e for e in edge_set({**p,'edges':edge}) if e[0] in P and e[1] in P};flags=dict(presence=P==G,kind=all(p['kind'][i]==g['kind'][i] for i in G),value=all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0),copy=all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0),edges=pe==ge,slots=all(p['slots'][i][j]==g['slots'][i][j] for i,j,r in ge));metrics=(oldrow if label=='historical' else row)['raw_metrics' if label=='raw' else 'calibrated_metrics'];assert all(flags.values())==bool(metrics['semantic_equivalence']);counts['complete']+=all(flags.values());counts['exact_components'].update(k for k,v in flags.items() if v)
    for kind in ('node','typed_edge','ordered_edge'):counts['mean_metrics'][kind+'_f1']+=metrics[kind]['f1']/512
    for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):counts['mean_metrics'][k]+=metrics[k]/512
    for relation,name in enumerate(roles):
     pred={e for e in pe if e[2]==relation};gold={e for e in ge if e[2]==relation};stat=counts['relations'][name];stat['false_positive']+=len(pred-gold);stat['false_negative']+=len(gold-pred);stat['wrong_slots_on_present_gold']+=sum(p['slots'][i][j]!=g['slots'][i][j] for i,j,r in pred&gold)
   out['policies'][label]=counts
  for row in rows:out['historical_to_matched_complete'][f"{int(old[row['semantic_sha256']]['calibrated_metrics']['semantic_equivalence'])}->{int(row['calibrated_metrics']['semantic_equivalence'])}"]+=1
  out['signed_matched_minus_historical']={role:{k:out['policies']['matched']['relations'][role][k]-out['policies']['historical']['relations'][role][k] for k in out['policies']['matched']['relations'][role]} for role in roles};cells[cell]=out
 assert json.loads(json.dumps(cells))==summary['results'][arm];results[arm]=cells;reportcounts[arm]={cell:{p:v['complete'] for p,v in x['policies'].items()} for cell,x in cells.items()}
paths=[root/'s17-paired-analysis.json',a.repo/'research/campaigns/extended-01/semantics/S17-analyze.py',a.repo/'research/campaigns/extended-01/semantics/S17-report.md'];paths=[p for p in paths if p.exists()];out=dict(policy_graphs=12288,all_components_macro_metrics_role_errors_signed_changes_and_paired_transitions_exact=True,profile_main_TRAIN128_scores_targets_thresholds_exact=True,profile_main_DEV_replay_examples=profile_replay,complete=reportcounts,input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-t,scope='Exploratory actualTRAIN128 global calibration. All original raw/historical policies retained; exact calibration/first64profile replay preserved. Macro graph-F1 distinct from micro; no DEV fitting/threshold search, new gate, confirmation or retrospective S15 promotion.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
