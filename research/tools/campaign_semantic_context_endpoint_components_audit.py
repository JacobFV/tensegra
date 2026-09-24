"""Independent sparse final S18 components/role errors and aggregate metric checks."""
import argparse,ast,collections,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--archive',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();load=lambda p:json.load(gzip.open(p,'rt'));base=a.repo/'research/results/campaign-01/semantics';s=json.loads(a.analysis.read_text());cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(base/'s15-shape-cache-v2/development.jsonl.gz','rt'))};roles=next(ast.literal_eval(n.value) for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES');checked=0
for arm in ('original','context','workspace_control'):
 if arm=='original':paths={'historical':base/'s15-mixed-main-v3/evaluation-u28672.json.gz','matched':base/'s17-calibration-main-v1/mixed/evaluation-u28672.json.gz'}
 else:paths={policy:a.archive/f's18-main-v1/{arm}/{policy}-u28672/evaluation-u28672.json.gz' for policy in ('historical','matched')}
 for policy,path in paths.items():
  d=load(path);rows={r['semantic_sha256']:r for r in d['rows']}
  for label in (('raw',policy) if policy=='historical' else (policy,)):
   for cell in ('3x3','3x4','4x3','4x4'):
    ids=sorted(k for k,r in cache.items() if f"{r['arity']}x{r['facts']}"==cell);component=collections.Counter();micro={k:collections.Counter() for k in ('node','typed_edge','ordered_edge')};macro=collections.Counter();relations={r:dict(false_positive=0,false_negative=0,wrong_slots_on_present_gold=0) for r in roles};complete=0
    for event in ids:
     row=rows[event];pred=row['raw'];gold=row['target'];P={i for i,v in enumerate(pred['presence']) if v};G={i for i,v in enumerate(gold['presence']) if v};ge=edge_set(gold);pe={e for e in edge_set({**pred,'edges':pred['edges'] if label=='raw' else row['calibrated_edges']}) if e[0] in P and e[1] in P};flags=dict(presence=P==G,kind=all(pred['kind'][i]==gold['kind'][i] for i in G),value=all(pred['value'][i]==v for i,v in enumerate(gold['value']) if v>=0),copy=all(pred['copy'][i]==v for i,v in enumerate(gold['copy']) if v>=0),edges=pe==ge,slots=all(pred['slots'][i][j]==gold['slots'][i][j] for i,j,r in ge));component.update({k:int(v) for k,v in flags.items()});complete+=all(flags.values());metric=row['raw_metrics' if label=='raw' else 'calibrated_metrics']
     for k in micro:
      micro[k].update({n:metric[k][n] for n in ('true_positive','predicted_count','gold_count')});macro[k+'_f1']+=metric[k]['f1']/512
     for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):macro[k]+=metric[k]/512
     for r,name in enumerate(roles):
      pp={e for e in pe if e[2]==r};gg={e for e in ge if e[2]==r};v=relations[name];v['false_positive']+=len(pp-gg);v['false_negative']+=len(gg-pp);v['wrong_slots_on_present_gold']+=sum(pred['slots'][i][j]!=gold['slots'][i][j] for i,j,_ in pp&gg)
     checked+=1
    got=dict(examples=512,complete=complete,exact_components=dict(component),micro_counts={k:{**v,'f1':2*v['true_positive']/max(1,v['predicted_count']+v['gold_count'])} for k,v in micro.items()},macro_graph_metrics=dict(macro),relations=relations);same(got,s['results'][arm]['4096'][label][cell])
out=dict(endpoint_policy_graphs_reconstructed=checked,all36_final_cell_summaries_exact=True,components_macro_metrics_micro_counts_and_every_role_errors_exact=True,analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),cpu_audit_wall_seconds=time.monotonic()-tick,scope='Independent sparse endpoint component/role-error reconstruction; all fixed arms and raw/historical/matched policies. Prior pergraph fullmetric audits reused for aggregate metric sums; no model calls.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
