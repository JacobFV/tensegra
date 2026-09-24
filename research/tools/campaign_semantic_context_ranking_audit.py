"""Independent S18 archived TRAIN rank/error/conjunction audit, no threshold fitting."""
import argparse,ast,collections,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import edge_set,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';summary=json.loads((base/'s18-train-ranking.json').read_text());analysis=json.loads((base/'s18-paired-analysis.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert summary['analysis_sha256']==sha(base/'s18-paired-analysis.json');assert summary['source_sha256']==sha(a.repo/'research/campaigns/extended-01/semantics/S18-train-ranking.py');selection=json.loads((base/'s17-calibration-selection/selection.json').read_text())['mixed'];roles=next(ast.literal_eval(n.value) for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES');verified=0;headline={}
def ranking(scores,y):
 assert np.isfinite(scores).all();P=int(y.sum());N=len(y)-P
 if not P or not N:return dict(positive=P,negative=N,auroc=None,average_precision=None)
 values,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True);pos=np.bincount(inverse,weights=y,minlength=len(values));ranks=np.cumsum(counts)-(counts-1)/2;auc=(float(np.dot(pos,ranks))-P*(P+1)/2)/(P*N);tp=np.cumsum(pos[::-1]);total=np.cumsum(counts[::-1]);ap=float(np.dot(pos[::-1]/P,tp/total));return dict(positive=P,negative=N,auroc=auc,average_precision=ap)
assert ranking(np.array([0.,0.]),np.array([False,True]))['auroc']==.5
for arm in ('original','context','workspace_control'):
 inv=summary['inputs'][arm];assert inv==analysis['artifact_inventory'][f'{arm}/4096/matched'];ep=Path(inv['evaluation_path']);npz=Path(inv['calibration_path']);ep=ep if ep.is_absolute() else a.repo/ep;npz=npz if npz.is_absolute() else a.repo/npz;assert sha(ep)==inv['evaluation_sha256'] and sha(npz)==inv['calibration_sha256'];d=json.load(gzip.open(ep,'rt'));cal=np.load(npz);scores=cal['scores'];labels=cal['targets'].astype(bool);offsets=cal['offsets'];errors=(scores>np.asarray(d['thresholds']))!=labels
 for cell in ('3x3','4x3','4x4'):
  ids=[i for i,x in enumerate(selection) if x['cell']==cell];indices=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids]);s=scores[indices];y=labels[indices];err=errors[indices];relation={}
  for j,name in enumerate(roles):
   pred=s[:,j]>d['thresholds'][j];relation[name]={**ranking(s[:,j],y[:,j]),'frozen_threshold':float(d['thresholds'][j]),'false_positive':int(np.sum(pred&~y[:,j])),'false_negative':int(np.sum(~pred&y[:,j])),'fixed_error_rate':float(np.count_nonzero(pred!=y[:,j])/len(pred))};verified+=1
  histogram=collections.Counter();patterns=collections.Counter();grapherrors=collections.Counter();full=edges=activefree=activefreefailed=0;errs=[]
  for i in ids:
   row=d['train_rows'][i];assert row['seed']==selection[i]['seed'];p=row['raw'];g=row['target'];P={j for j,v in enumerate(p['presence']) if v};G={j for j,v in enumerate(g['presence']) if v};ge=edge_set(g);pe={e for e in edge_set({**p,'edges':row['calibrated_edges']}) if e[0] in P and e[1] in P};flags=dict(presence=P==G,kind=all(p['kind'][j]==g['kind'][j] for j in G),value=all(p['value'][j]==v for j,v in enumerate(g['value']) if v>=0),copy=all(p['copy'][j]==v for j,v in enumerate(g['copy']) if v>=0),edges=pe==ge,slots=all(p['slots'][j][k]==g['slots'][j][k] for j,k,r in ge));full+=all(flags.values());edges+=flags['edges'];patterns[','.join(k for k,v in flags.items() if not v) or 'none']+=1;ee=errors[offsets[i]:offsets[i+1]];count=int(np.count_nonzero(ee));errs.append(count);histogram['0' if count==0 else '1' if count==1 else '2-5' if count<=5 else '6+']+=1;activefree+=count==0;activefreefailed+=count==0 and not all(flags.values());grapherrors.update(name for j,name in enumerate(roles) if np.any(ee[:,j]))
  got=dict(graphs=len(ids),active_pair_relation_decisions=int(err.size),active_positive_labels=int(y.sum()),active_fixed_errors=int(err.sum()),active_fixed_error_rate=float(err.mean()),complete_graphs=int(full),full_edges_exact=int(edges),active_error_free=int(activefree),active_error_free_but_full_graph_failed=int(activefreefailed),active_errors_per_graph_histogram=dict(histogram),active_errors_per_graph_quantiles={str(q):float(np.quantile(errs,q)) for q in (0,.25,.5,.75,1)},full_graph_failure_patterns=dict(patterns),graphs_with_error_by_relation=dict(grapherrors),relations=relation)
  expected=summary['results'][arm][cell]
  def check(x,y):
   if isinstance(x,dict):assert x.keys()==y.keys();[check(v,y[k]) for k,v in x.items()]
   elif x is None:assert y is None
   else:assert abs(x-y)<1e-10,(x,y)
  check(got,expected)
  if cell=='3x3':headline[arm]={k:got[k] for k in ('active_pair_relation_decisions','active_fixed_errors','complete_graphs','full_graph_failure_patterns')}
paths=[base/'s18-train-ranking.json',a.repo/'research/campaigns/extended-01/semantics/S18-train-ranking.py',a.repo/'research/campaigns/extended-01/semantics/S18-train-ranking-report.md'];out=dict(role_cell_arm_rank_checks=verified,all_fixed_error_histogram_quantile_component_conjunctions_exact=True,independent_auc_average_rank_and_AP_bucket_check=True,trained3x3=headline,input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-tick,scope='ActualTRAIN128 only, same labels fitted cutoffs; predicted-present-pair ranking conditioned on model presence. Absent-class roles undefined. No threshold search/refit/DEV/confirmation/modelcalls, no product-of-independent-edge-errors or no-information/capacity conclusion.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
