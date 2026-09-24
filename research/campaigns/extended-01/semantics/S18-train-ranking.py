"""Archived actualTRAIN128 only: ranking versus fixed-policy graph conjunction."""
import ast,collections,gzip,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;ROOT=Path('research/results/campaign-01/semantics')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.load(gzip.open(p,'rt'))
def rank_metrics(scores,truth):
 """Tie-aware AUROC and bucketed AP; no threshold optimization/selection."""
 positive=int(truth.sum());negative=len(truth)-positive
 if not positive or not negative:return {'positive':positive,'negative':negative,'auroc':None,'average_precision':None}
 order=np.argsort(scores,kind='stable');s=scores[order];y=truth[order].astype(np.int64);ends=np.r_[np.flatnonzero(np.diff(s))+1,len(s)];starts=np.r_[0,ends[:-1]];pos=np.add.reduceat(y,starts);neg=ends-starts-pos
 auc=float(np.sum(pos*(np.cumsum(neg)-neg+neg/2))/(positive*negative))
 p=pos[::-1];n=neg[::-1];ap=float(np.sum((p/positive)*np.cumsum(p)/(np.cumsum(p)+np.cumsum(n))))
 return {'positive':positive,'negative':negative,'auroc':auc,'average_precision':ap}
def main():
 p=HERE/'S17-analyze.py';spec=importlib.util.spec_from_file_location('s17',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 analysis=json.loads((ROOT/'s18-paired-analysis.json').read_text());selection=json.loads((ROOT/'s17-calibration-selection/selection.json').read_text())['mixed'];selection_byseed={s['seed']:s for s in selection}
 roles=None
 for n in ast.parse(Path('src/topoformer/thinking_language.py').read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES':roles=ast.literal_eval(n.value)
 results={};inputs={}
 for arm in ('original','context','workspace_control'):
  inv=analysis['artifact_inventory'][f'{arm}/4096/matched'];ep=Path(inv['evaluation_path']);npz=Path(inv['calibration_path']);assert sha(ep)==inv['evaluation_sha256'] and sha(npz)==inv['calibration_sha256'];d=load(ep);assert [r['seed'] for r in d['train_rows']]==[r['seed'] for r in selection]
  with np.load(npz,allow_pickle=False) as f:scores=f['scores'];labels=f['targets'].astype(bool);offsets=f['offsets']
  cut=np.asarray(d['thresholds']);error=(scores>cut)!=labels;cells={}
  for cell in ('3x3','4x3','4x4'):
   ids=[i for i,r in enumerate(d['train_rows']) if selection_byseed[r['seed']]['cell']==cell];index=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids]);ss=scores[index];yy=labels[index];ee=error[index];relation={}
   for i,role in enumerate(roles):
    pred=ss[:,i]>cut[i];y=yy[:,i];relation[role]={**rank_metrics(ss[:,i],y),'frozen_threshold':float(cut[i]),'false_positive':int((pred&~y).sum()),'false_negative':int((~pred&y).sum()),'fixed_error_rate':float((pred!=y).mean())}
   histogram=collections.Counter();patterns=collections.Counter();graph_role_errors=collections.Counter();full_success=0;edges_exact=0;active_error_free=0;active_error_free_full_fail=0;errors=[]
   for i in ids:
    row=d['train_rows'][i];g=a.unpack(row['target']);p=a.unpack(row['raw']);p['edges']=a.edge(row['calibrated_edges']);comp=a.components(p,g);full_success+=all(comp.values());edges_exact+=comp['edges'];patterns[','.join(k for k,v in comp.items() if not v) or 'none']+=1
    count=int(error[offsets[i]:offsets[i+1]].sum());errors.append(count);histogram['0' if count==0 else '1' if count==1 else '2-5' if count<=5 else '6+']+=1;active_error_free+=count==0;active_error_free_full_fail+=(count==0 and not all(comp.values()))
    graph_role_errors.update(role for j,role in enumerate(roles) if error[offsets[i]:offsets[i+1],j].any())
   cells[cell]=dict(graphs=len(ids),active_pair_relation_decisions=int(ee.size),active_positive_labels=int(yy.sum()),active_fixed_errors=int(ee.sum()),active_fixed_error_rate=float(ee.mean()),complete_graphs=int(full_success),full_edges_exact=int(edges_exact),active_error_free=int(active_error_free),active_error_free_but_full_graph_failed=int(active_error_free_full_fail),active_errors_per_graph_histogram=dict(histogram),active_errors_per_graph_quantiles={str(q):float(np.quantile(errors,q)) for q in (0,.25,.5,.75,1)},full_graph_failure_patterns=dict(patterns),graphs_with_error_by_relation=dict(graph_role_errors),relations=relation)
  results[arm]=cells;inputs[arm]=inv
 out=dict(scope='Posthoc actualTRAIN128 only, same labels used in calibration. Optimistic in-sample diagnostic, not generalization. Fixed archived cutoffs only; no threshold search, refit, model calls or independent-error assumption. Ranking is tie-aware AUROC and bucketed AP on predicted-present calibration pairs; absent-class roles undefined. Full packed graph conjunction reported separately.',inputs=inputs,analysis_sha256=sha(ROOT/'s18-paired-analysis.json'),source_sha256=sha(Path(__file__)),results=results)
 (ROOT/'s18-train-ranking.json').write_text(json.dumps(out,indent=2)+'\n')
 for arm,c in results.items():
  r=c['3x3'];print(arm,r['complete_graphs'],r['full_edges_exact'],r['active_errors_per_graph_histogram'],r['full_graph_failure_patterns']);print({k:r['relations'][k] for k in ('refers_to','argument')})
if __name__=='__main__':
 assert rank_metrics(np.array([0.,1.]),np.array([False,True]))['auroc']==1.
 assert rank_metrics(np.array([0.,0.]),np.array([False,True]))['auroc']==.5
 assert rank_metrics(np.array([0.,0.]),np.array([False,True]))['average_precision']==.5
 main()
