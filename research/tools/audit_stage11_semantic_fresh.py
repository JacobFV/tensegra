"""Independent frozen-threshold fresh-graph metric and provenance audit."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
from audit_stage11_semantic_text import components,same

def load(p):
 with (gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(path,training,audit_path):
 x=load(path);train=load(training);data=load(audit_path);repo=Path(__file__).resolve().parents[2]
 assert sha(training)==x['training_archive_sha256']and sha(audit_path)==x['audit_sha256']
 assert hashlib.sha256(json.dumps(x['config'],sort_keys=True).encode()).hexdigest()==x['config_sha256']
 assert hashlib.sha256(subprocess.check_output(['git','show','7936417:src/topoformer/semantic_text_acquisition_fresh.py'],cwd=repo)).hexdigest()==x['source_sha256']
 assert x['dependency_sha256']==train['source_sha256']and x['value_vocabulary']==train['value_vocabulary']
 assert [r['seed']for r in x['runs']]==[30,31,32]and len(data['rows'])==data['accepted']==512
 assert len({r['semantic_sha256']for r in data['rows']})==512
 baseline={};result=[];verified=0
 for run,old in zip(x['runs'],train['runs']):
  assert run['seed']==old['seed']and run['checkpoint_sha256']==old['curves'][-1]['checkpoint_sha256']
  assert run['thresholds']==[r['threshold']for r in old['curves'][-1]['calibration']]
  assert len(run['rows'])==512
  totals={policy:dict(exact=0,node_tp=0,node_pred=0,node_gold=0,edge_tp=0,edge_pred=0,edge_gold=0,ordered_tp=0,ordered_pred=0,ordered_gold=0,type_sum=0.,copy_sum=0.)for policy in ('raw','calibrated')}
  for row,identity in zip(run['rows'],data['rows']):
   assert row['graph_seed']==identity['seed']and row['graph_sha256']==identity['graph_sha256']
   key=row['graph_seed'];digest=hashlib.sha256(json.dumps(row['target'],sort_keys=True).encode()).hexdigest()
   if key in baseline:assert baseline[key]==digest
   else:baseline[key]=digest
   for policy in totals:
    m=components(row[policy],row['target']);same(m,row[policy+'_metrics']);verified+=1;acc=totals[policy];acc['exact']+=int(m['semantic_equivalence']);acc['type_sum']+=m['node_type_accuracy'];acc['copy_sum']+=m['identity_copy_accuracy']
    for short,full in [('node','node'),('edge','typed_edge'),('ordered','ordered_edge')]:
     for suffix,name in [('tp','true_positive'),('pred','predicted_count'),('gold','gold_count')]:acc[short+'_'+suffix]+=m[full][name]
  for policy,a in totals.items():
   result.append(dict(seed=run['seed'],policy=policy,examples=512,exact=a['exact'],gate_passed=a['exact']/512>.95,node_f1=2*a['node_tp']/(a['node_pred']+a['node_gold']),typed_edge_f1=2*a['edge_tp']/(a['edge_pred']+a['edge_gold']),ordered_edge_f1=2*a['ordered_tp']/(a['ordered_pred']+a['ordered_gold']),mean_node_type=a['type_sum']/512,mean_copy=a['copy_sum']/512,counts=a))
 return dict(graph_decisions_verified=verified,distinct_underlying_graphs=512,all_frozen_thresholds_and_checkpoints_match=True,all_source_and_archive_hashes_match=True,results=result,heldout_renderer_allowed=all(r['gate_passed']for r in result if r['policy']=='calibrated'),scope='Same512fresh known-English graphs across3frozen seeds; inference-only transfer from8training constructions, not an adequate-exposure scaling test.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);p.add_argument('--training',type=Path,required=True);p.add_argument('--data-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=audit(a.path,a.training,a.data_audit);a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='results'})
