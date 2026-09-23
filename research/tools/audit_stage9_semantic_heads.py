"""Independent reconstruction of privileged fixed-node decoder outcomes."""
import argparse,gzip,json
from pathlib import Path

def audit(paths):
 errors=[];counts=[];rows_checked=0
 for path in paths:
  with gzip.open(path,'rt') as f:data=json.load(f)
  init={}
  for run in data['runs']:
   key=run['seed'],run['head']
   if key in init and init[key]!=run['initial_state_sha256']:errors.append([str(path),key,'paired_init'])
   init[key]=run['initial_state_sha256']
   checkpoints=list(run['curve'])
   if run.get('calibrated'):
    c=run['calibrated']['rows'];checkpoints.append(dict(update='calibrated_train_final',exact_graph=sum(r['exact_graph']for r in c)/len(c),rows=c))
   for checkpoint in checkpoints:
    exacts=[]
    for row in checkpoint['rows']:
     p={tuple(e)for e in row['predicted_edges']};g={tuple(e)for e in row['gold_edges']};edge_errors=len(p^g)
     truepairs=sorted({e[:2]for e in g});gold=dict(zip(truepairs,row['gold_slots']));pred=dict(zip(truepairs,row['predicted_slots']))
     slot_errors=sum(pred[k]!=v for k,v in gold.items());exact=not edge_errors and not slot_errors
     if (edge_errors,slot_errors,exact)!=(row['edge_errors'],row['slot_errors_on_gold_edges'],row['exact_graph']):errors.append([str(path),run['seed'],checkpoint['update'],row['seed'],'counts'])
     if 'union_pairs'in row:
      union={tuple(e)for e in row['union_pairs']};want={e[:2]for e in p|g}
      up=dict(zip(map(tuple,row['union_pairs']),row['predicted_slots_on_union']))
      if union!=want or any(up[k]!=pred[k]for k in truepairs):errors.append([str(path),'union_slot'])
     exacts.append(exact);rows_checked+=1
    if sum(exacts)/len(exacts)!=checkpoint['exact_graph']:errors.append([str(path),'mean'])
   last=run['curve'][-1];counts.append(dict(file=path.name,seed=run['seed'],head=run['head'],objective=run['objective'],final_exact=sum(r['exact_graph']for r in last['rows']),graphs=len(last['rows']),updates=last['update'],calibrated_exact=sum(r['exact_graph']for r in run['calibrated']['rows'])if run.get('calibrated')else None))
 return dict(rows=rows_checked,runs=len(counts),final_counts=counts,errors=errors,passed=bool(counts)and not errors)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.paths);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='final_counts'}));raise SystemExit(0 if r['passed']else 1)
