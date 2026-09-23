"""CPU checkpoint replay and independent grouped-score threshold sweep."""
import argparse,hashlib,itertools,json
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import FixedNodeDecoder,fixtures
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();torch.set_num_threads(2)
saved=json.loads((a.root/'results.json').read_text());cfg=saved['config'];examples=fixtures(cfg['graphs'],cfg['width'],cfg['data_seed'],'cpu');runs=[];errors=[]
for run in saved['runs']:
 interaction=run['head']=='interaction';conditional=run['objective']=='edge_conditional';path=a.root/f"head-{interaction}-conditional-{conditional}-seed{run['seed']}.pt"
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=run['checkpoint_sha256']:errors.append([path.name,'checkpoint'])
 model=FixedNodeDecoder(cfg['width'],rank=cfg['rank'],interaction=interaction,edge_affine=True);model.load_state_dict(torch.load(path,weights_only=True));model.eval();cache=[]
 with torch.no_grad():
  for nodes,pairs,gold,info in examples:
   score,slots=model(nodes,pairs);cache.append((score,gold['edges'].flatten(0,1),slots.argmax(-1)-1,gold['slots'].flatten(),info['seed']))
 scores=torch.cat([x[0]for x in cache]);labels=torch.cat([x[1]for x in cache]);thresholds=[];records=[]
 for relation in range(scores.shape[1]):
  entries=sorted(zip(scores[:,relation].tolist(),labels[:,relation].tolist()));positive=sum(v for _,v in entries);negative=len(entries)-positive
  best=mistakes=negative;threshold=entries[0][0]-1
  for score,group in itertools.groupby(entries,key=lambda x:x[0]):
   group=list(group);npos=sum(y for _,y in group);mistakes+=npos-(len(group)-npos)
   if mistakes<best:best=mistakes;threshold=score
  old=run['calibrated']['relations'][relation]
  if (positive,negative,best)!=(old['positive'],old['negative'],old['train_errors']):errors.append([path.name,relation,'threshold_counts'])
  records.append(dict(relation=relation,positive=positive,negative=negative,minimum_train_errors=best,threshold_difference=threshold-old['threshold']))
  thresholds.append(threshold)
 threshold=torch.tensor(thresholds);rows=[]
 for score,truth,slot,goldslot,seed in cache:
  pred=score>threshold;edge_errors=int(pred.ne(truth).sum());slot_errors=int(slot[truth.any(-1)].ne(goldslot[truth.any(-1)]).sum());exact=edge_errors==0 and slot_errors==0
  old=next(x for x in run['calibrated']['rows']if x['seed']==seed)
  if (edge_errors,slot_errors,exact)!=(old['edge_errors'],old['slot_errors_on_gold_edges'],old['exact_graph']):errors.append([path.name,seed,'calibrated_counts'])
  rows.append(dict(graph_seed=seed,edge_errors=edge_errors,slot_errors=slot_errors,exact=exact))
 runs.append(dict(seed=run['seed'],head=run['head'],objective=run['objective'],checkpoint_sha256=digest,relations=records,graphs=rows))
print(json.dumps(dict(kind='independent CPU frozen inference and grouped-score calibration replay; same TRAIN fixtures only',runs=runs,errors=errors,passed=len(runs)==12 and not errors),indent=2))
