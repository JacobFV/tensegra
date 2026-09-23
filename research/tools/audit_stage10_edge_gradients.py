"""CPU frozen replay: explicit score derivatives and symmetric gradient direction."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import FixedNodeDecoder,fixtures
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('diagnostic',type=Path);a=p.parse_args();torch.set_num_threads(2);started=time.monotonic()
record=json.loads(a.diagnostic.read_text());cfg=record['config'];model=FixedNodeDecoder(cfg['width'],rank=cfg['rank'],edge_affine=True)
path=a.directory/'head-False-conditional-True-seed10.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();examples=fixtures(8,cfg['width'],cfg['data_seed'],'cpu');params=list(model.source.parameters())+list(model.target.parameters());gradients=[torch.zeros_like(v)for v in params]
for nodes,pairs,gold,_ in examples:
 scores,_=model(nodes,pairs);truth=gold['edges'].flatten(0,1);positive=int(truth.sum());negative=truth.numel()-positive
 derivative=(scores.detach().sigmoid()-truth.float())*torch.where(truth,.5/positive,.5/negative)/8
 parts=torch.autograd.grad(scores,params,grad_outputs=derivative)
 gradients=[a+b for a,b in zip(gradients,parts)]
original=[v.detach().clone()for v in params];epsilon=4.;values={}
for sign in (-1,1):
 with torch.no_grad():
  for v,base,g in zip(params,original,gradients):v.copy_(base+sign*epsilon*g)
  for nodes,pairs,_,info in examples:
   scores,_=model(nodes,pairs)
   for index,failure in enumerate(record['failures']):
    if failure['graph_seed']==info['seed']:values[index,sign]=float(scores[failure['source_index']*len(nodes)+failure['target_index'],3])
rows=[];errors=[]
for i,f in enumerate(record['failures']):
 estimate=(values[i,-1]-values[i,1])/(2*epsilon);expected=f['negative_full_gradient_logit_direction'];absolute=abs(estimate-expected)
 rows.append(dict(graph_seed=f['graph_seed'],source=f['source_index'],target=f['target_index'],symmetric_direction=estimate,reported_direction=expected,absolute_error=absolute))
 if absolute>2e-6 or estimate>=0:errors.append(i)
norm=float(sum(g.square().sum()for g in gradients).sqrt())
if abs(norm-record['edge_gradient_norm'])>1e-7:errors.append('gradient_norm')
if digest!=record['checkpoint_sha256']:errors.append('checkpoint')
print(json.dumps(dict(kind='CPU frozen replay; explicit sigmoid score derivatives plus symmetric finite difference along full TRAIN gradient; not an Adam update',epsilon=epsilon,checkpoint_sha256=digest,gradient_norm=norm,rows=rows,errors=errors,passed=not errors,seconds=time.monotonic()-started),indent=2))
