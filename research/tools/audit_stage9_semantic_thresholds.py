"""Independent CPU frozen inference and separability check; training fixtures only."""
import argparse,json,math
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import FixedNodeDecoder,fixtures
from topoformer.thinking_language import ROLES
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();torch.set_num_threads(2)
data=json.loads((a.root/'results.json').read_text());cfg=data['config'];graphs=fixtures(cfg['graphs'],cfg['width'],cfg['data_seed'],'cpu');runs=[]
for seed in cfg['seeds']:
 model=FixedNodeDecoder(cfg['width'],rank=cfg['rank'],edge_affine=cfg.get('edge_affine',False));model.load_state_dict(torch.load(a.root/f'head-False-conditional-False-seed{seed}.pt',weights_only=True));model.eval();records=[]
 with torch.no_grad():
  for nodes,pairs,gold,info in graphs:
   edge,slot=model(nodes,pairs);records.append((edge,gold['edges'].flatten(0,1),slot.argmax(-1)-1,gold['slots'].flatten(),info['seed']))
 scores=torch.cat([r[0]for r in records]);labels=torch.cat([r[1]for r in records]);threshold=[];relations=[]
 for index,name in enumerate(ROLES):
  positive=scores[:,index][labels[:,index]];negative=scores[:,index][~labels[:,index]]
  maximum=float(negative.max());minimum=float(positive.min())if len(positive)else None;threshold.append(maximum)
  relations.append(dict(relation=name,positive=len(positive),negative=len(negative),minimum_positive=minimum,maximum_negative=maximum,strictly_separable=minimum is None or minimum>maximum))
 t=torch.tensor(threshold);rows=[]
 for score,truth,slot,goldslot,seedgraph in records:
  exactslots=bool(slot[truth.any(-1)].eq(goldslot[truth.any(-1)]).all());positive=int(truth.sum());negative=truth.numel()-positive
  policies={'raw':score>0,'calibrated':score>t,'analytical_oracle':score+math.log(positive/negative)>0}
  rows.append(dict(graph_seed=seedgraph,**{f'{k}_errors':int(v.ne(truth).sum())for k,v in policies.items()},**{f'{k}_exact':bool(v.eq(truth).all())and exactslots for k,v in policies.items()}))
 runs.append(dict(seed=seed,relations=relations,graphs=rows))
print(json.dumps(dict(kind='independent CPU frozen inference; posthoc eight-training-graph diagnostic only',runs=runs),indent=2))
