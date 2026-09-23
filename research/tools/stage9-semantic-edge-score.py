"""CPU-only frozen c2 edge-logit localization; never selects heldout thresholds."""
import argparse,json
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import fixtures,FixedNodeDecoder

p=argparse.ArgumentParser();p.add_argument('result_directory',type=Path);a=p.parse_args()
torch.set_num_threads(2)
config=json.loads((a.result_directory/'results.json').read_text())['config']
examples=fixtures(config['graphs'],config['width'],config['data_seed'],'cpu');result=[]
for seed in config['seeds']:
    model=FixedNodeDecoder(config['width'],rank=config['rank'],edge_affine=config.get('edge_affine',False))
    model.load_state_dict(torch.load(a.result_directory/f'head-False-conditional-False-seed{seed}.pt',weights_only=True,map_location='cpu'))
    rows=[]
    with torch.no_grad():
        for nodes,pairs,gold,audit in examples:
            edge,_=model(nodes,pairs);truth=gold['edges'].flatten(0,1)
            rows.append(dict(seed=audit['seed'],fp=int((edge.gt(0)&~truth).sum()),fn=int((edge.le(0)&truth).sum()),minimum_positive=float(edge[truth].min()),maximum_negative=float(edge[~truth].max()),positive_quantiles=torch.quantile(edge[truth],torch.tensor([0.,.1,.5,.9,1.])).tolist(),negative_quantiles=torch.quantile(edge[~truth],torch.tensor([0.,.1,.5,.9,1.])).tolist()))
    result.append(dict(seed=seed,rows=rows))
print(json.dumps(result,indent=2))
