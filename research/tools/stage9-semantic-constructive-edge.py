"""Privileged constructive edge-head ceiling, not a learned acquisition result."""
import argparse,json,math
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import fixtures,FixedNodeDecoder,evaluate
from topoformer.thinking_language import ROLES

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
torch.set_num_threads(2);config=json.loads((a.directory/'results.json').read_text())['config']
examples=fixtures(config['graphs'],config['width'],config['data_seed'],'cpu');rows=[]
for seed in config['seeds']:
    model=FixedNodeDecoder(config['width'],rank=config['rank'],edge_affine=config.get('edge_affine',False))
    model.load_state_dict(torch.load(a.directory/f'head-False-conditional-False-seed{seed}.pt',weights_only=True,map_location='cpu'))
    with torch.no_grad():
        model.source.weight.zero_();model.target.weight.zero_()
        if model.source.bias is not None:model.source.bias.zero_();model.target.bias.zero_()
        for nodes,pairs,gold,audit in examples:
            n=len(nodes)
            if n>model.rank:raise ValueError('constructive rank must cover max local graph size')
            global_codes=nodes.argmax(-1)
            for j,code in enumerate(global_codes):model.target.weight[j,code]=1.
            for i,code in enumerate(global_codes):
                for relation in range(len(ROLES)):
                    model.source.weight[relation*model.rank:relation*model.rank+n,code]=(gold['edges'][i,:,relation].float()*2-1)*math.sqrt(model.rank)
    outcomes=evaluate(model,examples)
    rows.append(dict(seed=seed,complete_graphs=sum(r['exact_graph'] for r in outcomes),graphs=len(outcomes),edge_errors=sum(r['edge_errors'] for r in outcomes),slot_errors=sum(r['slot_errors_on_gold_edges'] for r in outcomes)))
print(json.dumps(dict(kind='gold-constructed edge weights plus frozen learned slots; capacity ceiling only',rows=rows),indent=2))
