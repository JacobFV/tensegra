"""Posthoc TRAIN-only frozen c2 calibration. Never overrides the raw gate."""
import argparse,json,math
from pathlib import Path
import torch
from topoformer.semantic_contracts_study import fixtures,FixedNodeDecoder,evaluate
from topoformer.thinking_language import ROLES

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
torch.set_num_threads(2)
config=json.loads((a.directory/'results.json').read_text())['config']
examples=fixtures(config['graphs'],config['width'],config['data_seed'],'cpu');runs=[]
for seed in config['seeds']:
    model=FixedNodeDecoder(config['width'],rank=config['rank'],edge_affine=config.get('edge_affine',False))
    model.load_state_dict(torch.load(a.directory/f'head-False-conditional-False-seed{seed}.pt',weights_only=True,map_location='cpu'))
    cached=[];weights=[]
    with torch.no_grad():
        for nodes,pairs,gold,audit in examples:
            edge,slots=model(nodes,pairs);truth=gold['edges'].flatten(0,1)
            positive=int(truth.sum());negative=truth.numel()-positive
            weights.append(dict(graph_seed=audit['seed'],positive=positive,negative=negative,
                positive_weight=1/(2*positive),negative_weight=1/(2*negative),
                natural_logit_correction=math.log(positive/negative)))
            cached.append((edge,truth,slots.argmax(-1)-1,gold,audit))
    logits=torch.cat([x[0] for x in cached]);truth=torch.cat([x[1] for x in cached]);thresholds=[];relations=[]
    for r,name in enumerate(ROLES):
        scores=logits[:,r];labels=truth[:,r];positive=scores[labels];negative=scores[~labels]
        # Minimize TRAIN misclassification over all possible threshold boundaries.
        # Equal scores cannot be split. Ties select the smallest threshold.
        values,order=scores.sort();ordered=labels[order].long();cum=ordered.cumsum(0)
        ends=torch.cat((torch.nonzero(values[1:].ne(values[:-1])).flatten(),torch.tensor([len(values)-1])))
        candidates=torch.cat((values[:1]-1,values[ends]));k=torch.cat((torch.tensor([-1]),ends))
        false_negative=torch.cat((torch.tensor([0]),cum[ends]));above=len(values)-k-1
        false_positive=above-(int(labels.sum())-false_negative)
        errors=false_negative+false_positive;choice=int(errors.argmin());threshold=float(candidates[choice]);thresholds.append(threshold)
        relations.append(dict(relation=name,positive=len(positive),negative=len(negative),minimum_positive=float(positive.min()) if len(positive) else None,maximum_negative=float(negative.max()) if len(negative) else None,strictly_separable=bool(not len(positive) or not len(negative) or positive.min()>negative.max()),threshold=threshold,train_errors=int(errors[choice])))
    t=torch.tensor(thresholds);rows=[]
    for edge,truth,slots,gold,audit in cached:
        active=truth.any(-1);slotcorrect=bool(slots[active].eq(gold['slots'].flatten()[active]).all())
        raw=edge.gt(0);cal=edge.gt(t);correction=next(w['natural_logit_correction'] for w in weights if w['graph_seed']==audit['seed'])
        analytical=(edge+correction).gt(0)
        rows.append(dict(graph_seed=audit['seed'],raw_errors=int(raw.ne(truth).sum()),calibrated_errors=int(cal.ne(truth).sum()),analytical_oracle_errors=int(analytical.ne(truth).sum()),raw_exact=bool(raw.eq(truth).all()) and slotcorrect,calibrated_exact=bool(cal.eq(truth).all()) and slotcorrect,analytical_oracle_exact=bool(analytical.eq(truth).all()) and slotcorrect))
    runs.append(dict(seed=seed,relations=relations,graph_weighting=weights,graphs=rows))
print(json.dumps(dict(kind='posthoc TRAIN-only diagnostic on eight frozen development graphs; not fresh inference generalization or retroactive gate pass',policy='One threshold per relation minimizes total TRAIN pair classification errors; all score boundaries considered, ties choose lowest threshold. Same threshold used across eight graphs. No test labels exist in this diagnostic.',derivation='For weighted binary CE, optimal logit q = logit p + log(w_positive/w_negative). Here each graph uses w_positive=1/(2*N_positive), w_negative=1/(2*N_negative), over all node pairs and all relation entries. Thus logit p = q + log(N_positive/N_negative). Graphs are averaged equally. This population-risk identity is not a finite-trained-model calibration guarantee. Counts are privileged gold graph prevalence and unavailable at deployment; analytical correction is explicitly an oracle diagnostic. Stage8 sampled-pair weighting differs and cannot inherit this constant.',runs=runs),indent=2))
