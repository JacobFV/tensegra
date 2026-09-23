"""Stage 10 frozen semantic-edge failure localization; CPU, no training."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from .semantic_contracts_study import FixedNodeDecoder,fixtures
from .semantic_contracts import train_relation_thresholds
from .semantic_scaling import build_tcn_example
from .thinking_language import ROLES


def negative_gradient_direction(output,parameters,loss_gradients):
    """Directional derivative along negative full gradient, not an Adam step."""
    local_grad=torch.autograd.grad(output,parameters,retain_graph=True)
    return -sum(float((a*b).sum()) for a,b in zip(local_grad,loss_gradients))


def localize(directory,output):
    torch.set_num_threads(2);directory=Path(directory)
    archived=json.loads((directory/'results.json').read_text());config=archived['config']
    examples=fixtures(config['graphs'],config['width'],config['data_seed'],'cpu')
    path=directory/'head-False-conditional-True-seed10.pt'
    model=FixedNodeDecoder(config['width'],rank=config['rank'],edge_affine=True)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    cached=[];losses=[]
    for nodes,pairs,gold,audit in examples:
        logits,slots=model(nodes,pairs);truth=gold['edges'].flatten(0,1)
        element=F.binary_cross_entropy_with_logits(logits,truth.float(),reduction='none')
        losses.append(.5*(element[truth].mean()+element[~truth].mean()))
        cached.append((logits,truth,gold,audit))
    full_loss=torch.stack(losses).mean()
    parameters=list(model.source.parameters())+list(model.target.parameters())
    gradients=torch.autograd.grad(full_loss,parameters,retain_graph=True)
    total_grad_norm=float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
    all_scores=torch.cat([c[0].detach() for c in cached]);all_truth=torch.cat([c[1] for c in cached])
    thresholds,records=train_relation_thresholds(all_scores,all_truth)
    run=next(r for r in archived['runs'] if r['seed']==10 and r['head']=='additive' and r['objective']=='edge_conditional')
    failures=[]
    for index,(logits,truth,gold,audit) in enumerate(cached):
        graph=build_tcn_example(audit['lesson'],audit['seed'],difficulty=.5).privileged.graph
        n=len(graph.nodes);prediction=logits.detach().gt(thresholds)
        positive=int(truth.sum());negative=truth.numel()-positive
        for pair,relation in prediction.ne(truth).nonzero().tolist():
            source,target=divmod(pair,n);score=logits[pair,relation]
            directional=negative_gradient_direction(score,parameters,gradients)
            target_value=bool(truth[pair,relation]);weight=.5/(positive if target_value else negative)/len(cached)
            historical=[]
            for checkpoint in run['curve']:
                row=next(r for r in checkpoint['rows'] if r['seed']==audit['seed'])
                historical.append(dict(update=checkpoint['update'],raw_prediction=[source,target,relation] in row['predicted_edges']))
            failures.append(dict(graph_seed=audit['seed'],lesson=audit['lesson'],source_index=source,target_index=target,source_node=graph.nodes[source].__dict__,target_node=graph.nodes[target].__dict__,relation=ROLES[relation],gold=target_value,logit=float(score.detach()),threshold=float(thresholds[relation]),signed_threshold_margin=float(score.detach()-thresholds[relation]),weighted_loss_logit_gradient=float((score.detach().sigmoid()-float(target_value))*weight),negative_full_gradient_logit_direction=directional,raw_prediction_trajectory=historical))
    relation_rows=[]
    for r,name in enumerate(ROLES):
        pos=all_scores[all_truth[:,r],r];neg=all_scores[~all_truth[:,r],r]
        relation_rows.append(dict(relation=name,minimum_positive=float(pos.min()) if len(pos) else None,maximum_negative=float(neg.max()) if len(neg) else None,ranking_gap=float(pos.min()-neg.max()) if len(pos) and len(neg) else None,**records[r]))
    result=dict(scope='Frozen historical seed10 additive/edge-conditional reference only; no optimizer steps',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=config,edge_loss=float(full_loss.detach()),edge_gradient_norm=total_grad_norm,failures=failures,relations=relation_rows,limitations=['The full-gradient direction is an infinitesimal SGD diagnostic, not the actual Adam update or proof of future convergence.','Only raw binary predictions and scalar losses survive at earlier updates; intermediate logits/gradients/checkpoints are unavailable.','All gradient labels are privileged known TRAIN graph targets; no fresh-data or semantic-transfer claim.'])
    Path(output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('output');a=p.parse_args();localize(a.directory,a.output)
