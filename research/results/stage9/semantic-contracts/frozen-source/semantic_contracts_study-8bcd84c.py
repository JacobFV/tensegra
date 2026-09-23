"""Stage 9 privileged fixed-node head acquisition, separated from text learning."""
import argparse
import hashlib
import json
import time
import resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .semantic_contracts import SlotScorer,slot_objective,single_slot_labels
from . import semantic_scaling as base
from .thinking_language import ROLES,state_hash


class FixedNodeDecoder(nn.Module):
    def __init__(self,width=1024,*,interaction=False,rank=128,edge_affine=False):
        super().__init__();self.rank=rank
        self.source=nn.Linear(width,len(ROLES)*rank,bias=edge_affine)
        self.target=nn.Linear(width,rank,bias=edge_affine)
        self.slots=SlotScorer(width,base.MAX_SLOT+1,interaction=interaction,interaction_width=rank)
    def forward(self,nodes,pairs):
        i,j=pairs.unbind(-1)
        src=self.source(nodes)[i].reshape(-1,len(ROLES),self.rank)
        edge=(src*self.target(nodes)[j,None,:]).sum(-1)/self.rank**.5
        return edge,self.slots(nodes,pairs)


def fixtures(count,width,data_seed,device):
    examples=[];seen=set();offset=0;attempt=0
    while len(examples)<count:
        example=base.build_tcn_example(base.LESSONS[attempt%3],data_seed+attempt,difficulty=.5);attempt+=1
        graph=example.privileged.graph;key=base.semantic_key(graph)
        if key in seen:continue
        seen.add(key);single_slot_labels(graph);n=len(graph.nodes)
        if offset+n>width:raise ValueError('privileged unique node code capacity exhausted')
        nodes=torch.zeros(n,width,device=device);nodes[torch.arange(n),offset+torch.arange(n)]=1.;offset+=n
        vocab=base.value_vocabulary([example]);public,_=base.surface_input(example,'english')
        gold=base.targets(graph,public,n,vocab);pairs=torch.cartesian_prod(torch.arange(n),torch.arange(n)).to(device)
        examples.append((nodes,pairs,gold,example.audit))
    return examples


def evaluate(model,examples):
    rows=[]
    with torch.no_grad():
        for nodes,pairs,gold,audit in examples:
            edge,slots=model(nodes,pairs);pred=edge.gt(0).cpu().reshape_as(gold['edges']);slot=slots.argmax(-1).cpu().reshape_as(gold['slots'])-1
            truth=gold['edges']; active=truth.any(-1);correct_edges=torch.equal(pred,truth)
            correct_slots=bool(slot[active].eq(gold['slots'][active]).all())
            rows.append(dict(seed=audit['seed'],lesson=audit['lesson'],nodes=len(nodes),exact_graph=correct_edges and correct_slots,
               edge_errors=int(pred.ne(truth).sum()),slot_errors_on_gold_edges=int(slot[active].ne(gold['slots'][active]).sum()),
               predicted_edges=pred.nonzero().tolist(),predicted_slots=slot[active].tolist(),gold_edges=truth.nonzero().tolist(),gold_slots=gold['slots'][active].tolist(),union_pairs=(active|pred.any(-1)).nonzero().tolist(),predicted_slots_on_union=slot[active|pred.any(-1)].tolist()))
    return rows


def run(config):
    torch.set_num_threads(2);device=config.get('device','cuda');out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    examples=fixtures(config['graphs'],config['width'],config['data_seed'],device);runs=[]
    for seed in config['seeds']:
      for interaction in (False,True):
       for conditional in (False,True):
        torch.manual_seed(seed);model=FixedNodeDecoder(config['width'],interaction=interaction,rank=config['rank'],edge_affine=config.get('edge_affine',False)).to(device)
        initial_hash=state_hash(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);start=time.monotonic();curve=[];losses=[]
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for update in range(config['updates']+1):
            if update in config['checkpoints']:
                rows=evaluate(model,examples);curve.append(dict(update=update,exact_graph=sum(r['exact_graph'] for r in rows)/len(rows),rows=rows))
            if update==config['updates']:break
            optimizer.zero_grad();parts=[]
            for nodes,pairs,gold,_ in examples:
                edges,slots=model(nodes,pairs);truth=gold['edges'].to(device).flatten(0,1);ordered=gold['slots'].to(device).flatten()+1
                loss=F.binary_cross_entropy_with_logits(edges,truth.float(),reduction='none')
                edge_loss=.5*(loss[truth].mean()+loss[~truth].mean())
                parts.append(edge_loss+slot_objective(slots,ordered,truth.any(-1),edge_conditional=conditional))
            total=torch.stack(parts).mean();total.backward();optimizer.step();losses.append(float(total.detach()))
        if device=='cuda':torch.cuda.synchronize()
        checkpoint=out/f'head-{interaction}-conditional-{conditional}-seed{seed}.pt';torch.save(model.state_dict(),checkpoint)
        runs.append(dict(initial_state_sha256=initial_hash,final_state_sha256=state_hash(model),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),optimizer_presentations=config['updates']*len(examples),actual_unique_graphs=len(examples),losses=losses,process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seed=seed,head='interaction' if interaction else 'additive',objective='edge_conditional' if conditional else 'all_pairs',parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None,curve=curve))
        (out/'results.json').write_text(json.dumps(dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('semantic_contracts.py'))},graph_audits=[a for _,_,_,a in examples],runs=runs,priors=['Oracle graph-and-node identity one-hot padded to width1024','Node attributes, presence and visible copies supplied exactly; edge existence/relation/slot learned','Gold edges select loss entries only; final existence uses predicted logits']),indent=2)+'\n')
    return runs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
