"""A04: learned content selection among explicitly grounded typed neighbors.

The public reverse instruction schedule is supplied. Exact targets never enter
forward inference. Attribute codes repeat globally but match one local neighbor.
"""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from .campaign_attention import RoutingModel, restore_node_order


@dataclass
class SelectorBatch:
    keys: torch.Tensor
    values: torch.Tensor
    adjacency: torch.Tensor
    relations: torch.Tensor
    starts: torch.Tensor
    attributes: torch.Tensor
    instructions: torch.Tensor
    degree: int


def generate(batch, nodes, depth, *, seed, groups=4, device='cpu', key_dim=64,
             classes=16, train=False, heldout_composition=False, balanced=False):
    if nodes % groups or nodes <= groups:
        raise ValueError('Each attribute must label multiple nodes; N must be divisible by K')
    g=torch.Generator().manual_seed(seed)
    order=torch.rand(batch,nodes,generator=g).argsort(-1)
    members=order.reshape(batch,groups,nodes//groups)
    codes=F.normalize(torch.randn(batch,groups,key_dim,generator=g),dim=-1)
    attributes=torch.empty(batch,nodes,key_dim)
    attributes.scatter_(1,members.reshape(batch,nodes,1).expand(-1,-1,key_dim),
                        codes[:,:,None,:].expand(-1,-1,nodes//groups,-1).reshape(batch,nodes,key_dim))
    if balanced:
        # Every relation/source-group/target-group block is a bijection.
        # A public instruction collapses N starts to N/K once, then preserves
        # that cardinality under arbitrary subsequent supplied instructions.
        permutations=torch.rand(batch,3,groups,groups,nodes//groups,generator=g).argsort(-1)
        destinations=members[:,None,None,:,:].expand(-1,3,groups,-1,-1).gather(-1,permutations)
        candidates=destinations.permute(0,1,2,4,3).reshape(batch,3,nodes,groups)
        ordered=F.one_hot(candidates,nodes).sum(-2).float()
        adjacency=torch.empty_like(ordered).scatter_(2,members.reshape(batch,nodes)[:,None,:,None].expand(-1,3,-1,nodes),ordered)
    else:
        choices=torch.randint(nodes//groups,(batch,3,nodes,groups),generator=g)
        candidates=members[:,None,None,:,:].expand(-1,3,nodes,-1,-1).gather(-1,choices[...,None]).squeeze(-1)
        adjacency=F.one_hot(candidates,nodes).sum(-2).float()
    relations=torch.randint(3,(batch,depth),generator=g)
    if train:
        for t in range(1,depth):
            relations[(relations[:,t-1]==2)&(relations[:,t]==2),t]=0
    if heldout_composition and depth>=2:relations[:,:2]=2
    desired=torch.randint(groups,(batch,depth),generator=g)
    instructions=codes.gather(1,desired[...,None].expand(-1,-1,key_dim))
    keys=F.normalize(torch.randn(batch,nodes,key_dim,generator=g),dim=-1)
    values=torch.randint(classes,(batch,nodes),generator=g)
    starts=torch.randint(nodes,(batch,),generator=g)
    return SelectorBatch(*(x.to(device) for x in (keys,values,adjacency,relations,starts,attributes,instructions)),groups)


def oracle_successors(batch):
    """Exact equality-like reference from public codes and adjacency only."""
    bi=torch.arange(len(batch.values),device=batch.values.device)
    result=[]
    for t in range(batch.relations.shape[1]):
        a=batch.adjacency[bi,batch.relations[:,t]]
        similarity=(batch.instructions[:,t,None,:]*batch.attributes).sum(-1)
        result.append(similarity[:,None,:].expand_as(a).masked_fill(~a.bool(),-torch.inf).argmax(-1))
    return torch.stack(result,1)


def targets(batch):
    successor=oracle_successors(batch)
    current=batch.values;out=[]
    for t in reversed(range(successor.shape[1])):
        current=current.gather(1,successor[:,t]);out.append(current)
    return torch.stack(out,1)


def permute_nodes(batch,order):
    b,n=order.shape;bi=torch.arange(b,device=order.device)[:,None];inv=order.argsort(-1)
    a=batch.adjacency.gather(2,order[:,None,:,None].expand(-1,3,-1,n)).gather(3,order[:,None,None,:].expand(-1,3,n,-1))
    return replace(batch,keys=batch.keys[bi,order],values=batch.values[bi,order],attributes=batch.attributes[bi,order],adjacency=a,starts=inv.gather(1,batch.starts[:,None]).squeeze(1))


def swap_instruction(batch):
    """Public counterfactual: replace first desired code by a distinct node code."""
    instruction=batch.instructions.clone()
    different=(batch.attributes-instruction[:,0,None]).square().sum(-1)>.01
    idx=different.int().argmax(-1)
    instruction[:,0]=batch.attributes[torch.arange(len(idx),device=idx.device),idx]
    return replace(batch,instructions=instruction)


def corrupt(batch,kind,seed):
    """A05 regular-degree controls; caller retains clean targets.

    Source-row shuffling preserves both degree sequences and local code coverage.
    Correspondence conjugation permutes only nodes with the same public code.
    """
    a=batch.adjacency;b,relations,n,_=a.shape
    g=torch.Generator(device=a.device).manual_seed(seed)
    if kind=='wrong':
        order=torch.rand(b,relations,n,generator=g,device=a.device).argsort(-1)
        changed=a.gather(2,order[...,None].expand(-1,-1,-1,n))
    elif kind=='identity':
        # Grouping uses complete public vectors, not privileged group labels.
        _,group=torch.unique(batch.attributes.reshape(b*n,-1),dim=0,return_inverse=True)
        members=group.reshape(b,n).argsort(-1).reshape(b,batch.degree,n//batch.degree)
        order_inside=torch.rand(members.shape,generator=g,device=a.device).argsort(-1)
        shuffled=members.gather(-1,order_inside)
        order=torch.empty(b,n,dtype=torch.long,device=a.device)
        order.scatter_(1,members.reshape(b,n),shuffled.reshape(b,n))
        changed=a.gather(2,order[:,None,:,None].expand(-1,relations,-1,n)).gather(3,order[:,None,None,:].expand(-1,relations,n,-1))
    elif kind=='clean':return batch
    else:raise ValueError(kind)
    return replace(batch,adjacency=changed)


class SelectorModel(RoutingModel):
    def __init__(self,width=1024,key_dim=64,classes=16,heads=8,strength=8.):
        super().__init__(width,key_dim,classes,heads,strength)
        self.attribute_encoder=nn.Linear(key_dim,width,bias=False)
        self.selector_log_scale=nn.Parameter(torch.tensor(math.log(8.)))

    def forward(self,batch,mode='soft',*,zero_strength=False,zero_content=False,selector_scale_override=None,context_scale_override=None):
        if mode not in {'soft','hard','context','none'}:raise ValueError(mode)
        h=self.embed(batch.values);b,n,w=h.shape;bi=torch.arange(b,device=h.device)
        attributes=batch.attributes
        address_reads=None
        if mode=='context':
            q=F.normalize(self.context_q(batch.keys),dim=-1);k=F.normalize(self.context_k(batch.keys),dim=-1)
            address_scale=self.context_log_scale.exp().clamp(max=64) if context_scale_override is None else float(context_scale_override)
            address_reads=(q@k.transpose(-1,-2)*address_scale).softmax(-1)
            attributes=address_reads@attributes
        z=self.attribute_encoder(attributes)
        k=F.normalize(self.k(z).reshape(b,n,self.heads,w//self.heads).transpose(1,2),dim=-1)
        logits=[];routes=[];weights_out=[];masses=[]
        for t in reversed(range(batch.relations.shape[1])):
            relation=batch.relations[:,t];a=batch.adjacency[bi,relation]
            q=self.q(self.attribute_encoder(batch.instructions[:,t])).reshape(b,self.heads,1,w//self.heads)
            q=F.normalize(q,dim=-1)
            scale=self.selector_log_scale.exp().clamp(max=64) if selector_scale_override is None else float(selector_scale_override)
            score=(q@k.transpose(-1,-2))*scale
            score=score.expand(-1,-1,n,-1)
            if zero_content:score=torch.zeros_like(score)
            if mode=='soft' and not zero_strength:score=score+self.strength[relation,:,None,None]*a[:,None]
            if mode=='hard':
                # Exact address gather is a supplied retrieval interface. The
                # learned selector still chooses among the gathered neighbors.
                degree=batch.degree
                neighbor=a.topk(degree,dim=-1).indices[:,None].expand(-1,self.heads,-1,-1)
                local=score.gather(-1,neighbor).softmax(-1)
                weights=torch.zeros_like(score).scatter(-1,neighbor,local)
            else:
                if mode=='context':score=score.masked_fill(~a[:,None].bool(),-torch.inf)
                weights=score.softmax(-1)
            if address_reads is not None:weights=weights@address_reads[:,None]
            v=h.reshape(b,n,self.heads,w//self.heads).transpose(1,2)
            retrieved=(weights@v).transpose(1,2).reshape(b,n,w)
            h=retrieved+.1*self.update(retrieved)
            logits.append(self.readout(self.norm(h)));mean=weights.mean(1)
            routes.append(mean.argmax(-1));weights_out.append(mean);masses.append((mean*a).sum(-1))
        return dict(logits=torch.stack(logits,1),routes=torch.stack(routes,1),weights=torch.stack(weights_out,1),edge_mass=torch.stack(masses,1))


def metrics(output,gold,batch):
    pred=output['logits'].argmax(-1);correct=pred==gold;b,d,n=correct.shape
    bi=torch.arange(b,device=pred.device);actual=batch.starts.clone();proposed=actual.clone();path=torch.ones(b,dtype=torch.bool,device=pred.device)
    successor=oracle_successors(batch)
    for t in range(d):
        proposed=output['routes'][bi,d-1-t,proposed];actual=successor[bi,t,actual];path&=proposed==actual
    selected=successor.flip(1)
    selected_mass=output['weights'].gather(-1,selected[...,None]).squeeze(-1)
    return dict(task=correct[:,-1].gather(1,batch.starts[:,None]).squeeze(1),all_node=correct.float().mean((1,2)),
                suffix_value_trajectory=correct.all((1,2)),exact_pointer_path=path,
                selected_mass=selected_mass.mean((1,2)),
                edge_mass=(output['weights']*batch.adjacency[bi[:,None],batch.relations.flip(1)]).sum(-1).mean((1,2)),
                supplied_edge_mass=output['edge_mass'].mean((1,2)))
