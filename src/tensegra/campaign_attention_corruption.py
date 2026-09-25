"""A08 joint missing/spurious edge replacement, preserving regular degrees."""
from dataclasses import replace
import torch
from torch.nn import functional as F


def replace_edges(batch,fraction,seed):
    if not 0<=fraction<=1: raise ValueError(fraction)
    if fraction==0:return batch
    a=batch.adjacency;b,r,n,_=a.shape;k=batch.degree;m=n//k
    # Attribute equality is public; no hidden generator group IDs are consumed.
    _,group=torch.unique(batch.attributes.reshape(b*n,-1),dim=0,return_inverse=True)
    members=group.reshape(b,n).argsort(-1).reshape(b,k,m)
    order=members.reshape(b,n)
    ordered=a.gather(2,order[:,None,:,None].expand(-1,r,-1,n)).gather(3,order[:,None,None,:].expand(-1,r,n,-1))
    blocks=ordered.reshape(b,r,k,m,k,m).permute(0,1,2,4,3,5)
    # This version explicitly requires the audited v2 bijection family.
    if not torch.all(blocks.sum(-1)==1) or not torch.all(blocks.sum(-2)==1):
        raise ValueError('Replacement requires one bijection per public attribute block')
    destination=blocks.argmax(-1)
    count=min(m,max(2,round(m*fraction)))
    g=torch.Generator(device=a.device).manual_seed(seed)
    selected=torch.rand(destination.shape,generator=g,device=a.device).argsort(-1)[...,:count]
    old=destination.gather(-1,selected)
    altered=destination.scatter(-1,selected,old.roll(1,-1))
    changed=F.one_hot(altered,m).permute(0,1,2,4,3,5).reshape(b,r,n,n).to(a.dtype)
    inverse=order.argsort(-1)
    changed=changed.gather(2,inverse[:,None,:,None].expand(-1,r,-1,n)).gather(3,inverse[:,None,None,:].expand(-1,r,n,-1))
    return replace(batch,adjacency=changed)
