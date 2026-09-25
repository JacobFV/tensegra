"""A07 all-edge record attention; no query-dependent graph mask or gather."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .campaign_attention_selector import SelectorModel


@dataclass
class EdgeRecords:
    features: torch.Tensor  # source key, target attribute, relation one-hot
    destination_keys: torch.Tensor


def tokenize(batch, order=None):
    """Expand ALL public edges. Never inspect instructions, starts or values."""
    b, relations, n, _ = batch.adjacency.shape
    edges = batch.adjacency.bool().nonzero().reshape(b, relations*n*batch.degree, 4)
    bi, r, source, destination = edges.unbind(-1)
    feature = torch.cat((batch.keys[bi, source], batch.attributes[bi, destination],
                         F.one_hot(r, relations).float()), -1)
    keys = batch.keys[bi, destination]
    if order is not None:
        feature = feature.gather(1, order[..., None].expand(-1,-1,feature.shape[-1]))
        keys = keys.gather(1, order[..., None].expand(-1,-1,keys.shape[-1]))
    return EdgeRecords(feature, keys)


class RecordAttention(SelectorModel):
    """Two dense reads per supplied reverse step: edge record then node payload.

    Relation/source feature coordinates are identity-initialized, trainable.
    Attribute compatibility starts from independent random projections. All-edge
    attention is unmasked. The destination identity is a soft value read, never
    an exact selected edge or hidden neighborhood lookup.
    """
    def __init__(self, width=1024, key_dim=64, classes=16, heads=8, strength=8.):
        super().__init__(width,key_dim,classes,heads,strength)
        self.record_q=nn.Linear(2*key_dim+3,width,bias=False)
        self.record_k=nn.Linear(2*key_dim+3,width,bias=False)
        self.key_dim=key_dim
        self.record_log_scale=nn.Parameter(torch.tensor(0.))
        d=width//heads
        if d < key_dim+4: raise ValueError('Head width must fit explicit identity and relation roles')
        with torch.no_grad():
            for layer in (self.record_q,self.record_k):
                layer.weight.zero_()
                for head in range(heads):
                    block=layer.weight[head*d:(head+1)*d]
                    amplitude=math.sqrt(8*math.sqrt(d))
                    block[:key_dim,:key_dim]=torch.eye(key_dim)*amplitude
                    block[key_dim:key_dim+3,2*key_dim:]=torch.eye(3)*amplitude
                    nn.init.normal_(block[key_dim+3:,key_dim:2*key_dim],std=amplitude/math.sqrt(d-key_dim-3))

    def forward(self,batch,mode='records',*,records=None,zero_content=False,
                zero_strength=False,selector_scale_override=None,context_scale_override=None):
        # zero_strength is inapplicable: this arm receives no adjacency bias.
        if mode != 'records': raise ValueError(mode)
        if records is None:
            count=3*batch.values.shape[1]*batch.degree
            order=torch.rand(len(batch.values),count,device=batch.values.device).argsort(-1)
            records=tokenize(batch,order)
        b,n,_=batch.keys.shape;w=self.embed.embedding_dim;hd=w//self.heads
        h=self.embed(batch.values)
        feature=records.features
        if zero_content:
            feature=feature.clone();feature[...,self.key_dim:2*self.key_dim]=0
        ek=self.record_k(feature).reshape(b,-1,self.heads,hd).transpose(1,2)
        nk=F.normalize(self.context_k(batch.keys),dim=-1)
        logits=[];routes=[];weights_out=[];masses=[]
        bi=torch.arange(b,device=h.device)
        for t in reversed(range(batch.relations.shape[1])):
            rel=F.one_hot(batch.relations[:,t],3).float()[:,None].expand(-1,n,-1)
            desired=batch.instructions[:,t,None].expand(-1,n,-1)
            q=self.record_q(torch.cat((batch.keys,desired,rel),-1)).reshape(b,n,self.heads,hd).transpose(1,2)
            scale=self.record_log_scale.exp().clamp(max=8) if selector_scale_override is None else float(selector_scale_override)/8
            edge_weight=(q@ek.transpose(-1,-2)/math.sqrt(hd)*scale).softmax(-1)
            returned=edge_weight@records.destination_keys[:,None]
            address=F.normalize(self.context_q(returned),dim=-1)
            address_scale=self.context_log_scale.exp().clamp(max=64) if context_scale_override is None else float(context_scale_override)
            weights=(address@nk[:,None].transpose(-1,-2)*address_scale).softmax(-1)
            v=h.reshape(b,n,self.heads,hd).transpose(1,2)
            retrieved=(weights@v).transpose(1,2).reshape(b,n,w)
            h=retrieved+.1*self.update(retrieved)
            logits.append(self.readout(self.norm(h)))
            mean=weights.mean(1);routes.append(mean.argmax(-1));weights_out.append(mean)
            # Adjacency is used only for this posthoc metric, never routing.
            masses.append((mean*batch.adjacency[bi,batch.relations[:,t]]).sum(-1))
        return dict(logits=torch.stack(logits,1),routes=torch.stack(routes,1),
                    weights=torch.stack(weights_out,1),edge_mass=torch.stack(masses,1))
