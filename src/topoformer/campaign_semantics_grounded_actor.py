"""S09: original workspace plus a copy-conditioned contextual text read.

Public-only inference. Copy logits serve as learned retrieval weights; supervision
never determines forward node positions, adjacency, types, or pointer choices.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .semantic_curriculum import SemanticCurriculumActor,encode_text
from .thinking_language import ActorInput,ROLES


class CopyConditionedActor(SemanticCurriculumActor):
    def __init__(self,*,context_read=True,read_scale=.5,**kwargs):
        super().__init__(**kwargs)
        self.context_read=bool(context_read);self.read_scale=float(read_scale)
        if not 0<=self.read_scale<=1:raise ValueError('read scale must be bounded')

    def _forward_batch(self,publics,*,pairs=None):
        if not self.context_read:return super()._forward_batch(publics,pairs=pairs)
        if not publics or any(not isinstance(p,ActorInput) for p in publics):raise TypeError('public ActorInput batch required')
        encoded=[];lengths=[]
        for public in publics:
            features,length=encode_text(public.text);lengths.append(length)
            encoded.append(torch.zeros_like(features[0,:1]) if self.no_input else features[0])
        features=nn.utils.rnn.pad_sequence(encoded,batch_first=True).to(self.initial.device)
        mask=torch.arange(features.shape[1],device=features.device)[None,:]<torch.tensor([len(e) for e in encoded],device=features.device)[:,None]
        memory=self.features(features);workspace=self.initial.expand(len(publics),-1,-1)
        context=memory.float();bias=memory.new_zeros((len(publics),1,1,memory.shape[1])).masked_fill(~mask[:,None,None,:],-torch.inf)
        for _ in range(self.microsteps):
            for block in self.blocks:
                workspace,_=block(workspace,memory,mask,0.)
                context,_=block(context,memory,mask,bias)
                context=context.masked_fill(~mask[:,:,None],0.)
        queries=self.queries.expand(len(publics),-1,-1)
        base=self.decode_nodes(queries,workspace,workspace,need_weights=False)[0]+queries
        copy=self.copy_query(base)@self.copy_key(features).transpose(-1,-2)/math.sqrt(self.width)
        pointer=copy.masked_fill(~mask[:,None,:],-torch.inf).softmax(-1)
        # Reuse the existing decoder value/output maps: no new parameters or
        # unrelated read address. Output bias is already present in base.
        values=F.linear(context,self.decode_nodes.in_proj_weight[2*self.width:],self.decode_nodes.in_proj_bias[2*self.width:])
        read=F.linear(pointer@values,self.decode_nodes.out_proj.weight,None)
        nodes=base+self.read_scale*read
        source=self.edge_source(nodes).reshape(len(publics),self.capacity,len(ROLES),self.edge_width);target=self.edge_target(nodes)
        presence=self.presence(nodes)[...,0];kind=self.kind(nodes);value=self.value(nodes)
        ss=self.slot_source(nodes);st=self.slot_target(nodes);outputs=[]
        for row in range(len(publics)):
            if pairs is None:
                edge=torch.einsum('nrd,md->nmr',source[row],target[row])/math.sqrt(self.edge_width);slots=ss[row,:,None,:]+st[row,None,:,:]
            else:
                i,j=pairs[row].to(nodes.device).unbind(-1);edge=(source[row,i]*target[row,j,None,:]).sum(-1)/math.sqrt(self.edge_width);slots=ss[row,i]+st[row,j]
            outputs.append(dict(presence=presence[row],kind=kind[row],value=value[row],copy=copy[row,:,:lengths[row]],edges=edge,slots=slots))
        return outputs
