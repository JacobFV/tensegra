"""S06 contextual public-token state path; unchanged graph prediction heads."""
import math
import torch
from torch import nn
from .semantic_curriculum import SemanticCurriculumActor,encode_text
from .thinking_language import ActorInput,ROLES


class ContextualTokenActor(SemanticCurriculumActor):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Retain shared initialization/state-dict structure, but this path does not read learned workspace rows.
        self.initial.requires_grad_(False)

    def _forward_batch(self, publics, *, pairs=None):
        if not publics or any(not isinstance(p,ActorInput) for p in publics):
            raise TypeError('nonempty public ActorInput batch required')
        encoded=[]; lengths=[]
        for public in publics:
            features,length=encode_text(public.text)
            lengths.append(length)
            encoded.append(torch.zeros_like(features[0,:1]) if self.no_input else features[0])
        features=nn.utils.rnn.pad_sequence(encoded,batch_first=True).to(self.initial.device)
        mask=torch.arange(features.shape[1],device=features.device)[None,:]<torch.tensor([len(e) for e in encoded],device=features.device)[:,None]
        memory=self.features(features)
        state=memory.float()
        self_bias=torch.zeros((len(publics),1,1,memory.shape[1]),device=memory.device,dtype=memory.dtype).masked_fill(~mask[:,None,None,:],-torch.inf)
        for _ in range(self.microsteps):
            for block in self.blocks:
                state,_=block(state,memory,mask,self_bias)
                state=state.masked_fill(~mask[:,:,None],0.)
        queries=self.queries.expand(len(publics),-1,-1)
        nodes=self.decode_nodes(queries,state,state,key_padding_mask=~mask,need_weights=False)[0]+queries
        source=self.edge_source(nodes).reshape(len(publics),self.capacity,len(ROLES),self.edge_width)
        target=self.edge_target(nodes)
        copy=self.copy_query(nodes)@self.copy_key(features).transpose(-1,-2)/math.sqrt(self.width)
        presence=self.presence(nodes)[...,0]; kind=self.kind(nodes); value=self.value(nodes)
        slot_source=self.slot_source(nodes); slot_target=self.slot_target(nodes)
        outputs=[]
        for row in range(len(publics)):
            if pairs is None:
                edge=torch.einsum('nrd,md->nmr',source[row],target[row])/math.sqrt(self.edge_width)
                slots=slot_source[row,:,None,:]+slot_target[row,None,:,:]
            else:
                i,j=pairs[row].to(nodes.device).unbind(-1)
                edge=(source[row,i]*target[row,j,None,:]).sum(-1)/math.sqrt(self.edge_width)
                slots=slot_source[row,i]+slot_target[row,j]
            copying=copy[row,:,:lengths[row]]
            if self.no_input: copying=copy[row].expand(-1,lengths[row])
            outputs.append(dict(presence=presence[row],kind=kind[row],value=value[row],copy=copying,edges=edge,slots=slots))
        return outputs

