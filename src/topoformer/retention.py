"""Isolated retention: existing four-block cell, immutable external records.

Protected-register exactness is supplied by the architecture. Every semantic
prediction is decoded from learned workspace; record fields never bypass it.
"""
import math
import torch
from torch import nn
from .thinking import ThinkingConfig, ThinkingModel

MODES=('once','protected','gated','persistent')
INTERVENTIONS=('none','event_drop','wrong_value','wrong_type','wrong_provenance',
               'delay_memory_drop','query_counterfactual','unrelated_activity','release','overwrite')


class ReturnRegister:
    def __init__(self,event):
        self.overwrite(event)

    def overwrite(self,event):
        self._record={k:v.clone() for k,v in event.items()}

    def release(self):
        self._record=None

    def read(self):
        return None if self._record is None else {k:v.clone() for k,v in self._record.items()}


class ReturnRetentionModel(nn.Module):
    def __init__(self,feature_dim=32,width=32,value_limit=8):
        super().__init__()
        self.core=ThinkingModel(ThinkingConfig(feature_dim=feature_dim,width=width,heads=4,structural_heads=0))
        self.query=nn.Linear(2,width)
        self.active_status=nn.Linear(1,width)
        self.read_gate=nn.Linear(width,1)
        self.norm=nn.LayerNorm(width)
        self.heads=nn.ModuleDict({k+'_head':nn.Linear(width,n) for k,n in dict(value=4*value_limit+1,type=3,operation=5,task=2,validity=2).items()})
        self.identity=nn.ModuleDict({k:nn.Linear(width,feature_dim) for k in ('argument0','argument1','provenance')})
        self.null_argument=nn.Linear(width,1)
        self.value_limit=value_limit

    def _cell(self,state,memory):
        b,s,_=state.shape
        mask=torch.ones(memory.shape[:2],dtype=torch.bool,device=state.device)
        bias=state.new_zeros(b,self.core.config.heads,s,s)
        for block in self.core.blocks:
            state,_=block(state,memory,mask,bias)
        return state

    def _decode(self,state,public):
        latent=self.norm(state.mean(1))
        result={k.removesuffix('_head'):head(latent) for k,head in self.heads.items()}
        for name,projection in self.identity.items():
            keys=public['provenance_keys'] if name=='provenance' else public['argument_keys']
            scores=torch.einsum('bf,bkf->bk',projection(latent),keys)/math.sqrt(keys.shape[-1])
            if name.startswith('argument'):
                scores=torch.cat((scores[:,:-1],self.null_argument(latent)),1)
            result[name]=scores
        return result

    def forward(self,public,delay,mode='protected',intervention='none',replacement_event=None):
        if mode not in MODES or intervention not in INTERVENTIONS or delay not in (1,2,4,8,16,32):
            raise ValueError('invalid retention condition')
        event={k:v.clone() for k,v in public['event'].items()}
        if intervention=='wrong_value':
            event['values']=torch.where(event['values']>0,-event['values'],torch.ones_like(event['values'])*self.value_limit)
        if intervention=='wrong_type': event['types']=(event['types']+1)%3
        if intervention=='wrong_provenance':
            keys=public['provenance_keys']
            current=(keys-event['provenance']).square().sum(-1).argmin(-1)
            event['provenance']=keys[torch.arange(keys.shape[0],device=keys.device),(current+1)%keys.shape[1]][:,None]
        register=ReturnRegister(event)
        durable=ReturnRegister(event)
        encoded=self.core.encode_events(event)
        b=encoded.shape[0]
        initial=self.core.initial_rows[None].expand(b,-1,-1)
        state=initial
        if intervention!='event_drop':
            routes=state.new_full((b,1,state.shape[1]),1/state.shape[1])
            state=self.core.inject_events(state,encoded,routes)
        else:
            register.release(); durable.release()
        start=state
        gates=[]
        for step in range(delay+1):
            # External lifecycle commands invalidate storage, never the free workspace.
            if step==delay//2:
                if intervention=='delay_memory_drop': register.release()
                if intervention=='release':
                    register.release(); durable.release()
                if intervention=='overwrite':
                    if replacement_event is None: raise ValueError('overwrite requires complete replacement event')
                    register.overwrite(replacement_event); durable.overwrite(replacement_event)
            if step==delay:
                before=self._decode(state,public)
            distractors=public['distractors'][:,step]
            if intervention=='unrelated_activity': distractors=-distractors
            memory=self.core.features(distractors)
            record=register.read()
            if record is None and mode=='persistent': record=durable.read()
            if mode!='once' and record is not None:
                read=self.core.encode_events(record)
                gate=self.read_gate(self.norm(state.mean(1))).sigmoid()[:,None]
                gates.append(gate.detach().mean())
                memory=torch.cat((memory,read*(gate if mode in ('gated','persistent') else 1)),1)
            status=state.new_full((b,1,1),float(register.read() is not None))
            memory=torch.cat((memory,self.active_status(status)),1)
            if step==delay:
                query=public['query'].clone()
                if intervention=='query_counterfactual': query[:,1]=1-query[:,1]
                memory=torch.cat((memory,self.query(query)[:,None]),1)
            state=self._cell(state,memory)
        return dict(before=before,after=self._decode(state,public),cell_passes=delay+1,
                    register_active=register.read() is not None,
                    read_gate_mean=torch.stack(gates).mean() if gates else state.new_tensor(0.),
                    drift=(state-start).square().mean().detach())
