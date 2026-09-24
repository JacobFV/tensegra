"""S19 fresh text encoder + causal typed-record decoder; no graph inputs.

Two distinct encoder and two distinct decoder layers, width1024/8heads/FFN4w,
dropout0. Public encode_text features plus fixed sinusoidal positions. Records
are [tag,a,b,c,d]: BOS0, NODE1(kind,value,copy,-1), EDGE2(src,dst,role,slot),
EOS3, PAD4; inactive fields=-1. Class logits use +1 for nullable value/copy/slot.
The codec owns graph reconstruction/semantic validity. This module supplies
exact public-token copying given a learned index, never a graph/term compiler.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .semantic_curriculum import encode_text
from .thinking_language import ActorInput,FEATURE_DIM,KINDS,ROLES

from .campaign_semantics_s19_codec import BOS,NODE,EDGE,EOS,PAD,decode_records,CodecError
RECORD_WIDTH=5


def positions(length,width,device,dtype,offset=0):
    if width%2:raise ValueError('even model width required')
    index=torch.arange(offset,offset+length,device=device,dtype=torch.float32)[:,None]
    frequency=torch.exp(torch.arange(0,width,2,device=device,dtype=torch.float32)*(-math.log(10000.)/width))
    phase=index*frequency
    return torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(-2).to(dtype)


class _Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.head_width=width//heads
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)
    def split(self,x):return x.reshape(x.shape[0],x.shape[1],self.heads,self.head_width).transpose(1,2)
    def kv(self,x):return self.split(self.k(x)),self.split(self.v(x))
    def attend(self,x,kv,mask):
        z=F.scaled_dot_product_attention(self.split(self.q(x)),kv[0],kv[1],attn_mask=mask,dropout_p=0.)
        return self.out(z.transpose(1,2).contiguous().flatten(-2))


class _EncoderLayer(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.attention=_Attention(width,heads);self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x,mask):
        norm=self.norm1(x);x=x+self.attention.attend(norm,self.attention.kv(norm),mask[:,None,None,:])
        return (x+self.ff(self.norm2(x))).masked_fill(~mask[:,:,None],0.)


class _DecoderLayer(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
        self.self_attention=_Attention(width,heads);self.cross_attention=_Attention(width,heads)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def full(self,x,key_mask,cross_kv,public_mask):
        norm=self.norm1(x);length=x.shape[1]
        causal=torch.ones(length,length,dtype=torch.bool,device=x.device).tril()
        x=x+self.self_attention.attend(norm,self.self_attention.kv(norm),key_mask[:,None,None,:]&causal[None,None])
        x=x+self.cross_attention.attend(self.norm2(x),cross_kv,public_mask[:,None,None,:])
        return (x+self.ff(self.norm3(x))).masked_fill(~key_mask[:,:,None],0.)
    def step(self,x,past,key_mask,cross_kv,public_mask):
        norm=self.norm1(x);current=self.self_attention.kv(norm)
        kv=current if past is None else tuple(torch.cat((a,b),dim=2) for a,b in zip(past,current))
        x=x+self.self_attention.attend(norm,kv,key_mask[:,None,None,:])
        x=x+self.cross_attention.attend(self.norm2(x),cross_kv,public_mask[:,None,None,:])
        x=(x+self.ff(self.norm3(x))).masked_fill(~key_mask[:,-1,None,None],0.)
        return x,kv


@dataclass(frozen=True)
class DecodeCache:
    memory: torch.Tensor
    public_mask: torch.Tensor
    cross_kv: tuple
    copy_keys: torch.Tensor
    copy_inputs: torch.Tensor
    self_kv: tuple
    record_mask: torch.Tensor
    position: int


class TypedRecordActor(nn.Module):
    def __init__(self,*,value_count,width=1024,heads=8,node_capacity=128,max_records=160,max_slot=32,autocast_dtype=None):
        super().__init__()
        if min(value_count,width,heads,node_capacity,max_records,max_slot)<=0 or width%heads or width%2:raise ValueError('invalid dimensions')
        if autocast_dtype not in (None,'bfloat16'):raise ValueError('only optional BF16 supported')
        self.width=width;self.node_capacity=node_capacity;self.max_records=max_records;self.max_slot=max_slot;self.value_count=value_count;self.autocast_dtype=autocast_dtype
        self.features=nn.Linear(FEATURE_DIM,width)
        self.encoder=nn.ModuleList([_EncoderLayer(width,heads) for _ in range(2)])
        self.encoder_norm=nn.LayerNorm(width)
        self.decoder=nn.ModuleList([_DecoderLayer(width,heads) for _ in range(2)])
        self.decoder_norm=nn.LayerNorm(width)
        self.tag_embedding=nn.Embedding(5,width,padding_idx=PAD)
        self.kind_embedding=nn.Embedding(len(KINDS)+1,width,padding_idx=0)
        self.value_embedding=nn.Embedding(value_count+1,width,padding_idx=0)
        self.source_embedding=nn.Embedding(node_capacity+1,width,padding_idx=0)
        self.target_embedding=nn.Embedding(node_capacity+1,width,padding_idx=0)
        self.role_embedding=nn.Embedding(len(ROLES)+1,width,padding_idx=0)
        self.slot_embedding=nn.Embedding(max_slot+1,width,padding_idx=0)
        self.copy_input=nn.Linear(width,width,bias=False)
        self.type_head=nn.Linear(width,3) # index0/1/2 means NODE/EDGE/EOS
        self.kind_head=nn.Linear(width,len(KINDS));self.value_head=nn.Linear(width,value_count+1)
        self.source_head=nn.Linear(width,node_capacity);self.target_head=nn.Linear(width,node_capacity)
        self.role_head=nn.Linear(width,len(ROLES));self.slot_head=nn.Linear(width,max_slot+1)
        self.copy_query=nn.Linear(width,width);self.copy_key=nn.Linear(width,width);self.copy_none=nn.Linear(width,1)
    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def autocast(self):
        return torch.autocast(self.features.weight.device.type,dtype=torch.bfloat16,enabled=self.autocast_dtype=='bfloat16')
    def _encode(self,publics):
        if not isinstance(publics,(list,tuple)) or not publics or any(not isinstance(p,ActorInput) for p in publics):raise TypeError('nonempty list of public ActorInput required')
        encoded=[]
        for public in publics:
            f,length=encode_text(public.text)
            if length<=0 or f.shape[1]!=length:raise ValueError('empty or inconsistent public token inventory')
            encoded.append(f[0])
        features=nn.utils.rnn.pad_sequence(encoded,batch_first=True).to(self.features.weight.device)
        lengths=torch.tensor([len(x) for x in encoded],device=features.device)
        mask=torch.arange(features.shape[1],device=features.device)[None,:]<lengths[:,None]
        x=self.features(features);x=(x+positions(x.shape[1],self.width,x.device,x.dtype)).masked_fill(~mask[:,:,None],0.)
        for layer in self.encoder:x=layer(x,mask)
        return self.encoder_norm(x).masked_fill(~mask[:,:,None],0.),mask
    def begin(self,publics):
        with self.autocast():
            memory,mask=self._encode(publics)
            return DecodeCache(memory,mask,tuple(l.cross_attention.kv(memory) for l in self.decoder),
                self.copy_key(memory),self.copy_input(memory),(None,None),torch.empty(len(publics),0,dtype=torch.bool,device=memory.device),0)
    def _embed(self,records,cache,offset=0,validate=True):
        if records.dtype!=torch.long or records.ndim!=3 or records.shape[0]!=cache.memory.shape[0] or records.shape[-1]!=5:raise ValueError('records must be int64 [batch,time,5]')
        if records.device!=cache.memory.device:raise ValueError('records and model must share device')
        if records.shape[1]==0 or offset+records.shape[1]>self.max_records:raise ValueError('record capacity overflow')
        tag,a,b,c,d=records.unbind(-1)
        if validate and ((tag<0)|(tag>PAD)).any():raise ValueError('invalid preceding record tag')
        node=tag.eq(NODE);edge=tag.eq(EDGE)
        fields=((a,node,len(KINDS)),(b,node,self.value_count),(a,edge,self.node_capacity),(b,edge,self.node_capacity),(c,edge,len(ROLES)),(d,edge,self.max_slot))
        for value,active,size in fields:
            if validate and ((value[active]<-1)|(value[active]>=size)).any():raise ValueError('preceding record field outside vocabulary')
        copy=c.masked_fill(~node,-1)
        if validate and ((copy<-1)|(copy>=cache.memory.shape[1])).any():raise ValueError('preceding copy index outside public inventory')
        selected=cache.copy_inputs.gather(1,copy.clamp_min(0)[:,:,None].expand(-1,-1,self.width))
        valid_copy=cache.public_mask.gather(1,copy.clamp_min(0))
        if validate and (copy.ge(0)&~valid_copy).any():raise ValueError('preceding copy points to padding')
        x=self.tag_embedding(tag)
        x=x+self.kind_embedding(a.masked_fill(~node,-1)+1)+self.value_embedding(b.masked_fill(~node,-1)+1)
        x=x+selected*copy.ge(0)[:,:,None]
        x=x+self.source_embedding(a.masked_fill(~edge,-1)+1)+self.target_embedding(b.masked_fill(~edge,-1)+1)
        x=x+self.role_embedding(c.masked_fill(~edge,-1)+1)+self.slot_embedding(d.masked_fill(~edge,-1)+1)
        return (x+positions(x.shape[1],self.width,x.device,x.dtype,offset)).masked_fill(tag.eq(PAD)[:,:,None],0.)
    def _heads(self,x,cache):
        x=self.decoder_norm(x)
        copy=self.copy_query(x)@cache.copy_keys.transpose(-1,-2)/math.sqrt(self.width)
        copy=copy.masked_fill(~cache.public_mask[:,None,:],-torch.inf)
        return {**{k:head(x).float() for k,head in (('type',self.type_head),('kind',self.kind_head),('value',self.value_head),('source',self.source_head),('target',self.target_head),('role',self.role_head),('slot',self.slot_head))},'copy':torch.cat((self.copy_none(x),copy),dim=-1).float()}
    def forward(self,publics,previous_records):
        """Logits after each supplied PREVIOUS record; first record must be BOS."""
        with self.autocast():
            cache=self.begin(publics);x=self._embed(previous_records,cache)
            if not previous_records[:,0,0].eq(BOS).all():raise ValueError('preceding sequence must start with BOS')
            mask=previous_records[:,:,0].ne(PAD)
            for layer,kv in zip(self.decoder,cache.cross_kv):x=layer.full(x,mask,kv,cache.public_mask)
            return self._heads(x,cache)
    def teacher_forced(self,publics,target_records):
        """Shift inside helper, ensuring output[t] cannot read target record[t]."""
        if target_records.ndim!=3 or target_records.shape[-1]!=5 or target_records.shape[1]==0:raise ValueError('target records require [batch,time,5]')
        bos=target_records.new_full((target_records.shape[0],1,5),-1);bos[:,:,0]=BOS
        return self(publics,torch.cat((bos,target_records[:,:-1]),dim=1))
    def step(self,previous_record,cache):
        return self._step(previous_record,cache,validate=True)
    def _step(self,previous_record,cache,validate):
        """Cached inference; one previous record [batch,5], returns next logits/cache."""
        if previous_record.ndim!=2:raise ValueError('one preceding record per batch item required')
        if validate and cache.position==0 and not previous_record[:,0].eq(BOS).all():raise ValueError('incremental decoding must begin with BOS')
        with self.autocast():
            x=self._embed(previous_record[:,None],cache,cache.position,validate)
            mask=torch.cat((cache.record_mask,previous_record[:,0,None].ne(PAD)),dim=1);updated=[]
            for layer,past,cross in zip(self.decoder,cache.self_kv,cache.cross_kv):
                x,kv=layer.step(x,past,mask,cross,cache.public_mask);updated.append(kv)
            logits={k:v[:,0] for k,v in self._heads(x,cache).items()}
            return logits,DecodeCache(cache.memory,cache.public_mask,cache.cross_kv,cache.copy_keys,cache.copy_inputs,tuple(updated),mask,cache.position+1)
    @torch.no_grad()
    def greedy(self,publics):
        """Public-only cached argmax. No gold counts, graph, schedule, or repair."""
        was_training=self.training;self.eval()
        try:
            cache=self.begin(publics);batch=len(publics);previous=torch.full((batch,5),-1,dtype=torch.long,device=cache.memory.device);previous[:,0]=BOS
            token_counts=cache.public_mask.sum(1).cpu().tolist()
            records=[[] for _ in publics];status=['pending']*batch;nodes=[0]*batch;edges=[set() for _ in publics];edge_phase=[False]*batch
            copying_kinds={KINDS.index('ident'),KINDS.index('entity')}
            for _ in range(self.max_records):
                logits,cache=self._step(previous,cache,validate=False)
                keys=tuple(logits);selected=torch.stack([logits[k].argmax(-1) for k in keys],-1).cpu().tolist()
                choice={k:[row[j] for row in selected] for j,k in enumerate(keys)}
                next_records=[]
                for i in range(batch):
                    r=[PAD,-1,-1,-1,-1]
                    if status[i]!='pending':next_records.append(r);continue
                    tag=choice['type'][i]+1
                    if tag==EOS:r=[EOS,-1,-1,-1,-1];status[i]='eos'
                    elif tag==NODE:
                        kind=choice['kind'][i];copy=choice['copy'][i]-1 if kind in copying_kinds else -1
                        value=-1 if kind in copying_kinds else choice['value'][i]-1;r=[NODE,kind,value,copy,-1]
                        if edge_phase[i]:status[i]='node_after_edge'
                        elif nodes[i]>=self.node_capacity:status[i]='node_capacity_overflow'
                        elif (kind in copying_kinds and copy<0) or (kind not in copying_kinds and value<0):status[i]='missing_node_value'
                        else:nodes[i]+=1
                    else:
                        r=[EDGE,choice['source'][i],choice['target'][i],choice['role'][i],choice['slot'][i]-1];edge_phase[i]=True
                        if r[1]>=nodes[i] or r[2]>=nodes[i]:status[i]='invalid_generated_node_reference'
                        elif tuple(r) in edges[i]:status[i]='duplicate_edge'
                        else:edges[i].add(tuple(r))
                    records[i].append(r);next_records.append(r if status[i]=='pending' else [PAD,-1,-1,-1,-1])
                if all(s!='pending' for s in status):break
                previous=torch.tensor(next_records,dtype=torch.long,device=cache.memory.device)
            status=['record_capacity_overflow' if s=='pending' else s for s in status]
            for i,s in enumerate(status):
                if s=='eos':
                    try:decode_records(records[i],token_count=token_counts[i],vocab_size=self.value_count,capacity=self.node_capacity,max_records=self.max_records)
                    except CodecError as exc:status[i]='malformed: '+str(exc)
            return [dict(records=r,status=s,node_count=n) for r,s,n in zip(records,status,nodes)]
        finally:self.train(was_training)
