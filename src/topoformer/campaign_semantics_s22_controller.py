"""Explicit privileged NODE boundaries for frozen S19 actors; never gold edges.

The public encoder is unchanged. Metadata is visible only to this controller.
Forced output records become PREVIOUS cache inputs on the following step, so no
current/future record is embedded early. Policy 'none' independently reproduces
unchanged greedy decoding, including strict failures and padding/stop behavior.
"""
from dataclasses import dataclass
import torch
from .campaign_semantics_s19_codec import BOS,NODE,EDGE,EOS,PAD,KINDS,CodecError,decode_records
from .thinking_language import ActorInput


@dataclass(frozen=True)
class NodeCount:
    count: int


@dataclass(frozen=True)
class NodeKinds:
    kinds: tuple[int,...]


@dataclass(frozen=True)
class NodePrefix:
    records: tuple[tuple[int,int,int,int,int],...]


POLICIES=('none','oracle_node_count','oracle_node_kinds','oracle_node_prefix')
COPY_KINDS=frozenset(KINDS.index(k) for k in ('ident','entity'))


def _boundaries(policy,boundaries,publics,model,token_counts):
    if policy not in POLICIES:raise ValueError('unknown S22 policy')
    if policy=='none':
        if boundaries is not None:raise TypeError('no-op policy accepts no privileged metadata')
        return [None]*len(publics),[None]*len(publics)
    expected={'oracle_node_count':NodeCount,'oracle_node_kinds':NodeKinds,'oracle_node_prefix':NodePrefix}[policy]
    if type(boundaries) not in (list,tuple) or len(boundaries)!=len(publics) or any(type(b) is not expected for b in boundaries):raise TypeError('exact policy-specific NODE boundary required')
    counts=[]
    for b,tokens in zip(boundaries,token_counts):
        if expected is NodeCount:count=b.count
        elif expected is NodeKinds:
            if type(b.kinds) is not tuple or any(type(k) is not int or not 0<=k<len(KINDS) for k in b.kinds):raise ValueError('invalid supplied kinds')
            count=len(b.kinds)
        else:
            if type(b.records) is not tuple:raise TypeError('immutable NODE prefix tuple required')
            count=len(b.records)
            for r in b.records:
                if type(r) is not tuple or len(r)!=5 or any(type(v) is not int for v in r):raise TypeError('NODE prefix requires integer5tuples')
                tag,kind,value,copy,unused=r
                if tag!=NODE or not 0<=kind<len(KINDS) or unused!=-1:raise ValueError('prefix may contain NODE records only')
                if kind in COPY_KINDS:
                    if value!=-1 or not 0<=copy<tokens:raise ValueError('invalid prefix public copy')
                elif copy!=-1 or not 0<=value<model.value_count:raise ValueError('invalid prefix value')
        if type(count) is not int or not 0<=count<=model.node_capacity or count>=model.max_records:raise ValueError('supplied node count exceeds supported capacity')
        counts.append(count)
    return list(boundaries),counts


@torch.no_grad()
def decode(model,publics,policy='none',boundaries=None):
    """Return unchanged greedy core plus explicit per-row privileged intervention stats.

    No source/target/role/slot/edge count can be supplied. For assisted policies,
    force NODE for exactly N steps; afterward exclude NODE while preserving the
    learned EDGE/EOS ranking. Fields/termination and strict codec remain unchanged.
    """
    if type(publics) not in (list,tuple) or not publics or any(type(p) is not ActorInput for p in publics):raise TypeError('public ActorInput sequence required')
    if policy not in POLICIES:raise ValueError('unknown S22 policy')
    was_training=model.training;model.eval()
    try:
        cache=model.begin(publics);batch=len(publics);token_counts=cache.public_mask.sum(1).cpu().tolist()
        boundaries,counts=_boundaries(policy,boundaries,publics,model,token_counts)
        previous=torch.full((batch,5),-1,dtype=torch.long,device=cache.memory.device);previous[:,0]=BOS
        records=[[] for _ in publics];status=['pending']*batch;nodes=[0]*batch;edges=[set() for _ in publics];edge_phase=[False]*batch
        stats=[dict(policy=policy,supplied_node_count=n,forced_node_tags=0,tag_replacements=0,forced_node_kinds=0,kind_replacements=0,supplied_prefix_records=0,prefix_value_replacements=0,prefix_copy_replacements=0,node_suppression_steps=0,node_suppression_replacements=0) for n in counts]
        for position in range(model.max_records):
            logits,cache=model._step(previous,cache,validate=False)
            keys=tuple(logits);chosen=torch.stack([logits[k].argmax(-1) for k in keys]+[logits['type'][:,1:].argmax(-1)],-1).cpu().tolist()
            choice={k:[r[j] for r in chosen] for j,k in enumerate(keys)};edge_or_eos=[r[-1]+EDGE for r in chosen];following=[]
            for i in range(batch):
                if status[i]!='pending':following.append([PAD,-1,-1,-1,-1]);continue
                tag=choice['type'][i]+1;raw_tag=tag;boundary=boundaries[i];s=stats[i]
                forced=policy!='none' and position<counts[i]
                if forced:
                    tag=NODE;s['forced_node_tags']+=1;s['tag_replacements']+=int(raw_tag!=NODE)
                elif policy!='none':
                    tag=edge_or_eos[i];s['node_suppression_steps']+=1;s['node_suppression_replacements']+=int(raw_tag==NODE)
                if tag==EOS:r=[EOS,-1,-1,-1,-1];status[i]='eos'
                elif tag==NODE:
                    kind=choice['kind'][i]
                    if forced and policy in ('oracle_node_kinds','oracle_node_prefix'):
                        supplied=boundary.kinds[position] if policy=='oracle_node_kinds' else boundary.records[position][1]
                        s['forced_node_kinds']+=1;s['kind_replacements']+=int(kind!=supplied);kind=supplied
                    copy=choice['copy'][i] if kind in COPY_KINDS else -1;value=-1 if kind in COPY_KINDS else choice['value'][i]
                    r=[NODE,kind,value,copy,-1]
                    if forced and policy=='oracle_node_prefix':
                        supplied=boundary.records[position];s['supplied_prefix_records']+=1
                        s['prefix_value_replacements']+=int(value!=supplied[2]);s['prefix_copy_replacements']+=int(copy!=supplied[3]);r=list(supplied)
                    if edge_phase[i]:status[i]='node_after_edge'
                    elif nodes[i]>=model.node_capacity:status[i]='node_capacity_overflow'
                    else:nodes[i]+=1
                else:
                    r=[EDGE,choice['source'][i],choice['target'][i],choice['role'][i],choice['slot'][i]-1];edge_phase[i]=True
                    if r[1]>=nodes[i] or r[2]>=nodes[i]:status[i]='invalid_generated_node_reference'
                    elif tuple(r) in edges[i]:status[i]='duplicate_edge'
                    else:edges[i].add(tuple(r))
                records[i].append(r);following.append(r if status[i]=='pending' else [PAD,-1,-1,-1,-1])
            if all(s!='pending' for s in status):break
            previous=torch.tensor(following,dtype=torch.long,device=cache.memory.device)
        status=['record_capacity_overflow' if s=='pending' else s for s in status]
        for i,s in enumerate(status):
            if s=='eos':
                try:decode_records(records[i],token_count=token_counts[i],vocab_size=model.value_count,capacity=model.node_capacity,max_records=model.max_records)
                except CodecError as exc:status[i]='malformed: '+str(exc)
        return [dict(records=r,status=s,node_count=n,controller_stats=t) for r,s,n,t in zip(records,status,nodes,stats)]
    finally:model.train(was_training)
