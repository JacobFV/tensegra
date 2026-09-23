"""Isolated progressive beliefs. Typed roles and observation IDs are public priors.

Only the oracle compares symbolic values exactly. Learned models receive nonce
features, never support labels. The protected ledger supplies idempotent insertion
and explicit retraction; learning supplies compatibility, not ledger semantics.
"""
import random
import torch
from torch import nn
from torch.nn import functional as F

ROLES = 4  # primitive, destination, ordered operand 0 and operand 1


def make_episodes(count, seed=0, candidates=8, key_dim=16, condition='clean'):
    if key_dim < 3 or candidates + 3 > 2**key_dim:
        raise ValueError('nonce capacity exceeded')
    if candidates < 2 or condition not in ('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty'):
        raise ValueError('invalid episode configuration')
    rng = random.Random(seed)
    episodes = []
    for episode_index in range(count):
        rng = random.Random(seed * 1000003 + episode_index)
        # New opaque identities every episode; no stable node-position mapping.
        keys = []
        while len(keys) < candidates + 3:
            k = tuple(rng.choice((-1.,1.)) for _ in range(key_dim))
            if k not in keys: keys.append(k)
        records = []
        while len(records) < candidates:
            a,b = rng.sample(range(2,min(len(keys),max(5,candidates//2+2))),2)
            r = (rng.randrange(2), rng.randrange(2), a,b)
            if r not in records: records.append(r)
        rng.shuffle(records)
        gold = rng.randrange(candidates)
        roles = list(range(ROLES))
        if condition == 'reorder': rng.shuffle(roles)
        events = [(i, 1, role, records[gold][role]) for i,role in enumerate(roles)]
        if condition in ('duplicate','long_duplicate'):
            repeats=2 if condition=='duplicate' else 8
            events=[event for event in events for _ in range(repeats)]
        if condition in ('contradiction','retract'):
            wrong = (records[gold][0]+1)%5
            events.append((4,1,0,wrong))
            if condition == 'retract': events.append((4,-1,0,wrong))
        if condition == 'partial': events = events[:2]
        if condition == 'empty': events = []
        # A no-op initial frame provides an explicit prior measurement.
        events = [(-1,0,0,0)] + events
        ledger, targets, supports = {}, [], []
        for identity,action,role,value in events:
            if action == 1 and identity not in ledger: ledger[identity] = (role,value)
            elif action == -1: ledger.pop(identity,None)
            support = [all(r[j] == v for j,v in ledger.values()) for r in records]
            n = sum(support)
            targets.append([float(s)/n if n else 0. for s in support]+[float(n==0)])
            supports.append(support)
        episodes.append(dict(keys=keys,records=records,events=events,posterior=targets,support=supports))
    return episodes


def collate(episodes):
    b,n,t = len(episodes),max(len(e['records']) for e in episodes),max(len(e['events']) for e in episodes)
    d = len(episodes[0]['keys'][0]); f = 5+d
    records = torch.zeros(b,n,ROLES,f); event = torch.zeros(b,t,f)
    roles = torch.zeros(b,t,dtype=torch.long); ids = roles.clone(); actions = roles.clone()
    valid = torch.zeros(b,n,dtype=torch.bool); frames = torch.zeros(b,t,dtype=torch.bool)
    target = torch.zeros(b,t,n+1); compatible = torch.zeros(b,t,n)
    def feature(ep,role,value):
        out = torch.zeros(f)
        if role == 0: out[value] = 1
        else: out[5:] = torch.tensor(ep['keys'][value])
        return out
    for i,ep in enumerate(episodes):
        m = len(ep['records']); valid[i,:m]=True
        for j,r in enumerate(ep['records']):
            for role,value in enumerate(r): records[i,j,role] = feature(ep,role,value)
        for j,(identity,action,role,value) in enumerate(ep['events']):
            frames[i,j]=True; roles[i,j]=role; ids[i,j]=identity; actions[i,j]=action
            if action: event[i,j]=feature(ep,role,value)
            target[i,j,:m]=torch.tensor(ep['posterior'][j][:-1]); target[i,j,-1]=ep['posterior'][j][-1]
            compatible[i,j,:m]=torch.tensor([float(r[role]==value) for r in ep['records']])
    return {'public':dict(records=records,event=event,roles=roles,ids=ids,actions=actions,valid=valid,frames=frames),
            'targets':dict(posterior=target,compatible=compatible)}


def observation_features(ids):
    """Stable public binary IDs; zero vector reserved for no-op ID -1."""
    if bool(((ids < -1) | (ids >= 65536)).any()):
        raise ValueError('observation IDs must be -1 or unsigned 16 bit')
    bits=((ids.clamp_min(0)[...,None] >> torch.arange(16,device=ids.device)) & 1).float()*2-1
    return bits * (ids>=0)[...,None]


class Phase(nn.Module):
    def __init__(self,width,inner):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,inner),nn.GELU(),nn.Linear(inner,width))
    def forward(self,x): return x+self.net(x)


class BeliefModel(nn.Module):
    def __init__(self,mode='protected',width=1024,key_dim=16,inner=2048,observation_id_features=False):
        super().__init__()
        if mode not in ('protected','recurrent'): raise ValueError(mode)
        self.mode=mode; self.width=width; self.observation_id_features=observation_id_features
        f=5+key_dim
        self.encode=nn.Linear(f*2+ROLES+3+(16 if observation_id_features else 0),width)
        self.phases=nn.ModuleList([Phase(width,inner) for _ in range(4)])
        self.compatibility=nn.Linear(width,1); self.readout=nn.Linear(width,1)
        self.null=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,public):
        p=public; records=p['records']; b,n,_,_=records.shape
        state=records.new_zeros(b,n,self.width); ledgers=[{} for _ in range(b)]
        scores=records.new_zeros(b,n); all_scores=[]; compat=[]
        ids=p['ids'].tolist(); actions=p['actions'].tolist(); frames=p['frames'].tolist()
        id_features=observation_features(p['ids']).to(records.dtype) if self.observation_id_features else None
        for t in range(p['event'].shape[1]):
            selected=records[torch.arange(b,device=records.device)[:,None],torch.arange(n,device=records.device)[None,:],p['roles'][:,t,None]]
            role=F.one_hot(p['roles'][:,t],ROLES).to(records.dtype)
            action=F.one_hot(p['actions'][:,t]+1,3).to(records.dtype)
            x=torch.cat((selected,p['event'][:,t,None].expand(-1,n,-1),role[:,None].expand(-1,n,-1),action[:,None].expand(-1,n,-1)),dim=-1)
            if id_features is not None: x=torch.cat((x,id_features[:,t,None].expand(-1,n,-1)),dim=-1)
            evidence=self.encode(x)
            local=evidence
            for phase in self.phases: local=phase(local)
            c=self.compatibility(local).squeeze(-1); compat.append(c)
            if self.mode=='protected':
                rows=[]
                for i in range(b):
                    identity=ids[i][t]; a=actions[i][t]
                    if frames[i][t]:
                        if a==1 and identity not in ledgers[i]: ledgers[i][identity]=F.logsigmoid(c[i])
                        elif a==-1: ledgers[i].pop(identity,None)
                    rows.append(sum(ledgers[i].values(),torch.zeros_like(scores[i])))
                scores=torch.stack(rows)
            else:
                next_state=state+evidence
                for phase in self.phases: next_state=phase(next_state)
                update=(p['frames'][:,t] & (p['actions'][:,t]!=0))[:,None,None]
                state=torch.where(update,next_state,state)
                scores=self.readout(state).squeeze(-1)
            masked=scores.masked_fill(~p['valid'],-1e9)
            null_features=torch.stack((masked.max(-1).values, scores.masked_fill(~p['valid'],0).sum(-1)/p['valid'].sum(-1)),dim=-1)
            null=self.null(null_features)
            all_scores.append(torch.cat((masked,null),dim=-1))
        return {'logits':torch.stack(all_scores,1),'compatibility':torch.stack(compat,1)}


def loss(output,batch):
    p,y=batch['public'],batch['targets']; logp=output['logits'].log_softmax(-1)
    posterior=-(y['posterior']*logp).sum(-1)[p['frames']].mean()
    mask=p['frames'][:,:,None]&p['valid'][:,None,:]&(p['actions'][:,:,None]==1)
    terms=F.binary_cross_entropy_with_logits(output['compatibility'],y['compatible'],reduction='none')
    compatibility=terms[mask].mean() if mask.any() else terms.sum()*0
    return {'posterior':posterior,'compatibility':compatibility}


def oracle(public):
    p=public; b,n,_,_=p['records'].shape; ledgers=[{} for _ in range(b)]; result=[]
    for t in range(p['event'].shape[1]):
        rows=[]
        for i in range(b):
            a=int(p['actions'][i,t]); identity=int(p['ids'][i,t]); role=int(p['roles'][i,t])
            if bool(p['frames'][i,t]):
                if a==1 and identity not in ledgers[i]: ledgers[i][identity]=(p['records'][i,:,role]==p['event'][i,t]).all(-1)
                elif a==-1: ledgers[i].pop(identity,None)
            support=p['valid'][i].clone()
            for s in ledgers[i].values(): support &= s
            count=int(support.sum()); rows.append(torch.cat((support.float()/max(count,1),support.new_tensor([count==0],dtype=torch.float))))
        result.append(torch.stack(rows))
    return torch.stack(result,1)
