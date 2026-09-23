"""Stage9 public empty-ledger contract intervention, separate from Stage8 source.

This is an inference-only architectural prior. Learned candidate/null updates are
unchanged whenever any observation remains active. No gold support is consumed.
"""
import copy
import random
import torch
from torch import nn
from .belief_state import make_episodes

ORIGINAL_CONDITIONS=('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty')
NEW_CONDITIONS=('full_retract','distinct_equal','candidate_permutation','id_rename')

def empty_ledger_frames(public):
    ids=public['ids'].tolist(); actions=public['actions'].tolist(); frames=public['frames'].tolist()
    masks=[]
    for row_ids,row_actions,row_frames in zip(ids,actions,frames):
        active=set(); row=[]
        for identity,action,valid in zip(row_ids,row_actions,row_frames):
            if valid:
                if action==1: active.add(identity)
                elif action==-1: active.discard(identity)
            row.append(not active)
        masks.append(row)
    return torch.tensor(masks,device=public['ids'].device,dtype=torch.bool)

class PriorContract(nn.Module):
    def __init__(self,base,exact_empty=False):
        super().__init__(); self.base=base; self.exact_empty=exact_empty
    def forward(self,public):
        output=self.base(public)
        if not self.exact_empty:return output
        # Public set bookkeeping is supplied equally for both comparators.
        prior=torch.zeros_like(output['logits'])
        prior[:,:,:-1]=prior[:,:,:-1].masked_fill(~public['valid'][:,None,:],-torch.inf)
        prior[:,:,-1]=-torch.inf
        return {**output,'logits':torch.where(empty_ledger_frames(public)[:,:,None],prior,output['logits'])}

def contract_episodes(count,seed,candidates=8,condition='clean'):
    if condition in ORIGINAL_CONDITIONS:return make_episodes(count,seed,candidates,condition=condition)
    if condition not in NEW_CONDITIONS:raise ValueError(condition)
    episodes=make_episodes(count,seed,candidates)
    for idx,ep in enumerate(episodes):
        if condition=='full_retract':
            ep['events'] += [(i,-1,r,v) for i,a,r,v in reversed(ep['events'][1:])]
        elif condition=='distinct_equal':
            ep['events']=[ep['events'][0]]+[x for i,a,r,v in ep['events'][1:] for x in [(i,a,r,v),(i+100,a,r,v)]]
        elif condition=='candidate_permutation':
            random.Random(seed+idx+987).shuffle(ep['records'])
        else:
            ids=random.Random(seed+idx+654).sample(range(256,65536),4)
            ep['events']=[(ids[i] if i>=0 else i,a,r,v) for i,a,r,v in ep['events']]
        ledger={}; post=[]; supports=[]
        for i,a,r,v in ep['events']:
            if a==1 and i not in ledger:ledger[i]=(r,v)
            elif a==-1:ledger.pop(i,None)
            support=[all(record[r]==v for r,v in ledger.values()) for record in ep['records']]
            n=sum(support);post.append([float(s)/n if n else 0. for s in support]+[float(n==0)]);supports.append(support)
        ep['posterior']=post;ep['support']=supports
    return episodes
