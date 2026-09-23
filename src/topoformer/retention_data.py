"""Perfect public return events, separate supervised targets and nonce identities."""
import torch


def make_batch(seed, batch_size, feature_dim=32, distractors=2, value_limit=8):
    if value_limit < 1 or feature_dim < 2 or distractors < 0:
        raise ValueError('invalid data dimensions')
    g = torch.Generator().manual_seed(seed)
    b = batch_size
    keys = torch.randn(b, 6, feature_dim, generator=g)
    provenance = torch.randn(b, 4, feature_dim, generator=g)
    ids = torch.stack([torch.randperm(6, generator=g)[:2] for _ in range(b)])
    pid = torch.randint(4,(b,),generator=g)
    op = torch.randint(5,(b,),generator=g)
    typ = torch.randint(2,(b,),generator=g)
    typ[op==4]=2
    value = torch.randint(-2*value_limit,2*value_limit+1,(b,),generator=g).float()/2
    value[typ==0] = value[typ==0].round()
    value[typ==2] = torch.randint(2,(int((typ==2).sum()),),generator=g).float()
    # Operand scalars witness an exact valid primitive application. Identity
    # vectors are independent nonce names, not operand values or target indices.
    x = torch.randint(-value_limit,value_limit+1,(b,),generator=g).float()
    y = value-x
    y[op==1]=x[op==1]-value[op==1]
    x[op==2]=value[op==2]; y[op==2]=1
    x[op==3]=-value[op==3]; y[op==3]=0
    x[op==4]=value[op==4]; y[op==4]=.5
    rows=torch.arange(b)
    args=keys[rows[:,None],ids].clone()
    args[op==3,1]=0
    ids[op==3,1]=6  # explicit absent second operand, not a nonce identity
    query=torch.stack((torch.randint(-2*value_limit,2*value_limit+1,(b,),generator=g).float()/2,
                       torch.randint(2,(b,),generator=g).float()),-1)
    event=dict(values=value[:,None],types=typ[:,None],operations=op[:,None],
               arguments=args[:,None],provenance=provenance[rows,pid][:,None],
               operand_values=torch.stack((x,y),-1)[:,None],argument_mask=torch.stack((torch.ones(b,dtype=torch.bool),op!=3),-1)[:,None])
    # A public zero null key supports the unary missing-operand class.
    keys=torch.cat((keys,torch.zeros(b,1,feature_dim)),1)
    public=dict(event=event,argument_keys=keys,provenance_keys=provenance,query=query,
                distractors=torch.randn(b,33,distractors,feature_dim,generator=g))
    targets=dict(value=(2*value+2*value_limit).long(),type=typ,operation=op,
                 argument0=ids[:,0],argument1=ids[:,1],provenance=pid,
                 task=((value>query[:,0]) ^ query[:,1].bool()).long())
    return dict(public=public,targets=targets)
