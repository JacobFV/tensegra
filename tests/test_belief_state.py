import torch
from topoformer.belief_state import make_episodes,collate,BeliefModel,loss,oracle


def test_oracle_conditions():
    for condition in ('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty'):
        b=collate(make_episodes(5,condition=condition))
        assert torch.equal(oracle(b['public']),b['targets']['posterior'])


def test_public_only_gradients_and_masks():
    b=collate(make_episodes(2,candidates=3)+make_episodes(1,candidates=5))
    for mode in ('protected','recurrent'):
        m=BeliefModel(mode,width=32,inner=8)  # unit-test-only width
        o=m(b['public']); sum(loss(o,b).values()).backward()
        assert all(p.grad is not None for p in m.parameters() if not (mode=='protected' and p is m.readout.weight or mode=='protected' and p is m.readout.bias))
        assert (o['logits'][0,:,:3].isfinite()).all()
        assert (o['logits'][0,:,3:5]<-1e8).all()
        assert m.width==32
    import inspect
    assert inspect.signature(BeliefModel).parameters['width'].default==1024


def test_causal_candidate_equivariance_and_private_targets():
    b=collate(make_episodes(2,candidates=5)); m=BeliefModel(width=32,inner=8)
    original=m(b['public'])['logits']
    b['targets']['posterior'].zero_()
    assert torch.equal(original,m(b['public'])['logits'])
    perm=torch.tensor([2,4,0,1,3]); p={k:v.clone() for k,v in b['public'].items()}
    p['records']=p['records'][:,perm]; p['valid']=p['valid'][:,perm]
    assert torch.allclose(m(p)['logits'][:,:,:5],original[:,:,:5][:,:,perm],atol=1e-6)
    p={k:v.clone() for k,v in b['public'].items()}; p['event'][:,-1].neg_()
    assert torch.equal(m(p)['logits'][:,:-1],original[:,:-1])


def test_ledger_redundancy_retraction_reordering():
    m=BeliefModel(width=32,inner=8)
    outputs={c:m(collate(make_episodes(3,seed=7,condition=c))['public'])['logits'] for c in ('clean','duplicate','retract','reorder')}
    assert torch.equal(outputs['clean'][:,-1],outputs['duplicate'][:,-1])
    assert torch.equal(outputs['clean'][:,-1],outputs['retract'][:,-1])
    # Independently reordered episodes use same records but consume different RNG
    # only after record generation; single-episode comparison is exact.
    x={c:m(collate(make_episodes(1,seed=7,condition=c))['public'])['logits'][:,-1] for c in ('clean','reorder')}
    assert torch.allclose(x['clean'],x['reorder'],atol=1e-6)


def test_episode_split_and_ambiguity():
    train=make_episodes(64,seed=2); test=make_episodes(64,seed=3)
    assert not ({str(e['records'])+str(e['keys']) for e in train}&{str(e['records'])+str(e['keys']) for e in test})
    assert sum(sum(p>0 for p in e['posterior'][1])>1 for e in train)>48
    assert all(sum(p>0 for p in e['posterior'][-1])==1 for e in train)


def test_gate_requires_every_validation_cell():
    from topoformer.belief_study import gate
    assert not gate([],[])
    assert not gate([],[(0,'clean',8)])
    row=dict(seed=0,condition='clean',candidates=8,count=512,split='validation',regime='iid',frames=[dict(support_accuracy=1.,impossible_mass=0.,posterior_l1=0.)])
    assert gate([row],[(0,'clean',8)])
    assert not gate([dict(row,split='test')],[(0,'clean',8)])
    assert not gate([row],[(0,'clean',8),(1,'clean',8)])
    assert not gate([dict(row,frames=[dict(support_accuracy=1.,impossible_mass=0.,posterior_l1=.5)])],[(0,'clean',8)])


def test_public_id_features_are_consumed_by_both_modes():
    from topoformer.belief_state import observation_features
    ids=torch.tensor([[-1,3,3,12]])
    code=observation_features(ids)
    assert code.shape==(1,4,16) and not code[0,0].any()
    assert torch.equal(code[0,1],code[0,2]) and not torch.equal(code[0,1],code[0,3])
    import pytest
    with pytest.raises(ValueError): observation_features(torch.tensor([65536]))
    b=collate(make_episodes(2,candidates=5))
    for mode in ('recurrent','protected'):
        m=BeliefModel(mode,width=32,inner=8,observation_id_features=True)
        captured=[]
        hook=m.encode.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
        baseline=m(b['public'])['logits']; first=captured[1]
        changed={k:v.clone() for k,v in b['public'].items()};changed['ids'][:,1]=500
        captured.clear();altered=m(changed)['logits'];second=captured[1]
        hook.remove()
        assert torch.equal(first[...,:-16],second[...,:-16])
        assert not torch.equal(first[...,-16:],second[...,-16:])
        assert torch.equal(baseline[:,0],altered[:,0])


def test_idmatched_arms_share_parameter_initialization():
    torch.manual_seed(79)
    a=BeliefModel('protected',width=32,inner=64,observation_id_features=True)
    torch.manual_seed(79)
    b=BeliefModel('recurrent',width=32,inner=64,observation_id_features=True)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(value,b.state_dict()[name]) for name,value in a.state_dict().items())
