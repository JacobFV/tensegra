"""Width16 CPU mechanics only; no acquisition or checkpoint outcome tests."""
import copy
import pytest
import torch
from tensegra.semantic_curriculum import SemanticCurriculumActor
from tensegra.campaign_semantics_grounded_actor import CopyConditionedActor
from tensegra.campaign_semantics_s18_actor import S18Actor
from tensegra.campaign_semantics_s18_compute import costs,plan
from tensegra.thinking_language import ActorInput


def actor(cls=S18Actor,**kwargs):
    torch.manual_seed(318)
    return cls(value_count=4,width=16,capacity=8,workspace_rows=3,edge_width=4,**kwargs)


def loss(model,public):
    return sum(x.square().mean() for x in model(public).values())


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


def test_original_forward_grad_and_inherited_adamw_replay():
    public=ActorInput('parent: bob, erin.',())
    original=actor(SemanticCurriculumActor,microsteps=2)
    opt=torch.optim.AdamW(original.parameters(),lr=1e-5)
    loss(original,public).backward();opt.step();opt.zero_grad(set_to_none=True)
    candidate=actor(arm='original');candidate.load_state_dict(copy.deepcopy(original.state_dict()))
    candidate_opt=torch.optim.AdamW(candidate.parameters(),lr=1e-5)
    candidate_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert [n for n,_ in original.named_parameters()]==[n for n,_ in candidate.named_parameters()]
    equal_tree(original.state_dict(),candidate.state_dict());equal_tree(opt.state_dict(),candidate_opt.state_dict())
    equal_tree(original(public),candidate(public))
    loss(original,public).backward();loss(candidate,public).backward()
    for a,b in zip(original.parameters(),candidate.parameters()):equal_tree(a.grad,b.grad)
    opt.step();candidate_opt.step()
    equal_tree(original.state_dict(),candidate.state_dict());equal_tree(opt.state_dict(),candidate_opt.state_dict())


def test_context_exact_historical_s09_and_no_parameter_change():
    candidate=actor(arm='context');historical=actor(CopyConditionedActor,microsteps=2)
    equal_tree(candidate.state_dict(),historical.state_dict())
    public=ActorInput('parent bob erin and alice',())
    equal_tree(candidate(public),historical(public))
    loss(candidate,public).backward();loss(historical,public).backward()
    for a,b in zip(candidate.parameters(),historical.parameters()):equal_tree(a.grad,b.grad)


def test_context_edge_loss_reaches_pointer_and_context_stream():
    model=actor(arm='context');public=ActorInput('parent bob erin and alice',())
    contextual=[]
    def record(module,args,output):
        if args[0].shape[1]!=3:
            output[0].retain_grad();contextual.append(output[0])
    handle=model.blocks[0].register_forward_hook(record)
    model(public)['edges'].square().mean().backward();handle.remove()
    assert len(contextual)==2 and all(x.grad is not None and x.grad.abs().sum()>0 for x in contextual)
    assert model.copy_query.weight.grad.abs().sum()>0
    assert model.copy_key.weight.grad.abs().sum()>0
    plain=actor(arm='original');plain(public)['edges'].square().mean().backward()
    assert plain.copy_query.weight.grad is None and plain.copy_key.weight.grad is None


@pytest.mark.parametrize('arm,kw',[('original',{}),('context',{}),('workspace_control',{'control_microsteps':5})])
def test_padding_pairs_and_public_only(arm,kw):
    model=actor(arm=arm,**kw);short=ActorInput('bob',());long=ActorInput('parent bob erin and alice',())
    single=model(short);batch=model.forward_batch([short,long])[0]
    assert single['copy'].shape[-1]==1
    for k in single:assert torch.allclose(single[k],batch[k],atol=3e-6,rtol=1e-5)
    pairs=torch.tensor([[0,1],[3,2],[4,6]])
    dense=model(long);sampled=model(long,pairs=pairs)
    for k in ('presence','kind','value','copy'):assert torch.equal(dense[k],sampled[k])
    for k in ('edges','slots'):assert torch.allclose(dense[k][pairs[:,0],pairs[:,1]],sampled[k],atol=1e-6)
    with pytest.raises(TypeError):model({'text':long.text,'gold_edges':'poison'})
    with pytest.raises(TypeError):model(long,gold='poison')


def test_control_matches_original_at_frozen_recurrence():
    model=actor(arm='workspace_control',control_microsteps=5)
    reference=actor(SemanticCurriculumActor,microsteps=5)
    equal_tree(model.state_dict(),reference.state_dict())
    equal_tree(model(ActorInput('alice bob',())),reference(ActorInput('alice bob',())))


def test_all_main_parameter_names_and_counts_on_meta_without_forwards():
    with torch.device('meta'):
        reference=SemanticCurriculumActor(value_count=4,width=1024,capacity=128,workspace_rows=8,microsteps=2)
        for arm,kwargs in [('original',{}),('context',{}),('workspace_control',{'control_microsteps':10})]:
            candidate=S18Actor(arm=arm,value_count=4,width=1024,capacity=128,workspace_rows=8,**kwargs)
            assert [(n,p.shape) for n,p in reference.named_parameters()]==[(n,p.shape) for n,p in candidate.named_parameters()]
            assert sum(p.numel() for p in reference.parameters())==sum(p.numel() for p in candidate.parameters())==57853781


def test_compute_matches_historical_s09_counts_and_global_rule():
    a=costs(54);b=costs(64)
    assert (a['original'],a['context'])==(1602093056,7349043200)
    assert (b['original'],b['context'])==(1771175936,8575385600)
    assert plan([54])['control_microsteps']==9 and plan([64])['control_microsteps']==10
    schedule=[54,64,64];result=plan(schedule);steps=result['control_microsteps']
    targets=sum(costs(n)['context'] for n in schedule);unit=sum(costs(n)['workspace_per_step'] for n in schedule)
    assert abs(steps*unit-targets)==min(abs(i*unit-targets) for i in range(2,30))
    assert plan(schedule*8)['control_microsteps']==steps
    with pytest.raises(ValueError):plan([])
    with pytest.raises(ValueError):costs(0)


@pytest.mark.parametrize('kwargs',[{'arm':'other'},{'arm':'context','read_scale':1},
    {'arm':'original','no_input':True},{'arm':'original','control_microsteps':10},
    {'arm':'workspace_control'},{'arm':'workspace_control','control_microsteps':2.5}])
def test_unregistered_recipe_rejected(kwargs):
    with pytest.raises(ValueError):actor(**kwargs)
