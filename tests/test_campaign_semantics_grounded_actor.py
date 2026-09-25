"""Explicit width16 mechanical fixtures; primary experiments remain1024."""
import torch
from tensegra.semantic_curriculum import SemanticCurriculumActor
from tensegra.campaign_semantics_grounded_actor import CopyConditionedActor
from tensegra.thinking_language import ActorInput


def model(cls,**kw):
    torch.manual_seed(301)
    return cls(value_count=4,width=16,capacity=8,workspace_rows=3,microsteps=1,edge_width=4,**kw)


def test_off_path_bitwise_baseline_and_same_parameters():
    reference=model(SemanticCurriculumActor);candidate=model(CopyConditionedActor,context_read=False)
    assert reference.state_dict().keys()==candidate.state_dict().keys()
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in reference.state_dict().items())
    p=ActorInput('parent: bob, erin.',())
    a=reference(p);b=candidate(p)
    assert all(torch.equal(a[k],b[k]) for k in a)
    a['edges'].sum().backward();b['edges'].sum().backward()
    assert all((p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad) for p,q in zip(reference.parameters(),candidate.parameters()))


def test_padding_sampled_heads_and_pointer_gradient():
    m=model(CopyConditionedActor);p=ActorInput('bob',());long=ActorInput('parent: erin, bob and alice.',())
    single=m(p);batched=m.forward_batch([p,long])[0]
    assert all(torch.allclose(single[k],batched[k],atol=2e-6) for k in single)
    pairs=torch.tensor([[0,1],[3,2],[4,6]])
    sampled=m(p,pairs=pairs)
    assert torch.allclose(single['edges'][pairs[:,0],pairs[:,1]],sampled['edges'],atol=1e-6)
    assert torch.allclose(single['slots'][pairs[:,0],pairs[:,1]],sampled['slots'],atol=1e-6)
    m.zero_grad();m(long)['edges'].square().mean().backward()
    assert m.copy_query.weight.grad is not None and m.copy_query.weight.grad.abs().sum()>0


def test_no_text_control_retains_only_public_copy_inventory_size():
    m=model(CopyConditionedActor,no_input=True)
    out=m(ActorInput('bob erin alice',()))
    assert out['copy'].shape==(8,3)
    assert torch.equal(out['copy'][:,0],out['copy'][:,2])
