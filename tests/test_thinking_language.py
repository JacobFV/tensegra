from dataclasses import replace
import pytest
import torch
from topoformer.tcn_data import build_tcn_example
from topoformer.thinking_language import (ActorInput, LanguageActor, public_input,
    graph_targets, graph_losses, graph_metrics, build_splits)


def test_gold_canary_and_public_boundary():
    torch.manual_seed(4)
    model=LanguageActor(width=16,node_capacity=128,microsteps=1).eval()
    example=build_tcn_example('variable_binding',10)
    other=build_tcn_example('set_operations',11)
    changed=replace(example,privileged=other.privileged,audit={'seed':-999})
    first=model(public_input(example,'english',3))
    second=model(public_input(changed,'english',3))
    assert all(torch.equal(first[key],second[key]) for key in first)
    with pytest.raises(TypeError): model(example)
    assert set(ActorInput.__dataclass_fields__)=={'text','options'}


def test_option_permutation_is_equivariant():
    torch.manual_seed(3)
    model=LanguageActor(width=16,microsteps=1).eval()
    public=public_input(build_tcn_example('unification',3),'english',8)
    reverse=ActorInput(public.text,tuple(reversed(public.options)))
    a,b=model(public),model(reverse)
    assert torch.allclose(a['choice'],b['choice'].flip(-1),atol=1e-5)
    assert torch.allclose(a['presence'],b['presence'],atol=1e-5)


def test_graph_loss_is_finite_and_decoder_has_fixed_capacity():
    model=LanguageActor(width=16,node_capacity=128,microsteps=1)
    example=build_tcn_example('set_operations',2)
    out=model(public_input(example,'english',2))
    target=graph_targets(example.privileged.graph,128)
    losses=graph_losses(out,target)
    sum(losses.values()).backward()
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert model.presence.weight.grad.abs().sum()>0
    assert model.edge_source.weight.grad.abs().sum()>0
    assert model.cell.blocks[0].qkv.weight.grad.abs().sum()>0
    assert out['presence'].shape==(1,128)
    with pytest.raises(ValueError): graph_targets(example.privileged.graph,1)
    metrics=graph_metrics(out,target)
    assert 0<=metrics['typed_edge']['recall']<=1


def test_splits_identity_disjoint_and_surface_distinct():
    train,test,collisions=build_splits(dict(train_count=6,eval_count=3,data_seed=120,
                                          node_capacity=128,difficulty=.5))
    assert {e.audit['semantic_digest'] for e in train}.isdisjoint(e.audit['semantic_digest'] for e in test)
    assert all(len({s.text for s in e.public})==3 for e in train+test)


def test_unseen_lexical_renaming_preserves_choice_alignment():
    from topoformer.thinking_language import rendered_answer
    original=build_tcn_example('variable_binding',10)
    mapping={name:f'novelentity{i}' for i,name in enumerate(original.public[0].options)}
    renamed=build_tcn_example('variable_binding',10,identifier_renaming=mapping)
    assert renamed.audit['semantic_digest']!=original.audit['semantic_digest']
    assert all(rendered_answer(renamed,s.language) in s.options for s in renamed.public)
    assert rendered_answer(renamed,'english')==mapping[rendered_answer(original,'english')]
    assert renamed.public[0].text!=original.public[0].text
