"""Behavioral contracts for the recurrent public-state actor."""
import importlib.util
import torch
import pytest


def model_fixture():
    from topoformer.thinking import ThinkingConfig, ThinkingModel
    torch.manual_seed(12)
    model = ThinkingModel(ThinkingConfig(feature_dim=6, output_classes=7))
    context, memory = torch.randn(2, 4, 6), torch.randn(2, 5, 6)
    return model, context, memory


def test_module_exists():
    assert importlib.util.find_spec('topoformer.thinking') is not None


def test_recurrence_reuses_parameters_and_has_no_history():
    model, context, memory = model_fixture()
    state = model.initialize({'context': context})
    count = sum(p.numel() for p in model.parameters())
    original = state.clone()
    for t in range(3):
        result = model.step(state, context, memory, microstep=t+1)
        state = result['workspace']
    assert state.shape == (2, 8, 32)
    assert not torch.allclose(original, state)
    assert count == sum(p.numel() for p in model.parameters())
    first = model.step(original, context, memory)
    repeat = model.step(original, context, memory)
    torch.testing.assert_close(first['workspace'], repeat['workspace'])
    assert len(model.blocks) == 4
    assert len({id(b) for b in model.blocks}) == 4
    assert result['binding_logits'].shape == (2, 2, 2, 6)


def test_structural_only_bias_and_zero_strength_equivalence():
    model, context, memory = model_fixture()
    state = model.initialize({'context': context})
    graph = torch.rand(2, 5, 5)
    zero = model.step(state, context, memory, adjacency=graph, structural_strength=0)
    none = model.step(state, context, memory, adjacency=torch.zeros_like(graph), structural_strength=0)
    torch.testing.assert_close(zero['workspace'], none['workspace'])
    active = model.step(state, context, memory, adjacency=graph, structural_strength=3)
    torch.testing.assert_close(zero['attentions'][0][:, :2], active['attentions'][0][:, :2])
    assert not torch.allclose(zero['attentions'][0][:, 2:], active['attentions'][0][:, 2:])
    assert not torch.allclose(active['grounding_q'], active['grounding_k'])
    assert (active['overlap'][:, 0, 1] > 0).all()


def test_gradients_reach_grounding_candidates_and_events():
    model, context, memory = model_fixture()
    state = model.initialize({'context': context})
    out = model.step(state, context, memory)
    events = {'values':torch.ones(2,2), 'types':torch.zeros(2,2,dtype=torch.long),
              'operations':torch.zeros(2,2,dtype=torch.long), 'arguments':torch.randn(2,2,2,6)}
    encoded = model.encode_events(events)
    injected = model.inject_events(out['workspace'], encoded, out['routes'])
    final = model.step(injected, context, memory)
    loss = final['output_logits'].square().mean() + out['op_logits'].square().mean() + out['binding_logits'].square().mean() + out['readiness_logits'].square().mean()
    loss.backward()
    for name in ('ground_query', 'ground_key', 'candidate_queries', 'event_value'):
        obj = getattr(model, name)
        p = obj if isinstance(obj, torch.nn.Parameter) else next(obj.parameters())
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_event_injection_is_additive_and_preserves_unselected_rows():
    model, context, memory = model_fixture()
    state = model.initialize({'context':context})
    encoded = torch.randn(2,2,32)
    route = torch.zeros(2,2,8); route[:,:,0] = 1
    dropped = model.inject_events(state, encoded, route, torch.zeros(2,2))
    torch.testing.assert_close(state,dropped)
    added = model.inject_events(state,encoded,route)
    torch.testing.assert_close(added[:,1:],state[:,1:])
    torch.testing.assert_close(added[:,0],state[:,0]+encoded.sum(1))
    torch.testing.assert_close(model.inject_events(state,torch.zeros_like(encoded),route),state)


def test_emission_bounds_and_public_causality():
    model, context, memory = model_fixture()
    state = model.initialize({'context':context, 'hidden_graph':torch.randn(2,9,9)})
    torch.testing.assert_close(state,model.initialize({'context':context}))
    early = model.step(state,context,memory,microstep=0)
    late = model.step(state,context,memory,microstep=model.config.max_microsteps)
    assert not early['emit'].any()
    assert late['emit'].all()
    assert (early['emit_probability'] == 0).all()
    assert (late['emit_probability'] == 1).all()
    changed = model.step(state,context,memory+2)
    assert not torch.allclose(changed['workspace'],model.step(state,context,memory)['workspace'])


def test_empty_masked_memory_is_finite_and_null_bindings():
    model,context,memory = model_fixture()
    state = model.initialize({'context':context})
    out = model.step(state,context,memory,memory_mask=torch.zeros(2,5,dtype=torch.bool))
    assert torch.isfinite(out['workspace']).all()
    assert (out['binding_logits'].softmax(-1)[...,-1] == 1).all()


def test_candidate_identity_points_to_declared_slots_separate_from_values():
    model,context,memory = model_fixture()
    state = model.initialize({'context':context})
    slots = torch.randn(2,7,6)
    mask = torch.ones(2,7,dtype=torch.bool); mask[:,-1] = False
    out = model.step(state,context,memory,identity_memory=slots,identity_mask=mask)
    assert out['candidate_id_logits'].shape == (2,2,7)
    assert torch.isneginf(out['candidate_id_logits'][...,-1]).all()
    assert out['binding_logits'].shape[-1] == 6
    out['candidate_id_logits'][...,:-1].square().sum().backward()
    assert model.identity_query.weight.grad.abs().sum() > 0


def test_event_roundtrip_has_independent_value_type_and_operation_heads():
    model,_,_ = model_fixture()
    encoded = model.encode_events(dict(values=torch.ones(2,2),types=torch.zeros(2,2,dtype=torch.long),operations=torch.ones(2,2,dtype=torch.long),arguments=torch.zeros(2,2,2,6)))
    heads = model.event_roundtrip(encoded)
    assert heads['value'].shape == (2,2)
    assert heads['semantic_logits'].shape == (2,2,3)
    assert heads['op_logits'].shape == (2,2,5)
    sum(v.square().mean() for v in heads.values()).backward()
    assert model.event_value.weight.grad.abs().sum() > 0


def test_predicted_typed_graph_depends_on_public_context():
    model,context,memory = model_fixture()
    state = model.initialize({'context':context})
    original = model.step(state,context,memory)
    changed = model.step(state,context+3,memory)
    assert original['predicted_adjacency'].shape == (2,2,5,5)
    assert not torch.allclose(original['predicted_adjacency'],changed['predicted_adjacency'])


def test_grounding_has_explicit_null_even_without_observable_nodes():
    model,context,memory = model_fixture()
    state = model.initialize({'context':context})
    out = model.step(state,context,memory,memory_mask=torch.zeros(2,5,dtype=torch.bool))
    for key in ('grounding_q','grounding_k'):
        assert out[key].shape == (2,2,8,6)
        assert (out[key][...,-1] == 1).all()
        assert (out[key][...,:-1] == 0).all()
    empty = model.step(state,context,memory[:,:0])
    assert (empty['grounding_q'] == 1).all()
    assert torch.isfinite(empty['workspace']).all()


def test_structural_geometry_preserves_relation_identity():
    model,context,memory = model_fixture()
    state = model.initialize({'context':context})
    graph = torch.zeros(2,2,5,5)
    graph[:,0,0,1] = 1
    with torch.no_grad():
        model.relation_strength[:,0] = 1
        model.relation_strength[:,1] = 4
    first = model.step(state,context,memory,adjacency=graph)
    swapped = model.step(state,context,memory,adjacency=graph.flip(1))
    assert not torch.allclose(first['attentions'][0][:,2:],swapped['attentions'][0][:,2:])
    torch.testing.assert_close(first['attentions'][0][:,:2],swapped['attentions'][0][:,:2])
