import pytest
import torch

from tensegra.grounding_study import (GroundingStudyConfig, batch_hash, build_model,
                                       conditions, evaluate, make_data, model_inputs, run)


def tiny(**overrides):
    return GroundingStudyConfig(**dict(dict(seeds=[0], variants=['soft'], steps=1,
        checkpoints=[0, 1], batch_size=2, eval_examples=2, eval_batch_size=2,
        nodes=4, key_dim=8, width=16, heads=2, classes=4,
        eval_depths=[1, 2], eval_sizes=[4], max_train_depth=2,
        corruptions=[], eval_distractors=[]), **overrides))


def test_model_boundary_excludes_all_gold():
    batch = make_data(tiny(), 1, depth=2)
    inputs = model_inputs(batch)
    assert not set(inputs) & {'targets', 'token_nodes', 'path_nodes', 'clean_adjacency', 'start_nodes'}
    assert batch_hash(batch) == batch_hash(make_data(tiny(), 1, depth=2))


def test_one_factor_grid_is_deduplicated():
    config = tiny()
    grid = conditions(config)
    assert [x['name'] for x in grid] == ['depth_1', 'depth_2', 'heldout_composition']
    assert len({str(x['settings']) for x in grid}) == len(grid)


def test_evaluation_distinguishes_pre_step_and_full_path():
    config = tiny()
    model, mode = build_model(config, 'soft', 0)
    result = evaluate(model, mode, config, conditions(config)[0], 100)
    assert result['oracle_task_accuracy'] == 1
    assert result['attention_oracle_task_accuracy'] == 1
    assert result['attention_oracle_exact_path_completion'] == 1
    assert 'exact_pre_step_grounding' in result
    assert 'exact_path_completion' in result
    assert 'clean_next_attention_mass' in result
    assert len(result['step_diagnostics']) == 1
    assert len(result['example'][0]['probabilities']) == config.nodes + 1


def test_resume_checks_config_and_source(tmp_path):
    config = tiny()
    rows = run(config, tmp_path)
    assert len(rows) == 1
    assert run(config, tmp_path) == rows
    with pytest.raises(ValueError, match='different experiment configuration'):
        run(tiny(learning_rate=.002), tmp_path)


def test_validation_rejects_incomplete_budget():
    with pytest.raises(ValueError, match='checkpoints'):
        tiny(checkpoints=[0])


def test_correct_start_does_not_count_as_completed_path():
    config = tiny()
    class WrongDestination(torch.nn.Module):
        def forward(self, batch, **kwargs):
            count, nodes, _ = batch['entity_keys'].shape
            start = (batch['start_keys'][:, None] * batch['entity_keys']).sum(-1).argmax(-1)
            successor = batch['adjacency'][torch.arange(count), batch['relations'][:, 0], start].argmax(-1)
            wrong = (successor + 1) % nodes
            pq = torch.nn.functional.one_hot(start, nodes + 1).float()[:, None]
            after = torch.nn.functional.one_hot(wrong, nodes + 1).float()[:, None]
            identities = (batch['token_keys'] @ batch['entity_keys'].transpose(-1, -2)).argmax(-1)
            pk = torch.nn.functional.one_hot(identities, nodes + 1).float()
            attention = (identities == wrong[:, None]).float()
            attention /= attention.sum(-1, keepdim=True)
            return torch.zeros(count, config.classes), [dict(pq=pq, pq_after=after, pk=pk, attention=attention[:, None, None])]
    result = evaluate(WrongDestination(), 'soft', config, conditions(config)[0], 100)
    assert result['exact_pre_step_grounding'] == 1.
    assert result['exact_path_completion'] == 0.
    assert result['exact_attention_path_completion'] == 0.


def test_periodic_grounders_preserve_common_parameter_initialization():
    config = tiny()
    base, _ = build_model(config, 'soft', 2)
    periodic, _ = build_model(config, 'period4', 2)
    for name, tensor in base.state_dict().items():
        if not name.startswith('grounders.') and name != 'strengths':
            torch.testing.assert_close(tensor, periodic.state_dict()[name], rtol=0, atol=0)


def test_keyed_controls_have_matched_parameters_and_content_prior():
    config = tiny()
    reference, mode = build_model(config, 'soft_keyed', 2)
    assert mode == 'soft'
    assert reference.content_identity_bias == 8.
    assert torch.all(reference.strengths == 16.)
    for variant, expected_mode in [('none_keyed', 'none'), ('graph_input_keyed', 'graph_input')]:
        model, actual_mode = build_model(config, variant, 2)
        assert actual_mode == expected_mode
        assert model.content_identity_bias == 8.
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    assert not any(name.endswith('_keyed') for name in GroundingStudyConfig().variants)
