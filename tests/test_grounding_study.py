import pytest
import torch

from topoformer.grounding_study import (GroundingStudyConfig, batch_hash, build_model,
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
