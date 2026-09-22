import json

import pytest
import torch

from topoformer.binding_study import BindingStudyConfig, VARIANTS, build_model, conditions, model_inputs, run
from topoformer.grounding_study import make_data


def test_matrix_is_full_cartesian():
    config = BindingStudyConfig()
    grid = conditions(config)
    assert len(grid) == 20
    assert {(c['settings']['nodes'], c['settings']['depth']) for c in grid} == {
        (n, d) for n in [16, 32, 64, 128] for d in [4, 8, 16, 32, 64]}


def test_no_gold_enters_model():
    batch = make_data(BindingStudyConfig(), 1, depth=2, batch_size=2)
    inputs = model_inputs(batch)
    assert not {'targets', 'path_nodes', 'token_nodes', 'start_nodes'} & inputs.keys()
    assert 'adjacency' in inputs and 'start_keys' in inputs


def test_registry_supervision_is_separate_and_cold_prior_controlled():
    assert len(VARIANTS) == 23
    for update in ('attention', 'pointer'):
        for suffix in ('aux001', 'aux01', 'aux1', 'ground1', 'nullcycle'):
            options = VARIANTS[f'random_cosine_{update}_{suffix}']
            assert options['identity_init'] is False
            assert options['null_init'] == 0
    assert VARIANTS['random_cosine_attention_nullprior']['null_init'] == .65


@pytest.mark.parametrize('variant,strength,mode,content', [('stage3_soft4',4.,'soft',0.), ('stage3_soft8',8.,'soft',0.), ('known',8.,'known',0.), ('graph_input_keyed',16.,'graph_input',8.)])
def test_stage3_baseline_logits_are_bitexact(variant,strength,mode,content):
    from topoformer.traversal_model import TraversalTransformer
    config = BindingStudyConfig()
    new, mode, _ = build_model(config, variant, 4)
    torch.manual_seed(4)
    old = TraversalTransformer(key_dim=16, width=32, heads=4, classes=8, relations=3,
                               temperature=.05, strength=strength, content_identity_bias=content)
    batch = model_inputs(make_data(config, 12, depth=4, batch_size=2))
    torch.testing.assert_close(new(batch, mode=mode), old(batch, mode=mode), atol=0, rtol=0)


def test_validation_resource_and_names():
    with pytest.raises(ValueError):
        BindingStudyConfig(variants=['missing'])
    with pytest.raises(ValueError):
        BindingStudyConfig(eval_batch_size=17)


def test_smoke_resume_and_reject_changed_configuration(tmp_path):
    config = BindingStudyConfig(seeds=[0], variants=['random_cosine_attention_aux1'], steps=1,
        checkpoints=[0, 1], batch_size=2, eval_examples=2, eval_batch_size=2,
        eval_depths=[2], eval_sizes=[4], nodes=4, max_train_depth=2,
        save_checkpoints=False)
    rows = run(config, tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row['training']['steps'] == 1
    assert len(row['initial_evaluations']) == 2
    assert len(row['evaluations']) == 1
    assert row['training']['losses'][0]['ground'] >= 0
    assert 0 <= row['evaluations'][0]['task_accuracy'] <= 1
    assert run(config, tmp_path) == rows
    saved = json.loads((tmp_path/'config.json').read_text())
    saved['steps'] = 2
    (tmp_path/'config.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='different configuration'):
        run(config, tmp_path)
