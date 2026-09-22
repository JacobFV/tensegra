import pytest
import torch

from topoformer.runtime_study import RuntimeStudyConfig, conditions, curriculum, policy_loss, rate


def test_resource_and_protocol_validation():
    with pytest.raises(ValueError):
        RuntimeStudyConfig(threads=8)
    with pytest.raises(ValueError):
        RuntimeStudyConfig(variants=['unknown'])
    with pytest.raises(ValueError):
        RuntimeStudyConfig(steps=5)
    config = RuntimeStudyConfig(steps=4, warmup_steps=2, checkpoints=[2])
    assert config.checkpoints == [0, 2, 4]


def test_curriculum_labels_real_policy_and_frozen_control():
    config = RuntimeStudyConfig()
    assert curriculum('warm_task', 199, config) == dict(beta=1., policy=False, frozen=False)
    assert curriculum('warm_task', 200, config) == dict(beta=0., policy=True, frozen=False)
    assert curriculum('warm_frozen', 200, config) == dict(beta=0., policy=False, frozen=True)
    assert curriculum('task_only', 0, config)['policy']


def test_score_function_changes_sampled_action_probability():
    logits = torch.tensor([0., 0.], requires_grad=True)
    log_prob = logits.log_softmax(-1)
    loss = policy_loss(log_prob, torch.tensor([1., 0.]), .5)
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]


def test_empty_conditional_is_missing_not_perfect():
    assert rate(0, 0) is None
    assert rate(1, 2) == .5


def test_depth_size_cartesian_matrix():
    config = RuntimeStudyConfig(extra_evaluations=False)
    assert {(c['nodes'], c['depth']) for c in conditions(config)} == {
        (n, d) for n in (8, 32, 64) for d in (4, 8, 16, 32)}


def test_equivalent_selector_supervision_and_null_are_finite():
    from topoformer.runtime_study import lowering_loss
    prediction = dict(op_logits=torch.zeros(1, 1, 2, requires_grad=True),
                      binding_logits=torch.zeros(1, 1, 3, requires_grad=True))
    gold = dict(ops=torch.zeros(1, 1, dtype=torch.long), selectors=torch.zeros(1, 1, dtype=torch.long),
                selector_mask=torch.tensor([[[True, True, False]]]))
    loss = lowering_loss(prediction, gold, torch.ones(1, 1))
    assert torch.allclose(loss, torch.tensor(3.).log())
    loss.backward()
    assert prediction['binding_logits'].grad[0, 0, 0] < 0
    assert prediction['binding_logits'].grad[0, 0, 1] < 0


def test_runtime_oracle_and_neural_metrics_are_separate():
    from topoformer.runtime_study import build_model, evaluate
    config = RuntimeStudyConfig(steps=0, warmup_steps=0, checkpoints=[0], eval_examples=2,
                                eval_sizes=[8], eval_depths=[2], extra_evaluations=False)
    model = build_model(config, 0)
    condition = dict(name='test', nodes=8, depth=2)
    exact = evaluate(model, 'oracle', config, condition, 123)
    neural = evaluate(model, 'neural', config, condition, 123)
    assert exact['execution_accuracy'] == 1.
    assert exact['data_hash'] == neural['data_hash']
    assert neural['execution_accuracy'] is None
    assert neural['complete_trajectory_accuracy'] is None
    assert neural['oracle_lifting_accuracy'] is None


def test_undefined_queries_are_not_successful_answers():
    from topoformer.runtime_study import build_model, evaluate
    config = RuntimeStudyConfig(steps=0, warmup_steps=0, checkpoints=[0], eval_examples=2)
    model = build_model(config, 0)
    for setting in ('ambiguous', 'invalid'):
        row = evaluate(model, 'oracle', config, dict(name=setting, nodes=8, depth=2, **{setting: True}), 5)
        assert row['counts']['defined_examples'] == 0
        assert row['task_accuracy'] is None
        assert row['counts']['appropriate_abstention'] == 2
        assert all(r['correct'] == 0 and r['defined_examples'] == 0 for r in row['confidence'])


def test_wrong_graph_changes_observable_data():
    from topoformer.runtime_study import make_data, data_hash
    config = RuntimeStudyConfig()
    clean = make_data(config, 5, nodes=8, depth=2, count=3)
    wrong = make_data(config, 5, nodes=8, depth=2, count=3, intervention='wrong_graph')
    assert data_hash(clean) != data_hash(wrong)
    assert torch.equal(clean['gold']['result'], wrong['gold']['result'])
    assert all(t.wrong_runtime is not None for t in wrong['tasks'])


def test_style_counts_partition_only_defined_answers():
    from topoformer.runtime_study import build_model, evaluate
    config = RuntimeStudyConfig(steps=0, warmup_steps=0, checkpoints=[0], eval_examples=6)
    model = build_model(config, 0)
    row = evaluate(model, 'oracle', config, dict(name='test', nodes=8, depth=2), 12)
    assert sum(x['defined_examples'] for x in row['style_breakdown'].values()) == 6
    assert sum(x['task_correct'] for x in row['style_breakdown'].values()) == row['counts']['task_correct']
    invalid = evaluate(model, 'oracle', config, dict(name='invalid', nodes=8, depth=2, invalid=True), 12)
    assert all(x['task_accuracy'] is None and x['oracle_lifting_accuracy'] is None for x in invalid['style_breakdown'].values())


def test_undefined_confidence_never_rewards_placeholder_completion(monkeypatch):
    from topoformer.runtime_study import build_model, evaluate
    from topoformer import runtime_execution
    original = runtime_execution.execute_batch
    def accidental_zero_completion(*args, **kwargs):
        rows = original(*args, **kwargs)
        for row in rows:
            row.valid, row.result, row.deferred = True, 0, 0
        return rows
    monkeypatch.setattr(runtime_execution, 'execute_batch', accidental_zero_completion)
    config = RuntimeStudyConfig(steps=0, warmup_steps=0, checkpoints=[0], eval_examples=2)
    row = evaluate(build_model(config, 0), 'oracle', config,
                   dict(name='ambiguous', nodes=8, depth=2, ambiguous=True), 5)
    assert all(r['answered'] == 2 and r['correct'] == 0 and r['defined_examples'] == 0
               for r in row['confidence'])
