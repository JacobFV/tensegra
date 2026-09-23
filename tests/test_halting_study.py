"""Independent halting contracts: hidden labels, real evidence, and strict gates."""
import importlib.util
import pytest
import torch


def api():
    assert importlib.util.find_spec('topoformer.halting_study'), 'halting diagnostic is missing'
    from topoformer import halting_study
    return halting_study


def test_generator_keeps_labels_private_and_holds_out_names_and_arrivals():
    h = api()
    train = h.generate_episodes(96, 4, 'train')
    held = h.generate_episodes(96, 4, 'validation')
    assert set(train['public']) == {'context', 'memory', 'mask'}
    assert set(train['names']).isdisjoint(held['names'])
    assert set(train['arrival'][~train['reject']].tolist()) == {2, 4, 6}
    assert set(held['arrival'][~held['reject']].tolist()) == {3, 5, 7}
    for i in range(96):
        stop = int(held['arrival'][i])
        rows = held['public']['memory'][i]
        key = held['public']['context'][i, :, 0, :7]
        genuine_second = ((rows[:, :, :7] == key[:, None]).all(-1)
                          & (rows[:, :, 8] == 1) & (rows[:, :, 9] == 1)).any(-1)
        assert not genuine_second[:stop-1].any()
        assert held['public']['mask'][i].all()
        if held['reject'][i]:
            assert stop == 8 and int(held['answer'][i]) == 2
            assert not genuine_second.any()
        else:
            assert genuine_second[stop-1:].all()
        assert held['public']['context'][i, -1, 0, -1] == 1
        assert not held['public']['context'][i, :-1, 0, -1].any()


def test_oracle_control_stops_on_evidence_and_charges_actual_compute():
    h = api()
    data = h.generate_episodes(12, 2, 'validation')
    model = h.make_model(width=8)
    records = h.evaluate(model, data, 'oracle')
    assert [r['stop'] for r in records] == data['arrival'].tolist()
    assert all(r['updates'] == r['stop'] for r in records)
    minimum = h.evaluate(model, data, 'minimum')
    assert all(r['updates'] == 1 for r in minimum)
    assert h.summarize(minimum)['premature_rate'] == 1


def test_supervised_losses_reach_real_four_block_cell_and_both_heads():
    h = api()
    model = h.make_model(width=8)
    losses = h.losses(model, h.generate_episodes(4, 9, 'train'))
    (losses['task'] + losses['halt']).backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.emit_probe.weight.grad.abs().sum() > 0
    assert all(b.qkv.weight.grad.abs().sum() > 0 for b in model.blocks)


def test_gate_requires_all_seeds_and_three_successful_arrivals():
    h = api()
    good = {'reject_accuracy': .99, 'stop_reject_accuracy': .99, 'premature_rate': 0., 'task_accuracy': .99,
            'arrival_accuracy': {'3': .99, '5': .99, '7': .99}}
    minimum = {'task_accuracy': .4}
    assert h.gate([{'learned': good, 'minimum': minimum}])['passed']
    bad = dict(good, premature_rate=.01)
    assert not h.gate([{'learned': good, 'minimum': minimum}, {'learned': bad, 'minimum': minimum}])['passed']
    assert not h.gate([{'learned': dict(good, arrival_accuracy={'3': 1., '5': 1.}), 'minimum': minimum}])['passed']


def test_misleading_values_do_not_encode_the_missing_answer():
    h = api()
    data = h.generate_episodes(96, 12, 'train')
    agreements = []
    for i in range(96):
        if data['reject'][i]:
            continue
        rows = data['public']['memory'][i, -1]
        key = data['public']['context'][i, -1, 0, :7]
        match = (rows[:, :7] == key).all(-1)
        true_b = rows[match & (rows[:, 8] == 1) & (rows[:, 9] == 1), 11].item()
        rumor_b = rows[rows[:, 9] == 0, 11].item()
        agreements.append(true_b == rumor_b)
    assert any(agreements) and not all(agreements)


def test_pilot_runner_never_authorizes_composition_and_does_not_inspect_test(tmp_path):
    h = api()
    import json
    config = dict(seeds=[0], width=8, steps=0, batch_size=4, train_examples=4,
                  eval_examples=4, eval_every=1, learning_rate=.001, main_budget_frozen=False)
    target = tmp_path/'result'
    result = h.run(config, target)
    assert not result['composition_allowed']
    assert not list(target.glob('seed-*/test-*.json'))
    manifest = json.loads((target/'manifest.json').read_text())
    assert manifest['seeds'][0]['optimizer_steps'] == 0
    assert len(manifest['seeds'][0]['hashes']['checkpoint.pt']) == 64


def test_evidence_arrival_preserves_role_certification_and_name_histograms():
    h = api()
    data = h.generate_episodes(12, 13, 'validation')
    for i in range(12):
        if data['reject'][i]:
            continue
        arrival = int(data['arrival'][i])
        before = data['public']['memory'][i, arrival-2]
        after = data['public']['memory'][i, arrival-1]
        assert torch.equal(before[:, 7:10].sum(0), after[:, 7:10].sum(0))
        assert torch.equal(before[:, :7].sum(0), after[:, :7].sum(0))


def test_gate_does_not_hide_failed_rejection_in_aggregate_accuracy():
    h = api()
    learned = dict(stop_reject_accuracy=.99, reject_accuracy=.94, premature_rate=0.,
                   task_accuracy=.99, arrival_accuracy={'3': 1., '5': 1., '7': 1.})
    assert not h.gate([{'learned': learned, 'minimum': {'task_accuracy': .3}}])['passed']


def test_unrevealed_share_counterfactual_preserves_every_public_prefix():
    h = api()
    left = h.generate_episodes(96, 10, 'train', missing_share_seed=1)
    right = h.generate_episodes(96, 10, 'train', missing_share_seed=2)
    changed = 0
    for i in range(96):
        stop = int(left['arrival'][i])
        prefix = 8 if left['reject'][i] else stop-1
        for key in left['public']:
            assert torch.equal(left['public'][key][i, :prefix], right['public'][key][i, :prefix])
        if left['answer'][i] != right['answer'][i]:
            changed += 1
    assert changed > 20


def test_exact_evidence_rule_solves_xor_and_waits_for_observable_deadline():
    h = api()
    data = h.generate_episodes(96, 18, 'validation')
    predictions = h.rule_predictions(data['public'])
    assert [p['prediction'] for p in predictions] == data['answer'].tolist()
    assert [p['stop'] for p in predictions] == data['arrival'].tolist()
    assert all(p['model_updates'] == 0 for p in predictions)
