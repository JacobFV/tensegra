import torch
from topoformer.interface_readiness import readiness_factors, make_readiness, Calibrator, select_threshold, readiness_metrics, gate_b


def test_product_is_score_schema_rejection_and_stable_wrong():
    factors = readiness_factors(.9, [.8, .7], False, 1.)
    assert factors[-2] == 0 and torch.tensor(factors).prod() == 0
    records = make_readiness(120, seed=8)
    assert any(r['kind'] == 'stable_wrong' and r['factors'][-1] == 1 and not r['correct'] for r in records)
    assert any(r['kind'] == 'zero_ready' for r in records)
    assert any(r['kind'] == 'multiple_ready' for r in records)


def test_threshold_uses_precision_and_executable_recall():
    scores = [0.99, 0.97, 0.8, 0.1]
    truth = [True, True, False, True]
    threshold = select_threshold(scores, truth)
    assert threshold == .97
    m = readiness_metrics(scores, truth, threshold)
    assert m['precision'] == 1 and m['recall'] == 2/3
    assert not gate_b(dict(m, local_advantage=0))
    assert gate_b(dict(m, local_advantage=.2))


def test_calibrator_schema_mask_is_exact_and_trainable():
    model = Calibrator()
    x = torch.tensor([[.9, .9, .9, 0., 1.], [.9, .9, .9, 1., 1.]])
    assert model(x)[0] == 0
    model(x)[1].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_correlated_and_high_support_wrong_are_present():
    records = make_readiness(120, seed=8)
    assert any(not r['correct'] and min(r['factors'][:3]) > .97 and r['factors'][-1] == 1 for r in records)
