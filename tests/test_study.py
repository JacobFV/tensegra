import json

import pytest
import torch

from topoformer.study import StudyConfig, first_crossing, run_study, tensor_hash


def test_first_crossing_is_censored():
    curve = [{'step': 0, 'normalized_mse': 3.0}, {'step': 25, 'normalized_mse': 1.0}]
    assert first_crossing(curve, 2) == 25
    assert first_crossing(curve, .5) is None


def test_config_rejects_invalid():
    with pytest.raises(ValueError):
        StudyConfig(checkpoints=[0, 10, 5])
    with pytest.raises(ValueError):
        StudyConfig(eval_batch_size=128)


def test_tensor_hash_is_sensitive():
    assert tensor_hash(torch.tensor([1, 2])) != tensor_hash(torch.tensor([2, 1]))


def test_smoke_artifacts_and_pairing(tmp_path):
    config = StudyConfig(suites=['efficiency'], domains=['sparse'], seeds=[0],
                        counts=[2, 4], checkpoints=[0, 1], steps=8,
                        validation_count=2, test_count=2, horizon=2,
                        width=8, heads=2, layers=1, batch_size=2,
                        modes=['none', 'soft4'], wall_seconds=60)
    summary = run_study(config, tmp_path / 'out')
    rows = summary['runs']
    assert len(rows) == 4
    assert all(row['status'] == 'complete' for row in rows)
    assert all(row['completed_steps'] == 1 for row in rows)
    assert rows[0]['schedule_hash'] == rows[1]['schedule_hash']
    assert rows[0]['shared_initialization_hash'] == rows[1]['shared_initialization_hash']
    assert rows[0]['normalization'] == rows[2]['normalization']
    assert rows[0]['graph_splits'] == rows[2]['graph_splits']
    assert rows[0]['evaluations'][0]['deterministic_rollout_mse'] >= 0
    assert rows[0]['evaluations'][0]['oracle_deterministic_rollout_mse'] == 0
    saved = [json.loads(line) for line in (tmp_path/'out'/'metrics.jsonl').read_text().splitlines()]
    assert saved == rows
    assert (tmp_path/'out'/'summary.json').exists()
    with pytest.raises(FileExistsError):
        run_study(config, tmp_path/'out')


def test_deadline_saves_partial_summary(tmp_path):
    config = StudyConfig(suites=['learned'], domains=['sparse'], seeds=[0],
                        counts=[2], checkpoints=[0, 100], steps=8,
                        validation_count=2, test_count=2, horizon=2,
                        width=8, heads=2, layers=1, batch_size=2, wall_seconds=.001)
    summary = run_study(config, tmp_path/'out')
    assert summary['status'] == 'partial'
    assert (tmp_path/'out'/'summary.json').exists()


def test_normalization_uses_only_smallest_subset():
    from topoformer.study import _dataset
    config = StudyConfig(counts=[2, 4], train_count=4, validation_count=2,
                        test_count=2, steps=8, horizon=2)
    train, validation, test, normalization = _dataset(config, 'sparse', 'uniform', 0)
    values = train[0]['series'][:2].flatten()
    assert normalization == {'mean': values.mean().item(), 'std': values.std(unbiased=False).item()}
    assert len({split[0]['trajectory_seed'] for split in (train, validation, test)}) == 3


def test_transfer_uses_shared_model_and_disjoint_graphs(tmp_path):
    config = StudyConfig(suites=['transfer'], seeds=[0], modes=['none'], counts=[2],
                        train_count=2, validation_count=2, test_count=2,
                        transfer_train_graphs=3, transfer_validation_graphs=2,
                        transfer_test_graphs=1, transfer_test_sizes=[12, 32],
                        checkpoints=[0, 1], steps=8, horizon=2, width=8,
                        heads=2, layers=1, batch_size=2, eval_windows=4)
    rows = run_study(config, tmp_path/'transfer')['runs']
    assert len(rows) == 2
    for row in rows:
        assert {evaluation['nodes'] for evaluation in row['evaluations']} == {12, 32}
        hashes = [{item['graph_hash'] for item in row['graph_splits'][split]}
                  for split in ('train', 'validation', 'test')]
        assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
        assert row['training_trajectories'] == 6
    assert rows[0]['graph_splits']['test'] == rows[1]['graph_splits']['test']


def test_corruption_cases_and_clean_only_runtime(tmp_path):
    from topoformer.study import _cases
    config = StudyConfig(suites=['corruption'], domains=['sparse'], seeds=[0],
                        modes=['hard'], counts=[2], train_count=2, validation_count=2,
                        test_count=2, checkpoints=[0, 1], steps=8, horizon=2,
                        width=8, heads=2, layers=1, batch_size=2)
    cases = list(_cases(config))
    assert len(cases) == 8
    assert {case['fraction'] for case in cases} == {0, .1, .25, .5}
    rows = run_study(config, tmp_path/'corruption')['runs']
    assert len(rows[0]['evaluations']) == 8
    assert all(len(row['evaluations']) == 1 for row in rows[1:])
    assert all(row['completed_steps'] == 1 for row in rows)


def test_signed_typed_information_label(tmp_path):
    config = StudyConfig(suites=['heterogeneous'], domains=['sparse'], seeds=[0],
                        modes=['typed'], counts=[2], train_count=2, validation_count=2,
                        test_count=2, checkpoints=[0, 1], steps=8, horizon=2,
                        width=8, heads=2, layers=1, batch_size=2)
    row = run_study(config, tmp_path/'signed')['runs'][0]
    assert row['mechanism'] == 'signed'
    assert row['information'] == 'signed_edge_weights'
    assert torch.tensor(row['validation_curve'][0]['coefficients']).abs().sum() == 0


def test_schedule_indices_match_selected_graph_sizes():
    from topoformer.study import _schedule
    config = StudyConfig(checkpoints=[0, 50], batch_size=4)
    train = [{'x': torch.zeros(3, 9, 4)}, {'x': torch.zeros(17, 12, 4)}]
    graph_indices, indices = _schedule(config, train, 43)
    for graph, batch in zip(graph_indices, indices):
        assert batch.min() >= 0
        assert batch.max() < len(train[graph]['x'])
    other_graphs, other_indices = _schedule(config, train, 43)
    assert torch.equal(graph_indices, other_graphs)
    assert torch.equal(indices, other_indices)
