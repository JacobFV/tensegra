"""Small CPU mechanical checks, not experimental acquisition evidence."""
import json
import torch
from topoformer.campaign_composition import make_lowering_batch, model_inputs
from topoformer.campaign_composition_acquire import private_labels, labels_from_public, controlled_rows, score, run
from topoformer.interface_proposals import ProposalModel


def test_public_control_targets_and_required_null_denominators():
    data = make_lowering_batch(48009, 128)
    batch = model_inputs(data['public']); labels = private_labels(data['labels'])
    public_labels = labels_from_public(data['public'])
    for k in labels: assert torch.equal(labels[k], public_labels[k])
    out = dict(primitive=torch.nn.functional.one_hot(labels['primitive'], 5).float(),
               pointers=torch.nn.functional.one_hot(labels['targets'], 11).float())
    unary = labels['primitive'] == 3
    out['pointers'][unary, 2] = torch.nn.functional.one_hot(labels['targets'][unary, 1], 11).float()
    counts = score(out, batch, labels)
    assert counts['full_semantic']['correct'] == 128
    assert counts['raw_null_pointer_diagnostic']['correct'] == 0
    assert counts['required_operand1']['total'] == int((~unary).sum())
    assert counts['canonical_null_schema']['correct'] == int(unary.sum())
    for kind in ('record_order', 'inventory_order', 'fresh_names', 'reverse_roles', 'unrelated_instructions'):
        altered = controlled_rows(data['public'], kind, 48010)
        fresh = labels_from_public(altered)
        model_inputs(altered)
        assert torch.equal(fresh['primitive'], labels['primitive'])
        if kind in ('record_order', 'fresh_names', 'unrelated_instructions'):
            assert torch.equal(fresh['targets'], labels['targets'])


def test_tiny_runner_artifact_selection_and_public_only_inputs(tmp_path, monkeypatch):
    import topoformer.campaign_composition_acquire as module
    # Architecture reduction solely tests I/O, backward, selection and exports.
    monkeypatch.setattr(module, 'make_model', lambda: ProposalModel(key_dim=32, hidden=16))
    config = dict(seed=400, hidden=1024, key_dim=32, cpu_threads=1, learning_rate=.001, weight_decay=.01,
                  batch_size=4, updates=1, checkpoints=[0, 1], control_seed=48020,
                  data={split: dict(seed=48021+i, count=8) for i, split in enumerate(('train', 'calibration', 'validation'))},
                  controls=['reverse_roles', 'inventory_order'])
    result = run(config, tmp_path / 'run', 'cpu')
    assert result['selected_step'] in (0, 1)
    assert result['optimizer_presentations'] == 4
    assert 1 <= result['unique_training_events_visited'] <= 4
    visits = torch.load(tmp_path / 'run' / 'training-visitation.pt', weights_only=True)
    assert int(visits.sum()) == 4
    import hashlib
    assert result['selected_checkpoint_sha256'] == hashlib.sha256((tmp_path / 'run' / 'selected.pt').read_bytes()).hexdigest()
    assert result['parameter_count'] == sum(p.numel() for p in ProposalModel(key_dim=32, hidden=16).parameters())
    saved = torch.load(tmp_path / 'run' / 'validation.pt', weights_only=True)
    assert not {'event', 'values', 'targets', 'labels', 'reference'} & saved['public'].keys()
    assert json.loads((tmp_path / 'run' / 'summary.json').read_text())['validation'] == result['validation']
