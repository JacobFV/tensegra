import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('audit_binding', Path(__file__).parents[1]/'scripts'/'audit_binding_checkpoints.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixture_checkpoint(tmp_path):
    state = {'grounders.0.query_null_logit': torch.tensor(.4), 'strengths': torch.tensor([[4.]])}
    config = {'strength': 4.}
    source = {'commit': 'test'}
    filename = 'cold-seed0.pt'
    torch.save(dict(state_dict=state, config=config, source=source, variant='cold', seed=0), tmp_path/filename)
    row = dict(checkpoint=filename, variant='cold', seed=0, config_hash=audit_module.fingerprint(config),
               source=source, variant_options={'identity_init': False, 'null_init': 0.},
               training={'final_state_hash': audit_module.fingerprint({k:audit_module.tensor_hash(v) for k,v in state.items()})})
    return row


def test_saved_state_matches_and_exposes_prior(tmp_path):
    result = audit_module.audit_checkpoint(fixture_checkpoint(tmp_path), tmp_path)
    assert result['initial_null_logit'] == 0.
    assert result['initial_identity_prior'] is False
    assert result['parameters']['grounders.0.query_null_logit'] == pytest.approx(.4)
    assert len(result['checkpoint_sha256']) == 64


def test_rejects_wrong_weights_and_metadata(tmp_path):
    row = fixture_checkpoint(tmp_path)
    row['training']['final_state_hash'] = 'bad'
    with pytest.raises(ValueError, match='state hash'):
        audit_module.audit_checkpoint(row, tmp_path)
    row = fixture_checkpoint(tmp_path)
    row['seed'] = 1
    with pytest.raises(ValueError, match='variant/seed'):
        audit_module.audit_checkpoint(row, tmp_path)
