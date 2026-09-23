import json
from topoformer.interface_study import run


def test_step_zero_artifacts_and_gates_do_not_claim_acquisition(tmp_path):
    result = run({'seeds': [2], 'main_budget_frozen':True, 'steps': 0, 'readiness_steps': 0, 'delta_steps': 0,
                  'train_size': 16, 'eval_size': 12, 'readiness_groups': 8, 'batch_size': 4,
                  'hidden': 16, 'eval_every': 1}, tmp_path)
    assert not result['gate_a_all_validation_seeds']
    assert result['composition_authorized'] is False
    assert (tmp_path/'manifest.json').exists()
    raw = json.loads((tmp_path/'seed2-raw.json').read_text())
    assert len(raw['curves']) == 1 and raw['curves'][0]['step'] == 0
    assert len(raw['readiness']['test']['records']) == 32
    assert raw['readiness']['threshold_source'] == 'validation'
    assert raw['data_hashes']['iid_validation'] != raw['data_hashes']['iid_test']
    assert raw['delta']['supplied_operation_and_operands']
