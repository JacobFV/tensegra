import json
from tensegra.interface_study import run


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


def test_progressive_training_blocks_missing_full_gate(tmp_path):
    import pytest
    from tensegra.interface_study import run_progressive
    with pytest.raises(ValueError, match='Gate A'):
        run_progressive({}, [0,1,2], tmp_path, steps=1)


def test_proposal_factors_use_frozen_public_current_and_past(tmp_path):
    from tensegra.interface_study import proposal_confidence_records
    from tensegra.interface_proposals import ProposalModel, make_proposals
    rows = proposal_confidence_records(ProposalModel(),make_proposals(8,seed=6),condition='missing',split='iid')
    assert len(rows) == 8
    assert all(len(r['factors']) == 6 and r['factors'][-1] == 1. for r in rows)
    assert all(r['correct'] == (r['full_correct'] and r['schema']) for r in rows)


def test_schema_uses_explicit_public_type_table_not_hidden_instruction_operations():
    from dataclasses import replace
    from tensegra.interface_study import proposal_confidence_records
    from tensegra.interface_proposals import ProposalModel, make_proposals
    model = ProposalModel()
    rows = make_proposals(4,seed=9)
    altered = [replace(x,instructions=tuple((r[0],(r[1]+1)%5,r[2],r[3]) for r in x.instructions)) for x in rows]
    a = proposal_confidence_records(model,rows,condition='missing',split='iid')
    b = proposal_confidence_records(model,altered,condition='missing',split='iid')
    assert [r['factors'] for r in a] == [r['factors'] for r in b]
