import pytest
from topoformer.campaign_returns import selection_score,group_counts


def test_selection_prioritizes_weakest_covered_cell():
    assert selection_score([dict(correct=98,total=100),dict(correct=98,total=100)]) > selection_score([dict(correct=97,total=100),dict(correct=100,total=100)])
    with pytest.raises(ValueError):selection_score([dict(correct=1,total=2),dict(correct=1,total=3)])


def test_exact_half_unit_error_support():
    row=dict(targets=dict(value=[16,17],type=[0,1],operation=[0,1]),predictions=dict(value=[17,17]))
    groups=group_counts(row)
    assert groups['value']['16']==dict(correct=0,total=1,signed_error_sum=.5,absolute_error_sum=.5)
    assert groups['value_type_operation']['17/1/1']['correct']==1


def test_confirmation_freezes_recipe_and_data_partitions():
    import json
    from pathlib import Path
    cfg=json.loads((Path(__file__).parents[1]/'configs/campaign-r01-confirmation.json').read_text())
    assert cfg['ce_updates']==cfg['fixed_ce_step']==900
    assert cfg['ridge_grid']==[.01]
    assert 32 not in cfg['delays'] and 32 in cfg['test_delays']
    seeds=[spec['seed'] for run in cfg['runs'] for spec in run['data'].values()]
    assert len(set(seeds))==12
    assert [run['backbone_seed'] for run in cfg['runs']]==[10,11,12]
    assert all(run['data']['validation']['size']>=1024 for run in cfg['runs'])


def test_diversity_uses_one_maximal_pool_and_fixed_updates():
    import json
    from pathlib import Path
    cfg=json.loads((Path(__file__).parents[1]/'configs/campaign-r02-development.json').read_text())
    assert cfg['pool_sizes']==[4096,8192,16384]
    assert cfg['data']['train']['size']==max(cfg['pool_sizes'])
    assert cfg['ce_updates']==900 and cfg['backbone_seed']==11
    assert cfg['data']['validation']['size']==2048
    assert 32 not in cfg['delays']


def test_exposure_preserves_original_nine_hundred_prefix():
    import json
    from pathlib import Path
    cfg=json.loads((Path(__file__).parents[1]/'configs/campaign-r03-development.json').read_text())
    assert cfg['endpoints']==[900,1800,3600]
    assert cfg['head_seed']==22000011 and cfg['batch_size']==256 and cfg['lr']==.003
    assert 'ce-16384.pt' in cfg['reference_path']
    assert 32 not in cfg['delays']


def test_r04_pairs_nested_diversity_without_delayed_test_selection():
    import json
    from pathlib import Path
    cfg=json.loads((Path(__file__).parents[1]/'configs/campaign-r04-confirmation.json').read_text())
    assert cfg['pool_sizes']==[4096,16384]
    assert cfg['ce_updates']==900 and cfg['ce_batch_size']==256
    assert 32 not in cfg['delays'] and 32 in cfg['test_delays']
    seeds=[spec['seed'] for run in cfg['runs'] for spec in run['data'].values()]
    assert len(set(seeds))==12
    for run in cfg['runs']:
        assert run['data']['train']['size']==max(cfg['pool_sizes'])
        assert run['data']['calibration']['size']==1024
        assert run['data']['validation']['size']==run['data']['test']['size']==4096


def test_balanced_grid_uses_observed_typed_values_without_relabeling():
    import torch
    from topoformer.retention_data import make_batch
    from topoformer.campaign_returns_balanced import balanced_indices
    batch=make_batch(987612,1024,distractors=0)
    ids=balanced_indices(batch,2)
    assert len(ids)==104 and len(ids.unique())==104
    pairs=torch.stack([batch['targets']['type'][ids],batch['targets']['value'][ids]],1)
    assert torch.unique(pairs,dim=0,return_counts=True)[1].tolist()==[2]*52
    assert torch.equal(batch['targets']['value'][ids],(2*batch['public']['event']['values'][ids,0]+16).long())
