import pytest
from tensegra.return_diagnostics import paired, analyze, FIELDS


def row(values):
    target={k:[0,1,2,3] for k in FIELDS}
    pred={k:list(v) for k,v in target.items()}; pred['value']=values
    return dict(split='test',seed=1,distractors=2,intervention='none',targets=target,predictions=pred)


def test_paired_counts_are_not_marginal_difference():
    a,b=row([0,0,2,0]),row([0,1,0,0])
    assert paired(a,b)['value']==dict(correct_to_correct=1,wrong_to_correct=1,correct_to_wrong=1,wrong_to_wrong=1)
    assert analyze(a)['counts']['joint']['correct']==2
    assert analyze(b)['counts']['joint']['correct']==2


def test_pairing_refuses_changed_identity_recipe():
    a,b=row([0,1,2,3]),row([0,1,2,3]); b['seed']=2
    with pytest.raises(ValueError): paired(a,b)


def test_exact_value_distinct_from_half_unit():
    stats=analyze(row([1,2,3,4]))
    assert stats['counts']['value']['correct']==0
    assert stats['within_half_unit']==4
    assert stats['absolute_error']==.5


def test_frozen_capture_matches_original():
    import torch
    from tensegra.return_memory import ReturnMemoryModel
    from tensegra.retention_data import make_batch
    from tensegra.return_diagnostics_probe import capture
    # Explicit mechanical fixture; primary captures use width1024.
    torch.manual_seed(4)
    model=ReturnMemoryModel(width=24,heads=4).eval()
    public=make_batch(30,2)['public']
    with torch.no_grad():
        features,state=capture(model,public,'factorized','persistent',(0,1,2))
        actual=model(public,2)['state']
        for delay in (0,1,2):
            expected=model.norm(model(public,delay)['state'])[:,0]
            torch.testing.assert_close(features[f'workspace_{delay}'],expected)
    torch.testing.assert_close(state,actual)
    assert features['scalar_premix'].shape==(2,24)


def test_scalar_grid_has_valid_distinct_ordered_identities():
    from tensegra.return_diagnostics_grid import batch
    public,y=batch(9400101,16,'cpu')
    event=public['event']
    assert (event['arguments'][:,:,0]!=event['arguments'][:,:,1]).any(-1).all()
    assert (event['operand_values'].sum(-1)==event['values']).all()
    assert (event['types']==1).all() and (event['operations']==0).all()
    assert len(y.unique())==33
