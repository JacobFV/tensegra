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
