from topoformer.return_horizon import acquisition_gate, schedule_counts


def test_schedule_matching_and_forbidden_horizon():
    short=schedule_counts([0,1,2,4,2,4],1000)
    wide=schedule_counts([0,1,2,4,8,16],1000)
    assert short['0']==wide['0']==167
    assert short['1']==wide['1']==167
    assert sum(short.values())==sum(wide.values())==1000
    import pytest
    with pytest.raises(ValueError): schedule_counts([32],1)


def test_acquisition_requires_complete_support():
    assert not acquisition_gate([])
    assert acquisition_gate([{'counts':{'joint':{'correct':32,'total':32}}}])
    assert not acquisition_gate([{'counts':{'joint':{'correct':31,'total':32}}}])
