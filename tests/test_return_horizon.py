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


def test_main_configuration_and_hash_roundtrip(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from topoformer.return_horizon_main import sha, write_json
    path=tmp_path/'receipt.json'
    write_json(path, {'events':32,'seed':10})
    assert json.loads(path.read_text())=={'events':32,'seed':10}
    assert sha(path)==hashlib.sha256(path.read_bytes()).hexdigest()
    config=json.loads((Path(__file__).parents[1]/'configs/stage11-return-main.json').read_text())
    assert config['backbone_seeds']==[10,11,12]
    assert config['eval_size']>=512
    assert config['test_seed']!=config['validation_seed']
    assert 32 in config['eval_delays'] and 32 not in config['curve_delays']
    for schedule in config['schedules'].values():
        assert 32 not in schedule
    training={config['training_seed_base']+i*10000+s for i in range(3) for s in range(config['updates'])}
    assert len(training)==3*config['updates']
    assert not(training & {config['test_seed'],config['validation_seed']})
