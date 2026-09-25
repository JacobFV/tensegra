"""Small serialization fixtures; no primary model training or GPU use."""
import copy,gzip,json
import pytest
from tensegra.campaign_semantics_continue import assert_replay


def test_replay_checks_exact_predictions_and_thresholds(tmp_path):
    row=dict(seed=1,raw={'presence':[True]},calibrated_edges='00',target={'presence':[True]})
    value=dict(train_rows=[row],rows=[row],thresholds=[.25])
    def save(name,x):
        p=tmp_path/name
        with gzip.open(p,'wt') as f:json.dump(x,f)
        return p
    old=save('old.gz',value);new=save('new.gz',value)
    assert assert_replay(old,new)
    changed=copy.deepcopy(value);changed['rows'][0]['raw']['presence']=[False]
    with pytest.raises(ValueError,match='replay mismatch'):assert_replay(old,save('changed.gz',changed))
    changed=copy.deepcopy(value);changed['thresholds']=[.5]
    with pytest.raises(ValueError,match='thresholds'):assert_replay(old,save('thresholds.gz',changed))


def test_next_batch_resume_preserves_shuffle_boundary():
    import torch
    from tensegra.campaign_semantics_continue import next_indices
    original=torch.Generator().manual_seed(31);restored=torch.Generator();restored.set_state(original.get_state())
    a=next_indices([3,2,1,0],3,original,8);b=next_indices([3,2,1,0],3,restored,8)
    assert a==b
    assert torch.equal(original.get_state(),restored.get_state())
