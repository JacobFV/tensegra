import copy
import numpy as np
import pytest
import torch
from tensegra.campaign_semantics_lr import override_learning_rate,calibration_losses


def test_only_lr_changes_after_restored_moments():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));optimizer=torch.optim.AdamW([p],lr=1e-4)
    p.square().sum().backward();optimizer.step();saved=copy.deepcopy(optimizer.state_dict())
    new=torch.optim.AdamW([torch.nn.Parameter(p.detach().clone())],lr=1e-5);new.load_state_dict(saved)
    assert new.param_groups[0]['lr']==1e-4
    record=override_learning_rate(new,1e-5);actual=new.state_dict()
    assert record['before']==[1e-4] and record['after']==[1e-5]
    expected=copy.deepcopy(saved);expected['param_groups'][0]['lr']=1e-5
    assert expected['param_groups']==actual['param_groups']
    assert all(torch.equal(v,actual['state'][i][k]) for i,state in expected['state'].items() for k,v in state.items())
    with pytest.raises(ValueError):override_learning_rate(new,float('nan'))


def test_archived_calibration_loss_support(tmp_path):
    path=tmp_path/'scores.npz';np.savez(path,scores=np.zeros((3,2)),targets=np.array([[1,0],[0,0],[0,0]],dtype=bool))
    x=calibration_losses(path)
    assert x['overall']['positive']==1 and x['overall']['negative']==5
    assert x['overall']['natural_BCE']==pytest.approx(np.log(2))
    assert x['per_relation'][1]['positive']==0
