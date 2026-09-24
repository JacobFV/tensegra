import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'research/tools/campaign_a14_analysis.py'
spec=importlib.util.spec_from_file_location('a14_analysis',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_shared_indices_do_not_invent_independent_seed_graphs():
    base=np.zeros((3,4),dtype=bool);shared=np.tile([True,True,False,False],(3,1));route=shared.copy()
    result=module.joint_statistics(base,shared,route,replicates=128,seed=31)
    rng=np.random.default_rng(31);indices=rng.integers(0,4,(128,4));expected=np.quantile(shared[0][indices].mean(1),[.025,.975])
    assert np.allclose(result['gain_percentile95'],expected)
    assert result['per_seed_conditional_task']==[1.,1.,1.]
    assert result['pooled_conditional_task']==1.
    assert result['per_seed_task_and_route_correct']==[2,2,2]


def test_one_zero_gain_seed_cannot_pass_and_zero_route_is_undefined():
    base=np.zeros((3,4),dtype=bool);shared=np.ones((3,4),dtype=bool);base[0]=True
    route=np.ones_like(base);route[1]=False
    result=module.joint_statistics(base,shared,route,replicates=128,seed=31)
    assert not result['replicated_effect_pass']
    assert result['per_seed_conditional_task']==[1.,None,1.]
    assert not result['conditional_mechanistic_pass']
    empty=module.joint_statistics(base,shared,np.zeros_like(base),replicates=128,seed=31)
    assert empty['pooled_conditional_task'] is None
    assert empty['conditional_percentile95'] is None
    assert empty['undefined_conditional_bootstrap_replicates']==128
