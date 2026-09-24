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


def contract_fixture():
    import json
    cfg=json.loads((path.parents[2]/'configs/campaign-a14-confirm-1401.json').read_text())
    public={key:'a'*64 for key in module.PUBLIC_KEYS}
    frozen=dict(initial_tensor_sha256='a'*64,final_tensor_sha256='a'*64,policies=list(module.POLICIES),public_sha256=public,record_order_sha256={f'c{i}':'b'*64 for i in range(3)})
    manifest=dict(seed=1401,steps=6000,presentations=96000,final_tensor_sha256='a'*64,confirmation=frozen)
    a={}
    for ci,c in enumerate(module.CONDITIONS):
        for name in ['task','suffix_value_trajectory','exact_pointer_path','all_node']:a[f'c{ci}_{name}']=np.zeros(1024,dtype=bool)
        for name in ['pred','route']:a[f'c{ci}_{name}']=np.broadcast_to(np.uint8(0),(1024,c['depth'],c['nodes']))
        names=['query_mean_route_correct']+[f'query_{kind}_{metric}' for kind in ['record','original_destination','used_destination'] for metric in ['head_correct','heads_agree']]
        for name in names:a[f'c{ci}_diagnostic_{name}']=np.broadcast_to(False,(1024,c['depth'],8))
    return cfg,manifest,{p:dict(a) for p in module.POLICIES}


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Malformed confirmation contract was accepted')


def test_loader_rejects_metadata_and_shape_contract_drift():
    import copy
    cfg,manifest,arrays=contract_fixture();module.validate_run_contract(cfg,manifest,arrays)
    for key,value in [('batch',8),('conditions',list(reversed(cfg['conditions'])))]:
        changed=copy.deepcopy(cfg);changed[key]=value;rejected(lambda:module.validate_run_contract(changed,manifest,arrays))
    for key,value in [('steps',5999),('presentations',95999)]:
        changed=copy.deepcopy(manifest);changed[key]=value;rejected(lambda:module.validate_run_contract(cfg,changed,arrays))
    for mutate in [lambda x:x['confirmation'].update(final_tensor_sha256='b'*64),
                   lambda x:x['confirmation'].update(policies=list(module.POLICIES[:-1])),
                   lambda x:x['confirmation']['public_sha256'].pop('c2_input_keys')]:
        changed=copy.deepcopy(manifest);mutate(changed);rejected(lambda:module.validate_run_contract(cfg,changed,arrays))
    for key,value in [('c2_task',np.zeros(1023)),('c2_diagnostic_query_mean_route_correct',np.zeros((1024,32,7))),('c1_route',np.zeros((1024,8,63)))]:
        changed={p:dict(a) for p,a in arrays.items()};changed['shared_hard'][key]=value
        rejected(lambda:module.validate_run_contract(cfg,manifest,changed))


def test_reference_loader_rejects_duplicate_or_substituted_identity():
    import json,copy
    cfg=json.loads((path.parents[2]/'configs/campaign-a14-engineering-references.json').read_text())
    public={key:'a'*64 for key in module.PUBLIC_KEYS}
    refs=[dict(r,policies=['a06_successful'],reference_mode=r['mode'],initial_tensor_sha256='a'*64,final_tensor_sha256='a'*64,public_sha256=public) for r in cfg['checkpoints']]
    manifest=dict(optimizer_updates=0,references=refs);module.validate_references(cfg,manifest,public)
    changed=copy.deepcopy(manifest);changed['references'][-1]=changed['references'][0]
    rejected(lambda:module.validate_references(cfg,changed,public))
    changed=copy.deepcopy(manifest);changed['references'][0]['sha256']='f'*64
    rejected(lambda:module.validate_references(cfg,changed,public))
