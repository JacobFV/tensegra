import importlib.util
from pathlib import Path


def evaluator():
    path=Path(__file__).resolve().parents[1]/'research/tools/campaign02_evaluate.py'
    spec=importlib.util.spec_from_file_location('campaign02_eval_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_descriptors_and_default():
    m=evaluator()
    assert m.condition_faults({})==()
    f=m.condition_faults({'fault':{'kind':'wrong_value','payload_path':[0,0]}})[0]
    assert f.payload_path==(0,0)
    try:
        m.condition_faults({'fault':{'kind':'absent'},'return_faults':[]})
        assert False
    except ValueError:
        pass


def test_fault_counts_do_not_equate_claimed_certificate_with_task_use():
    m=evaluator()
    outcome={'history':[
        {'action':{'kind':'retrieve','arguments':{'handle':'r'}},
         'feedback':{'record':{'primitive':'constrained_subset','certificate_valid':True,'status':'success'}},'state_version':0},
        {'action':{'kind':'use_return','arguments':{'handle':'r','as':'subset'}},
         'feedback':{'status':'rejected','reason':'capacity'},'state_version':0}],
        'return_fault_audit':[{'handles':['r'],'changed':[True],'status':'applied'}]}
    counts=m.episode_counts(outcome)
    assert counts['return_address_contract_valid']==1
    assert counts['fault_changed_application_rejected']==1
    assert counts['fault_changed_application_success']==0
    assert counts['fault_changed_records']==1


def test_paired_effect_support_and_duplicate_rejection():
    m=evaluator()
    def row(seed,success,changed):
        return {'seed':seed,'semantic_spec_hash':'same','outcome':{'verified_success':success,'utility':float(success)},
                'counts':{'fault_changed_records':int(changed),'fault_changed_use_attempts':int(changed)}}
    control=[row(1,True,False),row(2,True,False)]
    faulty=[row(1,False,True),row(2,True,False)]
    result=m.paired_condition_outcomes(control,faulty)
    assert result['success_transitions_control_to_intervention']=={'1->0':1,'1->1':1}
    assert result['changed_record_support']==1
    assert result['mean_utility_difference']==-.5
    try:
        m.paired_condition_outcomes([control[0],control[0]],[faulty[0],faulty[0]])
        assert False
    except ValueError:
        pass


def test_resource_pairing_preserves_physical_semantics():
    m=evaluator()
    a={'items':[1,2],'capacity':2,'work_limit':128,'compute_price':1}
    b={**a,'work_limit':0,'compute_price':2}
    assert m.semantic_spec_hash(a)==m.semantic_spec_hash(b)
    assert m.semantic_spec_hash(a)!=m.semantic_spec_hash({**a,'capacity':3})


def test_every_registered_sealed_condition_builds_through_evaluator():
    import json, sys
    from tensegra.campaign02_world import generate_world
    m = evaluator()
    tools = Path(__file__).resolve().parents[1]/'research/tools'
    sys.path.insert(0, str(tools))
    import campaign02_e09_config as e09
    cfg = json.loads(json.dumps(e09.build([], 2, [])))  # exact JSON round trip
    for condition in cfg['conditions']:
        spec = generate_world(condition['seed_start'], **m.world_kwargs(condition['world']))
        assert isinstance(spec.call_budgets, tuple)
        m.condition_faults(condition)
