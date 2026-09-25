from tensegra.campaign02_world import Action, Item, WorldSpec, Workshop, action_catalog, encode_observation
from tensegra.campaign02_interventions import FaultedWorkshop, ReturnFault, without_tools, two_subset_return_script


def spec():
    return WorldSpec((Item('a',0,5,1),Item('b',0,1,1),Item('c',1,5,1),Item('d',1,1,1)),
                     (0,1),2,4,(),((0,1,1),(1,2,1)),0,2)


def executor(primitive,problem,budget):
    if primitive=='constrained_subset':
        constrained='capacity' in problem
        payload=((1,3),2,2,2) if constrained else ((0,2),2,10,2)
    else:
        payload=((0,1,2),2)
    return {'status':'success','payload':payload,'work_units':1,'certificate_valid':True}


def create_pair(env):
    for action in two_subset_return_script(env.observe()):
        env.step(action)
    env.step(Action('call',{'problem':'diagnostic_constrained','budget':16}))
    return [r['handle'] for r in env.observe().records if r['kind']=='computation']


def test_swap_two_valid_formal_results_does_not_repair_world_selection():
    plain=Workshop(spec(),executor)
    old,new=create_pair(plain)
    plain.step(Action('retrieve',{'handle':new}))
    assert plain.step(Action('use_return',{'handle':new,'as':'subset'})).feedback['status']=='success'
    env=FaultedWorkshop(spec(),executor,faults=(ReturnFault('swap_payload',primitive='constrained_subset'),))
    old,new=create_pair(env)
    env.step(Action('retrieve',{'handle':new}))
    assert env.step(Action('use_return',{'handle':new,'as':'subset'})).feedback['reason']=='capacity'
    audit=env.evaluate()['return_fault_audit'][0]
    assert audit['status']=='applied'
    assert audit['original'][1]['payload'][0]==(1,3)
    assert audit['supplied'][1]['payload'][0]==(0,2)
    assert env._records[new]['payload'][0]==(1,3)
    assert env.evaluate()['work_units']==2


def test_absent_record_and_work_resource_ablation():
    env=FaultedWorkshop(spec(),executor,faults=(ReturnFault('absent'),))
    create_pair(env)
    assert len([r for r in env.observe().records if r['kind']=='computation'])==1
    audit=env.evaluate()['return_fault_audit'][0]
    missing=audit['handles'][0]
    assert env.step(Action('retrieve',{'handle':missing})).feedback['status']=='invalid_input'
    assert audit['supplied']==[None]
    disabled=Workshop(without_tools(spec()),executor)
    assert disabled.observe().remaining_work==0
    disabled.step(Action('inspect',{'target':'a'}))
    disabled.step(Action('start_subset',{'handle':'p'}))
    assert not [a for a in action_catalog(disabled.observe()) if a.kind=='call']
    assert disabled.step(Action('call',{'problem':'p','budget':1})).feedback['status']=='unavailable_resource'
    assert disabled.evaluate()['work_units']==0


def test_wrong_scalar_changes_visible_value_not_exact_stored_payload():
    env=FaultedWorkshop(spec(),executor,faults=(ReturnFault('wrong_value',ordinal=1,primitive='constrained_subset',payload_path=(2,),integer_delta=3),))
    old,new=create_pair(env)
    obs=env.step(Action('retrieve',{'handle':new}))
    assert obs.retrieved[new]['payload'][2]==5
    assert env._records[new]['payload'][2]==2
    # A metadata-only scalar fault need not change exact assignment behavior.
    assert env.step(Action('use_return',{'handle':new,'as':'subset'})).feedback['status']=='success'


def test_stale_route_rejected_without_changing_world():
    env=FaultedWorkshop(spec(),executor,faults=(ReturnFault('stale',primitive='shortest_path'),))
    env.step(Action('commit_subset',{'items':['b','d']}))
    env.step(Action('inspect',{'target':'map'}))
    env.step(Action('build_route',{'handle':'route','start':0,'goal':2}))
    handle=env.step(Action('call',{'problem':'route','budget':16})).feedback['return']
    env.step(Action('retrieve',{'handle':handle}))
    assert env.step(Action('use_return',{'handle':handle,'as':'route'})).feedback['reason']=='stale_result'
    assert env.observe().state_version==0 and env.observe().position==0


def test_faults_do_not_change_pre_return_observation_or_use_gold_selection():
    plain=Workshop(spec(),executor)
    faulted=FaultedWorkshop(spec(),executor,faults=(ReturnFault('wrong_value',ordinal=3),))
    assert plain.observe()==faulted.observe()
    assert encode_observation(plain.observe())==encode_observation(faulted.observe())
    create_pair(faulted)
    assert faulted.evaluate()['return_fault_audit']==[]
