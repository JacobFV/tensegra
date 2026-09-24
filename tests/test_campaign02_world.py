from dataclasses import replace
from itertools import product

from topoformer.campaign02_world import (Action,Workshop,generate_world,validate_subset,
    action_catalog,encode_action,encode_observation)


def feasible(spec):
    groups = [[x.handle for x in spec.items if x.category==c] for c in spec.categories]
    return next(h for h in product(*groups) if validate_subset(spec,h)[0])


def test_generation_feasible_and_private():
    for seed in range(20):
        spec=generate_world(seed)
        assert feasible(spec)
        obs=Workshop(spec).observe()
        assert obs.known_items == {} and obs.known_edges is None
        assert all(set(x)=={'handle','category'} for x in obs.item_inventory)
        assert 'seed' not in obs.to_dict()
        assert len(encode_observation(obs))==26
        assert len({len(encode_action(obs,a)) for a in action_catalog(obs)})==1


def test_public_catalog_observational_equivalence():
    spec=generate_world(5)
    hidden=replace(spec,items=tuple(replace(i,price=i.price+100) for i in spec.items),blocked_edge=(1,2))
    a,b=Workshop(spec).observe(),Workshop(hidden).observe()
    assert a==b
    assert action_catalog(a)==action_catalog(b)


def test_retrieval_is_addressed_persistent_and_copied():
    env=Workshop(generate_world(1))
    first=env.step(Action('inspect',{'target':env.observe().item_inventory[0]['handle']})).feedback['return']
    second=env.step(Action('inspect',{'target':'map'})).feedback['return']
    assert first != second and not first.startswith('return_')
    assert not env.observe().retrieved
    obs=env.step(Action('retrieve',{'handle':first}))
    assert obs.retrieved[first]['payload']['handle']==obs.item_inventory[0]['handle']
    obs.retrieved[first]['payload']['price']=-100
    assert env.observe().retrieved[first]['payload']['price'] >=1
    assert first in env.step(Action('think')).retrieved


def test_explicit_constraints_not_completed():
    calls=[]
    def executor(primitive,problem,budget):
        calls.append((primitive,problem,budget))
        return {'status':'unknown','payload':None,'work_units':1,'certificate_valid':True}
    env=Workshop(generate_world(2),executor)
    env.step(Action('inspect',{'target':env.observe().item_inventory[0]['handle']}))
    env.step(Action('start_subset',{'handle':'p'}))
    env.step(Action('add_constraint',{'problem':'p','constraint':'capacity'}))
    env.step(Action('call',{'problem':'p','budget':16}))
    assert calls[0][1]['constraints']==['capacity']
    assert 'max_cost' not in calls[0][1] and len(calls[0][1]['items'])==1
    assert env.evaluate()['work_units']==1


def test_commit_validator_and_obstacle_feedback():
    spec=generate_world(3,obstacle=True)
    env=Workshop(spec)
    assert env.step(Action('commit_subset',{'items':[]})).feedback['status']=='rejected'
    env.step(Action('commit_subset',{'items':list(feasible(spec))}))
    env.step(Action('inspect',{'target':'map'}))
    obs=env.step(Action('deliver',{'path':[0,spec.destination]}))
    assert obs.feedback['status']=='obstacle' and obs.state_version==1
    obs=env.step(Action('deliver',{'path':list(range(spec.destination+1))}))
    assert obs.delivered
    assert env.step(Action('verify')).verified
    assert env.evaluate()['verified_success']


def test_work_budget_and_stale_problem():
    env=Workshop(generate_world(4,obstacle=True),lambda *args: {'status':'success','payload':None,'work_units':1,'certificate_valid':True})
    env.step(Action('inspect',{'target':'map'}))
    env.step(Action('build_route',{'handle':'route','start':0,'goal':env.observe().goal['destination']}))
    assert env.step(Action('call',{'problem':'route','budget':99999})).feedback['status']=='unavailable_resource'
    env.step(Action('commit_subset',{'items':list(feasible(env._spec))}))
    env.step(Action('deliver',{'path':[0,env._spec.destination]}))
    assert env.step(Action('call',{'problem':'route','budget':16})).feedback['reason']=='stale_problem'


def test_invalid_returns_cannot_teleport():
    env=Workshop(generate_world(6),lambda *a:{'status':'success','payload':((0,),1,1,1),'work_units':1,'certificate_valid':True})
    env.step(Action('inspect',{'target':env.observe().item_inventory[0]['handle']}))
    env.step(Action('start_subset',{'handle':'p'}))
    handle=env.step(Action('call',{'problem':'p','budget':16})).feedback['return']
    assert env.step(Action('use_return',{'handle':handle,'as':'subset'})).feedback['reason']=='retrieve_required'
    env.step(Action('retrieve',{'handle':handle}))
    assert env.step(Action('use_return',{'handle':handle,'as':'route'})).feedback['status']=='invalid_input'
    assert not env.observe().delivered


def test_immutable_call_snapshot_and_result_features():
    env=Workshop(generate_world(7),lambda *a:{'status':'success','payload':((0,),1,2,3),'work_units':1,'certificate_valid':True})
    env.step(Action('inspect',{'target':env.observe().item_inventory[0]['handle']}))
    env.step(Action('start_subset',{'handle':'p'}))
    h=env.step(Action('call',{'problem':'p','budget':16})).feedback['return']
    before=env.observe().records[-1]['problem_snapshot']
    env.step(Action('add_constraint',{'problem':'p','constraint':'funds'}))
    assert env.observe().records[-1]['problem_snapshot']==before
    assert before['problem']['constraints']==[]
    action=Action('use_return',{'handle':h,'as':'subset'})
    old=encode_action(env.observe(),action)
    env.step(Action('retrieve',{'handle':h}))
    assert encode_action(env.observe(),action)!=old


def test_move_travel_cost_and_exhaustion():
    spec=generate_world(8,travel_limit=0)
    env=Workshop(spec)
    assert env.step(Action('move',{'destination':1})).feedback['reason']=='travel_budget'
    env=Workshop(replace(spec,travel_limit=64))
    env.step(Action('move',{'destination':1}))
    assert env.observe().position==1
    assert env.evaluate()['travel_distance']>0
    assert env.evaluate()['cost']>spec.action_price


def test_address_renaming_changes_only_spelling():
    spec=generate_world(9)
    a,b=Workshop(spec,address_seed=1),Workshop(spec,address_seed=2)
    action=Action('inspect',{'target':'map'})
    x,y=a.step(action),b.step(action)
    assert x.feedback['return'] != y.feedback['return']
    assert encode_observation(x)==encode_observation(y)
    ax=Action('retrieve',{'handle':x.feedback['return']});ay=Action('retrieve',{'handle':y.feedback['return']})
    assert encode_action(x,ax)==encode_action(y,ay)
