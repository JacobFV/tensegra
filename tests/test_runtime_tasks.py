from topoformer.runtime_tasks import make_batch, FAMILIES
from topoformer.runtime_execution import execute_batch


def test_oracle_all_families_and_public_has_no_future_nodes():
    for family in FAMILIES:
        batch = make_batch(2, nodes=3, depth=4, seed=19, family=family)
        results = execute_batch(batch['tasks'], batch['gold']['ops'], batch['gold']['selectors'])
        assert all(x.valid and x.complete and x.result == t.gold_result for x,t in zip(results,batch['tasks']))
        assert all(not any(n.kind in {'call','returned','argument'} for n in t.runtime.nodes.values()) for t in batch['tasks'])
        assert 'result' not in batch['public'] and 'ops' not in batch['public']


def test_determinism_and_permutation():
    a = make_batch(2, seed=12)
    b = make_batch(2, seed=12)
    assert all(a['public'][k].equal(b['public'][k]) for k in a['public'])
    assert a['gold']['result'].equal(b['gold']['result'])


def test_bad_type_and_null_are_rejected_without_gold_reset():
    batch = make_batch(1, depth=3, family='nested', seed=2)
    ops = batch['gold']['ops'].clone(); ops[0,0] = 1
    bad = execute_batch(batch['tasks'],ops,batch['gold']['selectors'])[0]
    assert not bad.valid and bad.rejected == 1 and not bad.complete
    null = batch['gold']['selectors'].clone(); null[0,0] = batch['public']['candidate_keys'].shape[1]
    assert execute_batch(batch['tasks'],batch['gold']['ops'],null)[0].rejected == 1


def test_confidence_deferral_is_failure():
    batch = make_batch(1,seed=4)
    low = batch['gold']['ops'].float()*0
    result = execute_batch(batch['tasks'],batch['gold']['ops'],batch['gold']['selectors'],low,.5)[0]
    assert result.deferred == 1 and not result.valid and not result.complete


def test_lexical_equivalence_has_shared_keys_and_supervision():
    batch = make_batch(1, nodes=8, depth=8, family='scope', noise=0., seed=3)
    task=batch['tasks'][0]; public=batch['public']; gold=batch['gold']
    for step,(op,selector) in enumerate(task.gold_actions):
        target=task.runtime.nodes[task.candidates[selector]]
        for index,node_id in enumerate(task.candidates):
            node=task.runtime.nodes[node_id]
            equivalent=(node.kind,node.payload)==(target.kind,target.payload)
            assert bool(gold['selector_mask'][0,step,index]) == equivalent
            if equivalent:
                assert public['candidate_keys'][0,index].equal(public['reference'][0,step])
    shadow_choices=gold['selector_mask'][0,0].nonzero().flatten()
    assert len(shadow_choices)==2
    swapped=gold['selectors'].clone(); swapped[0,0]=shadow_choices[0]
    result=execute_batch(batch['tasks'],gold['ops'],swapped)[0]
    assert result.binding_correct and result.complete


def test_all_initial_edges_and_payloads_are_public():
    batch=make_batch(1, nodes=3,depth=4,family='nested',seed=7)
    from topoformer.runtime_tasks import RELATIONS
    task=batch['tasks'][0]; table={node:i for i,node in enumerate(task.candidates)}
    assert len(table)==len(task.runtime.nodes)
    assert batch['public']['adjacency'].sum().item()==len(task.runtime.edges)
    for edge in task.runtime.edges:
        assert batch['public']['adjacency'][0,RELATIONS.index(edge.relation),table[edge.source],table[edge.target]]==1
    for node_id,node in task.runtime.nodes.items():
        if node.kind=='scalar':
            assert batch['public']['candidate_values'][0,table[node_id]].item()==node.payload/64


def test_wrong_binding_persists_and_world_is_immutable():
    batch=make_batch(1,nodes=3,depth=4,family='composition',seed=8)
    task=batch['tasks'][0]; initial_nodes=len(task.runtime.nodes)
    choices=[i for i,n in enumerate(task.candidates) if task.runtime.nodes[n].kind=='binding'
             and not batch['gold']['selector_mask'][0,0,i]]
    selected=batch['gold']['selectors'].clone(); selected[0,0]=choices[0]
    result=execute_batch(batch['tasks'],batch['gold']['ops'],selected)[0]
    assert result.valid and not result.binding_correct
    assert result.trace[0]['register'] != task.gold_registers[0]
    assert len(task.runtime.nodes)==initial_nodes


def test_corruption_matches_symbolic_and_public_graphs():
    from topoformer.runtime_tasks import RELATIONS
    batch=make_batch(1,nodes=4,depth=4,family='nested',seed=33,corruption=1.)
    task=batch['tasks'][0]; ids={n:i for i,n in enumerate(task.candidates)}
    assert task.wrong_runtime is not None
    assert batch['public']['adjacency'].sum().item()==len(task.wrong_runtime.edges)
    for edge in task.wrong_runtime.edges:
        assert batch['public']['adjacency'][0,RELATIONS.index(edge.relation),ids[edge.source],ids[edge.target]]==1


def test_unbound_and_schema_stress_have_no_hidden_answer():
    for setting in ('ambiguous','invalid'):
        batch=make_batch(2,seed=3,**{setting:True})
        assert not batch['gold']['expected_valid'].any()
        assert all(t.gold_result is None for t in batch['tasks'])
        results=execute_batch(batch['tasks'],batch['gold']['ops'],batch['gold']['selectors'])
        assert all(not x.valid and x.rejected==1 for x in results)
        if setting=='ambiguous':
            assert batch['gold']['selector_mask'][:,0,-1].all()
            assert not batch['public']['reference'][:,0].any()


def test_forward_execution_ignores_gold():
    from topoformer.runtime_execution import run_actions
    batch=make_batch(1,seed=2)
    task=batch['tasks'][0]; actions=task.gold_actions
    before=run_actions(task,actions)
    task.gold_result=1234; task.gold_actions=(); task.gold_registers=()
    after=run_actions(task,actions)
    assert before.result==after.result and before.trace==after.trace


def test_output_style_is_not_fixed_by_family():
    seen={family:set() for family in FAMILIES}
    for seed in range(12):
        for task in make_batch(6,depth=1,nodes=1,seed=seed)['tasks']:
            seen[task.family].add(task.style)
    assert all(styles=={0,1,2} for styles in seen.values())
