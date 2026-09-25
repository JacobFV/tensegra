import pytest
from tensegra.tiny_runtime import Runtime, RuntimeFault, PRIMITIVES


def test_aliases_slots_and_shadowing_have_distinct_nodes():
    r = Runtime()
    v = r.scalar(13)
    obj = r.record({'radius': v})
    a = r.bind('a', obj, r.root_scope)
    b = r.bind('b', obj, r.root_scope)
    inner = r.frame(r.root_scope)
    c = r.bind('a', v, inner)
    assert len({a,b,c,obj,v}) == 5
    assert r.resolve_name('a',inner) == c
    assert r.resolve_name('b',inner) == b
    assert r.value_of(a) == r.value_of(b) == obj
    assert r.read_field(obj,'radius') == v
    assert any(e.relation == 'field' and r.nodes[e.target].kind == 'field_slot' for e in r.edges)


def test_calls_have_distinct_invocations_ordered_arguments_and_returns():
    r = Runtime()
    f = r.builtin('sub')
    a,b = r.scalar(9),r.scalar(2)
    assert r.to_python(r.call(f,[a,b])) == 7
    assert r.to_python(r.call(f,[b,a])) == -7
    calls = [n for n in r.nodes.values() if n.kind == 'call']
    assert len(calls) == 2
    assert sum(e.relation == 'callee' and e.target == f for e in r.edges) == 2
    assert sum(e.relation == 'returns' for e in r.edges) >= 2
    assert 'call' in PRIMITIVES


def test_invalid_operations_leave_graph_unchanged():
    r = Runtime()
    obj = r.record({'x':r.scalar(3)})
    f = r.builtin('add')
    for op in [lambda:r.read_index(obj,0),lambda:r.read_field(obj,'missing'),lambda:r.call(f,[obj,obj])]:
        before = (dict(r.nodes),list(r.edges),list(r.trace))
        with pytest.raises(RuntimeFault): op()
        assert (r.nodes,r.edges,r.trace) == before


def test_arrays_reject_negative_boolean_and_out_of_bounds_indices():
    r=Runtime(); a=r.array([r.scalar(1)])
    for i in [-1,1,True]:
        with pytest.raises(RuntimeFault): r.read_index(a,i)


def test_return_rejects_second_write_and_lifting_is_read_only():
    r=Runtime(); frame=r.frame(); value=r.record({'x':r.scalar(2)})
    r.return_value(frame,value)
    before=(len(r.nodes),len(r.edges),len(r.trace))
    assert r.to_python(value)=={'x':2}
    assert before==(len(r.nodes),len(r.edges),len(r.trace))
    with pytest.raises(RuntimeFault): r.return_value(frame,value)


def test_failed_user_call_rolls_back_partial_execution():
    from tensegra.tiny_language import parse
    r=Runtime(); a=r.scalar(2)
    f=r.define_function('f',['a'],parse('let b=3; missing'))
    before=(dict(r.nodes),list(r.edges),list(r.trace))
    with pytest.raises(RuntimeFault): r.call(f,[a])
    assert (r.nodes,r.edges,r.trace)==before


def test_call_trace_exposes_no_future_return_node():
    r=Runtime(); result=r.call(r.builtin('add'),[r.scalar(2),r.scalar(3)])
    started=next(t for t in r.trace if t['primitive']=='call')
    ended=next(t for t in r.trace if t['primitive']=='return')
    assert int(ended['result'][1:]) >= started['node_count']
    assert int(result[1:]) >= started['node_count']
