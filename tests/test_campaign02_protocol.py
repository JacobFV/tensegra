from dataclasses import replace
from itertools import product
from topoformer.campaign02_protocol import Budget, Call, execute, execute_isolated, validate_result


def test_arithmetic_and_immutable_types():
    for name,expected in [('sub',-5),('add',9),('mul',14),('compare',-1)]:
        c=Call(name,(2,7)); r=execute(c)
        assert r.status=='success' and r.payload==expected and validate_result(c,r)
    assert execute(Call('add',(True,3))).status=='invalid'
    assert execute(Call('add',([1],2))).status=='invalid'
    assert execute(Call('unknown',())).status=='unavailable'
    assert execute(Call('add',(1,2),budget=Budget(0))).status=='timeout'


def test_lookup_filter_set():
    for c,out in [(Call('lookup',((8,3),1)),3),(Call('filter',((3,1,7),3)),(3,7)),(Call('intersection',((3,1,3),(3,2))),(3,))]:
        r=execute(c); assert r.payload==out and validate_result(c,r)


def test_path_and_provenance():
    c=Call('shortest_path',(4,((0,1,8),(0,2,1),(2,1,2),(1,3,1)),0,3),source_version=7)
    r=execute(c)
    assert r.payload==((0,2,1,3),4) and validate_result(c,r)
    assert not validate_result(c,replace(r,source_version=8))
    assert not validate_result(c,replace(r,payload=((0,3),1)))
    assert execute(Call('shortest_path',(2,(),0,1))).status=='infeasible'
    assert execute(Call('shortest_path',(2,((0,1,-1),),0,1))).status=='invalid'


def test_csp_timeout_not_unsat():
    c=Call('csp',(((0,1),(0,1)),((0,0,1,0),(0,1,1,1))))
    r=execute(c); assert r.payload==(0,1) and validate_result(c,r)
    impossible=Call('csp',(((0,),),((0,0,0,0),)))
    assert execute(impossible).status=='infeasible'
    assert execute(replace(impossible,budget=Budget(0))).status=='timeout'


def test_subset_exhaustive_reference():
    for values in product(range(-1,3),repeat=3):
        c=Call('subset',((1,2,3),values,3)); r=execute(c)
        expected=max(sum(v*x for v,x in zip(values,bits)) for bits in product((0,1),repeat=3) if sum(w*x for w,x in zip((1,2,3),bits))<=3)
        assert r.payload[1]==expected and validate_result(c,r)
    r=execute(replace(c,budget=Budget(2)))
    assert r.status=='timeout' and dict(r.certificate)['optimal'] is False and validate_result(c,r)


def test_constrained_selection():
    c=Call('constrained_subset',(((0,1,2,5),(0,2,1,8),(1,2,2,7)),4,4,((1,2),)))
    r=execute(c)
    assert r.payload==((0,2),12,3,4) and validate_result(c,r)
    assert execute(replace(c,arguments=(c.arguments[0],0,4,()))).status=='infeasible'
    assert not validate_result(c,replace(r,payload=((1,2),15,4,3)))


def test_isolated():
    c=Call('add',(2,4)); r=execute_isolated(c)
    assert r.status=='success' and r.payload==6


def test_result_envelope_rejection():
    c = Call("add", (2, 3), budget=Budget(10))
    r = execute(c)
    assert not validate_result(c, None)
    assert not validate_result(c, {"payload": 5})
    for bad in (
        replace(r, api_version="2"), replace(r, status="invented"),
        replace(r, work_units=-1), replace(r, work_units=11),
        replace(r, work_units=True), replace(r, cpu_seconds=-1),
        replace(r, cpu_seconds=float("nan")),
        replace(r, cpu_seconds=float("inf")), replace(r, cpu_seconds=True),
        replace(r, certificate=[]),
    ):
        assert not validate_result(c, bad)
    assert validate_result(c, r)
