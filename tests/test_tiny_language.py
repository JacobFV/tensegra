import pytest
from topoformer.tiny_language import execute
from topoformer.tiny_runtime import RuntimeFault


def test_nested_objects_arrays_and_aliases():
    result=execute('let car = {wheels: [{radius:10},{radius:11},{radius:12},{radius:13}]}; let b=car; b.wheels[3].radius')
    assert result.value == 13


def test_scopes_functions_composition_and_argument_order():
    result=execute('fn f(a,b) { return sub(a,b); } fn g(a) { return mul(a,2); } let x=5; { let x=3; f(g(x),1); } f(g(x),3)')
    assert result.value == 7
    assert len([n for n in result.runtime.nodes.values() if n.kind=='call']) >= 4


def test_no_closures_recursion_or_host_execution():
    for source in ['let x=2; fn f(a) { return add(a,x); } f(1)', 'fn f(a) { return f(a); } f(1)', '__import__(1)', 'let a=1; a=2']:
        with pytest.raises(RuntimeFault): execute(source)


def test_return_stops_body_and_duplicate_bindings_rejected():
    assert execute('fn f(a) { return a; missing; } f(3)').value == 3
    with pytest.raises(RuntimeFault): execute('let a=1; let a=2; a')
