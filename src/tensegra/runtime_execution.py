"""Protected execution of predicted lowering. No gold register is used to execute."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from .tiny_runtime import RuntimeFault


@dataclass
class Execution:
    result: int | None = None
    valid: bool = False
    complete: bool = False
    correct_steps: int = 0
    binding_correct: bool = False
    primitive_correct: bool = False
    invoked: int = 0
    deferred: int = 0
    rejected: int = 0
    trace: list[dict] = field(default_factory=list)
    symbolic_seconds: float = 0.
    binding_correct_steps: int = 0
    primitive_correct_steps: int = 0
    recovering_steps: int = 0


def semantic_selector(task, index):
    if not 0 <= index < len(task.candidates): return ('null',None)
    node=task.runtime.nodes[task.candidates[index]]
    # Aliased lexical names and repeated literal/index selectors are equivalent.
    return (node.kind,node.payload)


def run_actions(task, actions, confidences=None, threshold=0., wrong_graph=False):
    """Execute on one persistent runtime copy; rejected/deferred trajectories stop.

    Gold information is deliberately absent from this function's control flow.
    Registers change only after successful typed primitive execution.
    """
    start=time.perf_counter(); result=Execution()
    rt=copy.deepcopy(task.wrong_runtime if wrong_graph and task.wrong_runtime is not None else task.runtime)
    register=None
    from .runtime_tasks import OPS
    for step,(op,selector) in enumerate(actions):
        confidence=1. if confidences is None else float(confidences[step])
        entry={'step':step,'op':int(op),'selector':int(selector),'confidence':confidence,
               'register_before':register}
        if confidence < threshold:
            result.deferred+=1; entry['status']='deferred'; result.trace.append(entry); break
        try:
            if not 0<=int(op)<len(OPS): raise RuntimeFault('unknown operation')
            if not 0<=int(selector)<len(task.candidates): raise RuntimeFault('null/unbound selector')
            selected=task.candidates[int(selector)]; node=rt.nodes[selected]; opname=OPS[int(op)]
            if opname=='resolve':
                if node.kind!='binding': raise RuntimeFault('resolve requires binding/name selector')
                next_register=rt.value_of(rt.resolve_name(node.payload,task.scope))
            elif opname=='field':
                if node.kind!='field_slot': raise RuntimeFault('field requires field-slot/name selector')
                next_register=rt.read_field(register,node.payload)
            elif opname=='index':
                if node.kind!='index_slot': raise RuntimeFault('index requires index-slot selector')
                next_register=rt.read_index(register,node.payload)
            else:
                if node.kind!='scalar': raise RuntimeFault('arithmetic operand must be scalar')
                function=rt.builtin('sub' if opname=='rsub' else opname)
                args=[selected,register] if opname=='rsub' else [register,selected]
                next_register=rt.call(function,args,task.scope)
            register=next_register; result.invoked+=1
            entry.update(status='executed',register=register,
                         register_kind=rt.nodes[register].kind,
                         scalar=rt.nodes[register].payload if rt.nodes[register].kind=='scalar' else None)
        except (RuntimeFault,KeyError,TypeError) as error:
            result.rejected+=1; entry.update(status='rejected',error=str(error)); result.trace.append(entry); break
        result.trace.append(entry)
    if result.invoked==len(actions) and register is not None and rt.nodes[register].kind=='scalar':
        result.valid=True; result.result=int(rt.to_python(register))
    result.symbolic_seconds=time.perf_counter()-start
    return result


def evaluate_execution(task, actions, result):
    """Gold traces are consulted only after actual execution, solely for metrics."""
    selected=[semantic_selector(task,int(a[1])) for a in actions]
    gold=[semantic_selector(task,a[1]) for a in task.gold_actions]
    result.binding_correct_steps=sum(a==b and a is not None for a,b in zip(selected,gold))
    result.primitive_correct_steps=sum(int(a[0])==b[0] for a,b in zip(actions,task.gold_actions))
    result.binding_correct=len(actions)==len(gold) and result.binding_correct_steps==len(gold)
    result.primitive_correct=len(actions)==len(gold) and result.primitive_correct_steps==len(gold)
    # Call allocations have stable IDs only when exact prior execution agrees;
    # compare scalar semantics for call steps and entity IDs for object pointers.
    expected=run_actions(task,task.gold_actions)
    correct=[]
    for actual,target in zip(result.trace,expected.trace):
        match=actual['status']=='executed' and (
            actual.get('scalar')==target.get('scalar') if target.get('register_kind')=='scalar'
            else actual.get('register')==target.get('register'))
        correct.append(bool(match))
    result.correct_steps=sum(correct)
    result.complete=result.valid and len(correct)==len(gold) and all(correct)
    result.recovering_steps=sum(not a and b for a,b in zip(correct,correct[1:]))
    return result


def execute_batch(tasks, ops, selectors, confidences=None, threshold=0.,
                  permuted=False, wrong_graph=False):
    """Tensor/list boundary. Permutation is a cyclic relabeling, no hidden reset."""
    def rows(value):
        return value.detach().cpu().tolist() if hasattr(value,'detach') else value
    ops,selectors=rows(ops),rows(selectors)
    confidences=None if confidences is None else rows(confidences)
    results=[]
    for row,task in enumerate(tasks):
        indices=selectors[row]
        if permuted:
            indices=[(int(i)+1)%len(task.candidates) if int(i)<len(task.candidates) else int(i) for i in indices]
        actions=list(zip(ops[row],indices))
        executed=run_actions(task,actions,None if confidences is None else confidences[row],threshold,wrong_graph)
        results.append(evaluate_execution(task,actions,executed))
    return results
