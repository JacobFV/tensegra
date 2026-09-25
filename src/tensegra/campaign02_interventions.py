"""Harness-owned causal faults for workshop-v1; not policy actions.

Targets use public chronological computation-record ordinals, never solution
correctness. Canonical protected records remain immutable; a fault changes the
actor-visible/accessed view. These are deliberately violated memory contracts,
not examples of valid solver execution with silently repaired payloads.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from .campaign02_world import Action, Observation, Workshop, WorldSpec

VERSION = "workshop-faults-v1"


@dataclass(frozen=True)
class ReturnFault:
    kind: str
    ordinal: int = 0
    second_ordinal: int = 1
    primitive: str | None = None
    payload_path: tuple[int, ...] = (2,)
    integer_delta: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("absent", "wrong_value", "swap_payload", "stale"):
            raise ValueError("unknown return fault")
        if type(self.ordinal) is not int or self.ordinal < 0 or type(self.second_ordinal) is not int or self.second_ordinal < 0:
            raise ValueError("ordinals must be nonnegative integers")
        if self.kind == "swap_payload" and self.ordinal == self.second_ordinal:
            raise ValueError("swap requires distinct ordinals")
        if not isinstance(self.payload_path,tuple) or any(type(i) is not int or i < 0 for i in self.payload_path):
            raise ValueError("payload path must contain nonnegative indices")
        if type(self.integer_delta) is not int or self.integer_delta == 0:
            raise ValueError("wrong-value delta must be a nonzero integer")


def without_tools(spec: WorldSpec) -> WorldSpec:
    """Public zero-work resource ablation. Other world facts are identical."""
    return replace(spec, work_limit=0)


def _change_integer(value: Any, path: tuple[int,...], delta: int) -> Any:
    if not path:
        if type(value) is not int:
            raise ValueError("fault target is not an exact integer")
        return value+delta
    if not isinstance(value,(list,tuple)) or path[0] >= len(value):
        raise ValueError("fault payload path unavailable")
    changed = list(value)
    changed[path[0]] = _change_integer(changed[path[0]],path[1:],delta)
    return tuple(changed) if isinstance(value,tuple) else changed


class FaultedWorkshop(Workshop):
    """View-level return intervention, instantiated only by the evaluator.

    Multiple faults are allowed but applied in declared order; confirmation
    should normally isolate one fault. Actual-world commit/route checks remain
    the ordinary independent simulator checks. No mutation of the hidden world.
    """
    def __init__(self, spec: WorldSpec, executor=None, address_seed: int = 0,
                 *, faults: tuple[ReturnFault,...] = ()):
        super().__init__(spec,executor,address_seed)
        self._faults = faults
        self._fault_audit: list[dict[str,Any]] = []
        self._audited: set[tuple[Any,...]] = set()

    def _record_view(self) -> dict[str,dict[str,Any]]:
        view = deepcopy(self._records)
        for index,fault in enumerate(self._faults):
            candidates = [h for h,r in self._records.items() if r.get("kind")=="computation"
                          and (fault.primitive is None or r.get("primitive")==fault.primitive)]
            needed = max(fault.ordinal,fault.second_ordinal) if fault.kind=="swap_payload" else fault.ordinal
            if len(candidates) <= needed:
                continue
            handles = [candidates[fault.ordinal]]
            if fault.kind == "swap_payload":
                handles.append(candidates[fault.second_ordinal])
            key = (index,*handles)
            originals = [deepcopy(self._records[h]) for h in handles]
            status = "applied"
            try:
                if any(h not in view for h in handles):
                    raise ValueError("target absent after earlier fault")
                if fault.kind == "absent":
                    del view[handles[0]]
                elif fault.kind == "stale":
                    # Snapshot version 0 becomes -1; the world version is not
                    # incremented and no obstacle/world change is invented.
                    view[handles[0]]["state_version"] = -1
                elif fault.kind == "wrong_value":
                    record = view[handles[0]]
                    record["payload"] = _change_integer(record["payload"],fault.payload_path,fault.integer_delta)
                else:
                    a,b = (view[h] for h in handles)
                    if a.get("primitive") != b.get("primitive") or a.get("payload") is None or b.get("payload") is None:
                        raise ValueError("swap requires nonempty same-primitive payloads")
                    a["payload"],b["payload"] = deepcopy(b["payload"]),deepcopy(a["payload"])
            except (ValueError,TypeError,IndexError) as exc:
                status = f"not_applied: {exc}"
            if key not in self._audited:
                self._audited.add(key)
                self._fault_audit.append({"version":VERSION,"fault_index":index,"kind":fault.kind,
                    "target_rule":"chronological public computation ordinal", "handles":handles,
                    "status":status,"changed":[self._records[h] != view.get(h) for h in handles],
                    "original":originals,
                    "supplied":[deepcopy(view.get(h)) for h in handles]})
        return view

    def observe(self) -> Observation:
        observation = super().observe()
        view = self._record_view()
        return replace(observation,
            records=tuple({k:deepcopy(v) for k,v in r.items() if k!="payload"} for r in view.values()),
            retrieved={h:deepcopy(view[h]) for h in self._retrieved if h in view})

    def _apply(self, action: Action) -> dict[str,Any]:
        if action.kind not in ("retrieve","use_return"):
            return super()._apply(action)
        canonical = self._records
        self._records = self._record_view()
        try:
            return super()._apply(action)
        finally:
            self._records = canonical

    def evaluate(self) -> dict[str,Any]:
        result = super().evaluate()
        result["intervention_version"] = VERSION
        result["return_fault_audit"] = deepcopy(self._fault_audit)
        return result


def two_subset_return_script(observation: Observation) -> tuple[Action,...]:
    """Public scripted diagnostic, not learned orchestration or a gold solver.

    It pays normal inspection/build/call costs and creates one unconstrained and
    one fully constrained instance of the same primitive. Success, distinctness
    and payload feasibility must be measured, never presumed. No hidden item
    attributes are read to select actions. Caller chooses work budgets publicly.
    """
    actions = [Action("inspect",{"target":item["handle"]}) for item in observation.item_inventory]
    actions += [Action("start_subset",{"handle":"diagnostic_unconstrained"}),
                Action("call",{"problem":"diagnostic_unconstrained","budget":min(1024,observation.remaining_work)}),
                Action("start_subset",{"handle":"diagnostic_constrained"})]
    actions += [Action("add_constraint",{"problem":"diagnostic_constrained","constraint":c})
                for c in ("capacity","funds","incompatibility")]
    # The caller must cap the second budget using current public remaining_work;
    # this script intentionally ends before that adaptive decision.
    return tuple(actions)
