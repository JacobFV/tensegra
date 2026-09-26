"""Branching support for extended-04 (design v2 revision 5 / review F7).

- ``SolverCache``: an exact memo around a depworld executor
  ``executor(primitive, problem, max_work) -> dict``, keyed by (primitive,
  canonical problem JSON, budget). The frozen protocol solver is a deterministic
  function of work units, so a cached result is the result an uncontended call
  returns. The single source of nondeterminism is the solver's **wall-clock
  deadline** (2 s per call): a call that hits it returns ``timeout`` with
  work_units below the budget (in-process ``execute``), or a killed worker's
  ``timeout``/``unknown`` with the ``process_killed_or_no_result`` certificate
  (``BoundedSolver``). Such wall-limited results are **never cached** (passed
  through and counted), so a cache hit can only ever return a deterministic,
  work-limited outcome. Consequently a cache hit changes a status only where the
  uncached call itself would have been wall-limited (nondeterministic); on
  deterministic calls the cached and uncached results are identical in status,
  payload, work_units, certificate and certificate_valid. Hits report zero
  solver CPU (``cpu_seconds``/``child_cpu_seconds`` = 0.0; the environment's
  ``solver_cpu_seconds`` is logging only and never enters utility).
- ``clone_branch``: copy an episode for counterfactual branching: the
  environment (``DepWorkshop.clone()``, sharing the executor), the actor's
  recurrent state row (if any) and the progress tracker state.

Torch is imported lazily (only when a hidden tensor is cloned).
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable

CACHE_VERSION = "solver-cache-v1"
WALL_CERTIFICATE = ("process_killed_or_no_result", True)


def cache_key(primitive: str, problem: dict, max_work: int) -> str:
    """sha256 of the canonical (primitive, problem, budget) JSON (collisions negligible)."""
    return hashlib.sha256(json.dumps([primitive, problem, max_work], sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def wall_limited(result: dict, max_work: int) -> bool:
    """True when the outcome may depend on wall-clock time rather than work units."""
    status = result.get("status")
    certificate = tuple(tuple(c) if isinstance(c, list) else c for c in (result.get("certificate") or ()))
    if WALL_CERTIFICATE in certificate:
        return True
    if status == "unknown":
        return True
    return status == "timeout" and result.get("work_units", max_work) < max_work


class SolverCache:
    """Exact solver-result memo; share one instance across clones of an episode (or a
    whole evaluation condition). Callable with the executor signature."""

    def __init__(self, executor: Callable, max_entries: int | None = None):
        self.executor = executor
        self.max_entries = max_entries
        self._store: dict[str, dict] = {}
        self.hits = self.misses = self.uncacheable = self.evictions = 0

    def __call__(self, primitive: str, problem: dict[str, Any], max_work: int) -> dict[str, Any]:
        key = cache_key(primitive, problem, max_work)
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            return {**deepcopy(hit), "cpu_seconds": 0.0, "child_cpu_seconds": 0.0}
        self.misses += 1
        result = self.executor(primitive, problem, max_work)
        if wall_limited(result, max_work):
            self.uncacheable += 1
            return result
        if self.max_entries is not None and len(self._store) >= self.max_entries:
            self._store.pop(next(iter(self._store)))
            self.evictions += 1
        self._store[key] = deepcopy(result)
        return result

    def __deepcopy__(self, memo):
        return self  # a cache is shared, never copied (it may wrap a live solver worker)

    def stats(self) -> dict[str, Any]:
        return {"version": CACHE_VERSION, "entries": len(self._store), "hits": self.hits, "misses": self.misses,
                "uncacheable_wall_limited": self.uncacheable, "evictions": self.evictions}


def clone_branch(env, hidden=None, tracker=None):
    """(env copy sharing the executor, hidden-state copy or None, tracker copy or None).

    ``hidden`` is the actor's recurrent state for this episode (a tensor row, e.g.
    ``hidden[i:i+1]``, or None for the lightweight family, which has no recurrent state).
    """
    env_copy = env.clone()
    hidden_copy = None if hidden is None else hidden.detach().clone()
    tracker_copy = None if tracker is None else tracker.copy()
    return env_copy, hidden_copy, tracker_copy
