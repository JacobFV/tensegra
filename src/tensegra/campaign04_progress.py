"""Progress diagnostic v1 for depworld (extended-04 design §4 + v2 revision 9).

Public information only: every quantity below is read from ``DepObservation``
objects (and the action the actor chose). Nothing here touches the hidden spec,
the evaluator, the solver or any RNG, so running the tracker never changes a
trajectory. Torch-free and standard-library only.

Relevant-state signature (the equivalence relation)
---------------------------------------------------
Two public decision states are **equivalent** iff their signature tuples are
equal. The signature (``signature()``) consists of:

- **drafts**, keyed by their agent-chosen public name (``problem_N``, including
  the foreign drafts registered at t=0): the exact problem content (canonical
  JSON) plus ``depends_on`` (the requirement versions / selection it was built
  against). ``created_step`` is excluded. Drafts stay keyed by name as in P2a:
  two differently named drafts are two workspace objects.
- **pending choices**: pending item set, pending slot assignment.
- **commitments**: selection_id, assignment_id, verified.
- **computation records, keyed by content** (v2 rev. 9), never by the random
  record handle: (primitive, canonical problem snapshot, depends_on, budget,
  status), plus the public **plan context** the call was made under (committed
  selection_id, assignment_id and all requirement versions at the calling
  step; foreign records: the t=0 context). A deterministic re-call with
  unchanged inputs and unchanged plan therefore adds nothing. ``depends_on`` is
  part of the content because it decides applicability. The plan context makes
  a call "after a dependency change" count as new work even when the draft's
  own content is unchanged (e.g. a route recomputed after a new assignment,
  which changes the deadline slack it is judged against; this mirrors the
  public stage-freshness rule ``campaign03_depworld._stage_open``). Without it
  the supplied dep_recompute schedule would score as no-progress.
- **retrieved set**, also by record content.
- **position** (travel is excluded, as in P2a: A->B->A is a cycle).
- **requirement versions and fired events** (kind, argument, affected, revoked;
  never the event step).
- **information availability**: requirements known, map known, the set of
  inspected item handles.

Excluded: step counters, remaining steps/work/travel, cumulative travel/work/
cost, the attempt log, the one-step feedback, ``created_step`` fields, record
handles and applicability relation values. The relation values are functions of
components already in the signature (records' content + requirements, known
items/edges, selection, position, versions), so they are redundant; the tests
assert that equal signatures give equal relation values.

Several components are monotone (versions, events, inspected set, record set),
so a state after a new inspection, an event or a new computation can never equal
an earlier one: those steps break cycles automatically.

Per-step classes (explicit precedence, first match wins)
--------------------------------------------------------
1. ``rejected``           environment rejection (status other than success on a
                          non-call action, or a call that did not execute), with
                          its reason. ``repeat_rejection`` marks a rejection of
                          the same canonical action from the same signature.
                          (``abstain`` is ``terminal``, listed separately.)
2. ``useful_inspection``  inspect of a target not yet in the availability set.
3. ``solver_work``        an executed call on new inputs: a new (primitive,
                          problem, depends_on) (new problem or dependency
                          change), or a budget larger than every earlier record
                          on those inputs, or a retry after a possibly
                          stochastic failure (status ``unknown``, or ``timeout``
                          with work_units < budget, i.e. wall-clock limited).
4. ``triggered_revision`` a successful ``uncommit`` after a trigger since the
                          revised piece was committed (a public event, a
                          rejection, a call result infeasible/timeout/unknown,
                          or a newly retrieved record), or an ``add_constraint``
                          that refreshes a draft's ``depends_on``. It records a
                          trigger, not a rationality judgement.
5. ``short_cycle``        the post-step signature equals the signature after one
                          of the preceding W=6 steps (or the episode start),
                          cycle length L in 2..W, with no intervening progress
                          mark (a useful inspection, solver work or an event).
6. ``idempotent``         the signature is unchanged (cycle length 1). ``think``
                          with an unchanged signature is idempotent (consistent
                          with P2a); its work is charged separately by the
                          environment. ``latent_stall`` flags a run of more than
                          2 consecutive unchanged-signature thinks.
7. ``other_progress``     anything else that changes the signature.
``terminal``: abstain (ends the episode; not no-progress).

**No-progress** = ``idempotent`` or ``short_cycle``. Retries are suppressed only
when the relevant inputs are unchanged: a repeated call with the same content is
no-progress (unless the earlier result was possibly stochastic); a repeated
action after new evidence, budget or dependencies is classified by what changed.

R-mask flags (design F5)
------------------------
When step t is no-progress, the pair (pre-step signature s, canonical action key
a) is flagged: ``a`` is masked whenever the current signature equals ``s``. No
separate release rule is needed: any change of the relevant signature releases
it (and it re-applies if the state returns to ``s``). ``verify`` and ``abstain``
are never masked; at most ``mask_cap`` keys are masked per signature. Canonical
keys equal ``campaign03_depworld.action_key`` except that fresh draft names are
dropped from start_* actions and record handles are replaced by record content.

Documented differences from the P2a registered operationalization
(research/tools/campaign03_p2a_analysis.py): P2a's state omits records and uses
retrieved handles; any call breaks a P2a cycle and a call step is never a P2a
cycle. Here repeated identical calls/retrievals are no-progress (content keyed),
triggered uncommits are exempt from cycles, executed calls with non-success
status can be no-progress retries, and the inspected-item set/requirement
versions are explicit state rather than cycle-breaking marks.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

VERSION = "progress-diagnostic-v1"
WINDOW = 6
MASK_CAP = 8
LATENT_STALL_RUN = 2
CLASSES = ("rejected", "useful_inspection", "solver_work", "triggered_revision", "short_cycle", "idempotent",
           "other_progress", "terminal")
NO_PROGRESS = ("short_cycle", "idempotent")
CODES = {"rejected": "R", "useful_inspection": "I", "solver_work": "S", "triggered_revision": "V",
         "short_cycle": "C", "idempotent": "D", "other_progress": "O", "terminal": "T"}
DECODE = {v: k for k, v in CODES.items()}
NEVER_MASK = ("verify", "abstain")
FRESH_HANDLE_KINDS = ("start_subset", "start_assign", "build_route")
RECORD_HANDLE_KINDS = ("retrieve", "use_return")
TRIGGER_STATUSES = ("infeasible", "timeout", "unknown")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def record_content(record: dict) -> tuple:
    """Content identity of a public computation record (never its handle)."""
    return (record.get("primitive"), record.get("problem_snapshot"), record.get("depends_on"), record.get("budget"),
            record.get("status"))


def plan_context(o) -> tuple:
    """The public plan state a call is made under: committed selection/assignment and all
    requirement versions (the dependencies a stage's result is judged against)."""
    return (o.selection_id, o.assignment_id, tuple(sorted(o.requirement_versions.items())))


def record_key(record: dict, context: tuple | None = None) -> str:
    """Record identity: content plus the plan context it was computed under."""
    return _digest([record_content(record), context])


def record_inputs(record: dict) -> str:
    """(primitive, problem, depends_on, plan context): the inputs of the call behind a record."""
    return _digest([record.get("primitive"), record.get("problem_snapshot"), record.get("depends_on"),
                    record.get("context")])


def draft_key(entry: dict) -> str:
    return _digest([entry.get("primitive"), entry.get("problem"), entry.get("depends_on")])


def possibly_stochastic(record: dict) -> bool:
    """Public evidence that a result may not repeat: a wall-clock limited timeout (work
    below the budget) or an unknown status. A worker killed at its deadline reports
    work_units == budget and is indistinguishable from a work-limited timeout."""
    status = record.get("status")
    return status == "unknown" or (status == "timeout" and record.get("work_units", 0) < record.get("budget", 0))


def signature(o, *, _records=None, _drafts=None) -> tuple:
    """The relevant-state signature of a public observation (see module docstring)."""
    # Without the tracker's record identities (which add the plan context each record was
    # computed under), records are keyed by content alone.
    records = _records if _records is not None else {r["handle"]: record_key(r) for r in o.records}
    drafts = _drafts if _drafts is not None else {name: draft_key(e) for name, e in o.problems.items()}
    return (tuple(sorted(drafts.items())),
            tuple(sorted(o.pending)), tuple(sorted(o.pending_assignment.items())),
            o.selection_id, o.assignment_id, bool(o.verified),
            frozenset(records.values()),
            frozenset(records[h] for h in o.retrieved if h in records),
            o.position,
            tuple(sorted(o.requirement_versions.items())),
            tuple((e.get("kind"), json.dumps(e.get("argument")), e.get("affected"), tuple(e.get("revoked") or ()))
                  for e in o.events),
            o.requirements is not None, o.known_edges is not None, frozenset(o.known_items))


@dataclass(frozen=True)
class StepClass:
    step: int
    cls: str
    no_progress: bool
    cycle_length: int | None
    changed: bool
    reason: str | None = None
    repeat_rejection: bool = False
    latent_stall: bool = False
    event: bool = False
    flagged: bool = False

    @property
    def code(self) -> str:
        return CODES[self.cls]


class ProgressTracker:
    """Incremental, online progress diagnostic for one episode.

    tracker = ProgressTracker(initial_observation)
    step_class = tracker.update(observation_after_step, action)   # feedback defaults to observation.feedback
    tracker.flagged_keys()          # canonical action keys masked at the current signature
    tracker.mask(actions)           # per-candidate bool list for the current observation
    tracker.copy()                  # independent copy (branching)
    """

    def __init__(self, observation, *, window: int = WINDOW, mask_cap: int = MASK_CAP,
                 mask_repeat_rejections: bool = False, verify: bool = False):
        self.window, self.mask_cap = window, mask_cap
        self.mask_repeat_rejections = mask_repeat_rejections
        self.verify = verify
        self._records: dict[str, str] = {}        # record handle -> identity (content + plan context)
        self._record_objs: dict[str, dict] = {}   # identity -> a public record (inputs/status/budget/context)
        self._drafts: dict[str, str] = {}         # draft name -> draft key
        self._obs = observation
        self._sig = self._signature(observation, (), plan_context(observation))
        self.sigs = [self._sig]       # signature after k steps
        self.marks = [False]          # marks[k]: step k was a progress mark
        self.classes: list[str] = []
        self.cycle_lengths: list[int] = []
        self.masks: dict[tuple, set] = {}
        self._rejected: set = set()
        self._think_run = 0
        self.latent_stalls = 0
        self.repeat_rejections = 0
        self.flags = 0
        self.interventions: list[list] = []
        # triggers since the last commitment of each stage (select / assign)
        self._triggered = {"select": False, "assign": False}

    # --- signature -----------------------------------------------------------
    def _signature(self, o, touched, context) -> tuple:
        """Incremental signature. New records get the plan context of the state the call was
        made in (``context``: the pre-step state); only drafts named by the action change."""
        for r in o.records:
            h = r["handle"]
            if h not in self._records:
                key = record_key(r, context)
                self._records[h] = key
                obj = {k: r.get(k) for k in ("primitive", "problem_snapshot", "depends_on", "budget", "status",
                                             "work_units")}
                self._record_objs.setdefault(key, {**obj, "context": context})
        for name, entry in o.problems.items():
            if name not in self._drafts or name in touched:
                self._drafts[name] = draft_key(entry)
        if len(self._drafts) != len(o.problems):
            self._drafts = {n: self._drafts[n] if n in self._drafts else draft_key(e) for n, e in o.problems.items()}
        sig = signature(o, _records=self._records, _drafts=self._drafts)
        if self.verify:
            full = {name: draft_key(e) for name, e in o.problems.items()}
            assert sig == signature(o, _records=self._records, _drafts=full), "incremental draft keys drifted"
        return sig

    @property
    def current_signature(self) -> tuple:
        return self._sig

    # --- canonical action keys -------------------------------------------------
    def canonical_key(self, action) -> str:
        kind, args = action.kind, dict(action.arguments)
        if kind in FRESH_HANDLE_KINDS:
            args.pop("handle", None)
        elif kind in RECORD_HANDLE_KINDS and args.get("handle") in self._records:
            args["handle"] = "content:" + self._records[args["handle"]]
        return _digest({"kind": kind, "arguments": args})[:16]

    def flagged_keys(self) -> set:
        """Canonical action keys flagged as stagnant at the current signature (R-mask)."""
        return set(self.masks.get(self._sig, ()))

    def is_flagged_state(self) -> bool:
        return bool(self.masks.get(self._sig))

    def mask(self, actions) -> list[bool]:
        """True = masked, for each candidate action at the current observation."""
        keys = self.masks.get(self._sig)
        if not keys:
            return [False] * len(actions)
        return [a.kind not in NEVER_MASK and self.canonical_key(a) in keys for a in actions]

    # --- update --------------------------------------------------------------
    def update(self, observation, action, feedback: dict | None = None) -> StepClass:
        pre, pre_sig = self._obs, self._sig
        fb = observation.feedback if feedback is None else feedback
        kind, args = action.kind, action.arguments
        t = len(self.sigs)
        canon = self.canonical_key(action)  # handles resolved against the pre-step records
        pre_records = frozenset(self._records.values())
        touched = tuple(x for x in (args.get("problem"), args.get("handle")) if isinstance(x, str))
        sig = self._signature(observation, touched, plan_context(pre))
        status = fb.get("status")
        executed_call = kind == "call" and "return" in fb
        event = "event" in fb
        changed = sig != pre_sig
        cls, reason, cycle, repeat_rej, stall = None, None, None, False, False
        mark = event

        if kind == "abstain":
            cls = "terminal"
        elif not executed_call and status != "success":
            cls, reason = "rejected", fb.get("reason")
            repeat_rej = (pre_sig, canon) in self._rejected
            self._rejected.add((pre_sig, canon))
            self.repeat_rejections += repeat_rej
        elif kind == "inspect" and self._new_information(pre, args.get("target")):
            cls, mark = "useful_inspection", True
        elif executed_call and self._solver_work(fb.get("return"), pre_records):
            cls, mark = "solver_work", True
        elif self._triggered_revision(pre, observation, action, status):
            cls = "triggered_revision"
        else:
            cycle = self._cycle_length(sig, t)
            if cycle == 1:
                cls = "idempotent"
            elif cycle is not None:
                cls = "short_cycle"
            else:
                cls = "other_progress"

        # latent stall: > 2 consecutive thinks with unchanged signature
        if kind == "think" and cls == "idempotent":
            self._think_run += 1
            stall = self._think_run > LATENT_STALL_RUN
            self.latent_stalls += stall
        else:
            self._think_run = 0

        # triggers for uncommit, collected since the revised piece's commitment (a commit
        # resets them first; an event fired by the committing step itself still counts)
        if status == "success" and "selection_id" in fb:
            self._triggered["select"] = self._triggered["assign"] = False
        if status == "success" and "assignment_id" in fb:
            self._triggered["assign"] = False
        if cls == "rejected" or event or (executed_call and status in TRIGGER_STATUSES) or (
                kind == "retrieve" and status == "success" and cls not in NO_PROGRESS):
            self._triggered = {"select": True, "assign": True}

        flagged = False
        no_progress = cls in NO_PROGRESS
        if (no_progress or (repeat_rej and self.mask_repeat_rejections)) and kind not in NEVER_MASK:
            keys = self.masks.setdefault(pre_sig, set())
            if canon not in keys and len(keys) < self.mask_cap:
                keys.add(canon)
                flagged = True
                self.flags += 1
        self.sigs.append(sig)
        self.marks.append(mark)
        self.classes.append(cls)
        if cycle is not None and no_progress:
            self.cycle_lengths.append(cycle)
        self._obs, self._sig = observation, sig
        return StepClass(t, cls, no_progress, cycle if no_progress else None, changed, reason, repeat_rej, stall,
                         event, flagged)

    def _new_information(self, pre, target) -> bool:
        if target == "requirements":
            return pre.requirements is None
        if target == "map":
            return pre.known_edges is None
        return target not in pre.known_items

    def _solver_work(self, handle, pre_records) -> bool:
        key = self._records.get(handle)
        if key is None:
            return False
        rec = self._record_objs[key]
        if key in pre_records:
            # same inputs, plan context, budget and status as an earlier record: a retry;
            # justified only if the earlier result was possibly stochastic
            return possibly_stochastic(rec)
        inputs = record_inputs(rec)
        earlier = [r for k, r in self._record_objs.items() if k in pre_records and record_inputs(r) == inputs]
        if not earlier:
            return True  # new problem, or a dependency / plan change since the last such call
        return rec.get("budget", 0) > max(r.get("budget", 0) for r in earlier) or any(
            possibly_stochastic(r) and r.get("budget") == rec.get("budget") for r in earlier)

    def _triggered_revision(self, pre, post, action, status) -> bool:
        if status != "success":
            return False
        if action.kind == "uncommit":
            target = action.arguments.get("target")
            return self._triggered["select" if target == "select" else "assign"]
        if action.kind == "add_constraint":
            name = action.arguments.get("problem")
            before, after = pre.problems.get(name), post.problems.get(name)
            return before is not None and after is not None and before.get("depends_on") != after.get("depends_on")
        return False

    def _cycle_length(self, sig, t) -> int | None:
        # Compare with the states after steps t-1 .. t-W (0 = episode start). A progress mark
        # at step j+1 (useful inspection, solver work, event) separates states <= j from t.
        # The current step is not a mark here (marks take precedence), except an event,
        # which bumps a requirement version and so can never match an earlier state.
        for j in range(t - 1, max(0, t - self.window) - 1, -1):
            if j + 1 < t and self.marks[j + 1]:
                break
            if self.sigs[j] == sig:
                return t - j
        return None

    # --- bookkeeping -----------------------------------------------------------
    def note_intervention(self, step: int, kind: str, masked: int = 0, changed: bool = False) -> None:
        self.interventions.append([step, kind, masked, int(changed)])

    def copy(self) -> "ProgressTracker":
        other = copy(self)
        for name in ("_records", "_record_objs", "_drafts", "_triggered"):
            setattr(other, name, dict(getattr(self, name)))
        for name in ("sigs", "marks", "classes", "cycle_lengths"):
            setattr(other, name, list(getattr(self, name)))
        other.masks = {k: set(v) for k, v in self.masks.items()}
        other._rejected = set(self._rejected)
        other.interventions = [list(x) for x in self.interventions]
        return other

    def summary(self) -> dict[str, Any]:
        """Compact per-episode record (stored in evaluator rows)."""
        counts = {c: 0 for c in CLASSES}
        for c in self.classes:
            counts[c] += 1
        return {"version": VERSION, "window": self.window, "classes": "".join(CODES[c] for c in self.classes),
                "counts": counts, "no_progress": sum(counts[c] for c in NO_PROGRESS),
                "cycle_lengths": list(self.cycle_lengths), "latent_stalls": self.latent_stalls,
                "repeat_rejections": self.repeat_rejections, "flags": self.flags,
                "interventions": [list(x) for x in self.interventions]}


def classify_episode(observations, actions, **kwargs) -> ProgressTracker:
    """Offline: observations[0] is the initial observation, observations[k] after actions[k-1]."""
    tracker = ProgressTracker(observations[0], **kwargs)
    for o, a in zip(observations[1:], actions):
        tracker.update(o, a)
    return tracker
