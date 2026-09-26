"""Track C telemetry d_t (extended-04 design Track C; v2 revisions 5-8; review F8, F11).

d_t is a measured, immutable, versioned record of **public** facts available at
decision t, before the step-t action is chosen. Every input comes from one of:

- the public ``DepObservation`` objects of the episode (never the hidden spec,
  the evaluator, the solver or any RNG);
- the actor's action distribution at the current observation (``PolicyStats``,
  computed by the runner from the frozen actor's logits under ``no_grad``;
  numbers only, so no gradient can reach the actor trunk);
- the progress diagnostic's public state (``campaign04_progress.ProgressTracker``);
- the agent's own control history (the interventions it chose) and the compute
  units the deployment itself charged for actor forwards.

Therefore identical visible histories give identical telemetry
(``tests/test_campaign04_meta.py``). Labels, outcomes and utilities are never
inputs. ``public_cost`` reconstructs the environment's cumulative cost from public
facts only (tested equal to ``DepWorkshop.evaluate()["cost"]`` minus the
metacontrol charge, which is excluded on purpose so that charging or not
charging the metacontroller never changes behaviour).

Feature layout (``LAYOUT``; fixed scaling, no data-dependent normalization). Each
entry is (group, name). Groups can be zeroed at evaluation (reliance) or dropped
in training (necessity): see ``GROUPS`` / ``group_mask``.

policy      p_top1, entropy/log(n), margin (p_top1 - p_top2), log1p(n)/6, p_default
budget      steps_used/limit, steps_remaining/limit, work_remaining/work_0,
            travel_remaining/travel_0, remaining_steps/96
cost        public_cost*10, compute_units/100, calls/10, inspections/20
diagnostic  last class one-hot (7 non-terminal classes + none), per-class counts
            (log1p/4, 7), flagged-state bit, masked keys/8, default != greedy bit,
            no-progress run/6, latent stalls/4, repeat rejections/4
feedback    last status one-hot (STATUSES), last reason one-hot (REASONS + none + other)
info_events new-information flag (last step added to the availability set, a new
            record or an event), event this step, events so far, any revocation,
            requirements_version/5, requirements known, map known, items known fraction
stage       selection committed, assignment committed, verified, pending items /
            categories, pending assignment non-empty, at destination, drafts/10,
            records/10, retrieved/10
control     last intervention one-hot (INTERVENTIONS + none), interventions/10

Torch-free, standard library only.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math

from .campaign03_depworld import REASONS
from .campaign04_progress import CLASSES, NO_PROGRESS

TELEMETRY_VERSION = "c-telemetry-v1"
INTERVENTIONS = ("default", "sample", "mask_top", "stop")
STATUSES = ("ready", "success", "rejected", "invalid_input", "timeout", "infeasible", "incomplete",
            "unavailable_resource", "invalid", "unknown", "other")
STEP_CLASSES = tuple(c for c in CLASSES if c != "terminal")
REASON_SLOTS = REASONS + ("none", "other")


def _layout():
    out = []
    out += [("policy", n) for n in ("p_top1", "entropy_norm", "margin", "log_candidates", "p_default")]
    out += [("budget", n) for n in ("steps_used", "steps_remaining", "work_remaining", "travel_remaining",
                                    "steps_remaining_abs")]
    out += [("cost", n) for n in ("public_cost", "compute_units", "calls", "inspections")]
    out += [("diagnostic", f"class_last_{c}") for c in STEP_CLASSES + ("none",)]
    out += [("diagnostic", f"count_{c}") for c in STEP_CLASSES]
    out += [("diagnostic", n) for n in ("flagged_state", "masked_keys", "default_not_greedy", "no_progress_run",
                                        "latent_stalls", "repeat_rejections")]
    out += [("feedback", f"status_{s}") for s in STATUSES]
    out += [("feedback", f"reason_{r}") for r in REASON_SLOTS]
    out += [("info_events", n) for n in ("new_information", "event_now", "events", "revoked",
                                         "requirements_version", "requirements_known", "map_known",
                                         "items_known")]
    out += [("stage", n) for n in ("selection_committed", "assignment_committed", "verified", "pending_items",
                                   "pending_assignment", "at_destination", "drafts", "records", "retrieved")]
    out += [("control", f"u_last_{u}") for u in INTERVENTIONS + ("none",)]
    out += [("control", "interventions")]
    return tuple(out)


LAYOUT = _layout()
DIM = len(LAYOUT)
GROUPS = tuple(dict.fromkeys(g for g, _ in LAYOUT))
INDEX = {name: i for i, (_, name) in enumerate(LAYOUT)}
if len(INDEX) != DIM:
    raise AssertionError("telemetry feature names must be unique")


def group_indices(group: str) -> list[int]:
    if group not in GROUPS:
        raise ValueError(f"unknown telemetry group {group!r}")
    return [i for i, (g, _) in enumerate(LAYOUT) if g == group]


def group_mask(drop=()) -> list[float]:
    """1.0 for kept features, 0.0 for the dropped groups' features."""
    dropped = {i for g in drop for i in group_indices(g)}
    return [0.0 if i in dropped else 1.0 for i in range(DIM)]


@dataclass(frozen=True)
class PolicyStats:
    """The actor's distribution at the current observation (valid candidates only)."""
    p_top1: float
    entropy: float
    margin: float
    candidates: int
    p_default: float
    default_not_greedy: bool
    masked: int
    flagged_state: bool


def policy_stats(probabilities, default_index: int, greedy_index: int, masked: int, flagged_state: bool) -> PolicyStats:
    """probabilities: the float64 softmax over the valid candidates (a list)."""
    n = len(probabilities)
    ordered = sorted(probabilities, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if n > 1 else 0.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return PolicyStats(float(top1), float(entropy), float(top1 - top2), n, float(probabilities[default_index]),
                       default_index != greedy_index, int(masked), bool(flagged_state))


@dataclass(frozen=True)
class Telemetry:
    version: str
    step: int
    values: tuple

    def vector(self) -> list[float]:
        return list(self.values)

    def get(self, name: str) -> float:
        return self.values[INDEX[name]]


def public_cost(o, inspections: int, work_used: int, compute_units: float) -> float:
    """Cumulative cost from public facts (same expression as DepWorkshop._cost)."""
    p = o.prices
    return (o.step * p["action"] + o.travel * p["travel"] + inspections * p["observation"]
            + work_used * p["work"] + compute_units * p["compute"])


def _status(fb) -> str:
    s = fb.get("status")
    return s if s in STATUSES else "other"


def _reason(fb) -> str:
    r = fb.get("reason")
    if r is None:
        return "none"
    return r if r in REASONS else "other"


class TelemetryRecorder:
    """Running public bookkeeping for one episode.

    rec = TelemetryRecorder(initial_observation)
    d_t = rec.features(observation_t, tracker, stats)       # before choosing the step-t action
    rec.update(observation_after, action, step_class, u, compute_units)   # after the step
    """

    def __init__(self, observation):
        o = observation
        self.step_limit = o.step + o.remaining_steps
        self.work0 = o.remaining_work
        self.travel0 = max(1, o.remaining_travel)
        self.items = max(1, len(o.item_inventory))
        self.categories = max(1, len(o.goal.get("categories", ())) or 1)
        self.counts = {c: 0 for c in STEP_CLASSES}
        self.last_class = None
        self.np_run = 0
        self.inspections = 0
        self.calls = 0
        self.compute_units = 0.0   # actor/task compute charged by the deployment (metacontrol excluded)
        self.last_u = None
        self.interventions = 0
        self.new_information = False
        self.event_now = False
        self.revoked = False

    def copy(self) -> "TelemetryRecorder":
        other = copy(self)
        other.counts = dict(self.counts)
        return other

    def public_cost(self, o) -> float:
        return public_cost(o, self.inspections, self.work0 - o.remaining_work, self.compute_units)

    def features(self, o, tracker, stats: PolicyStats) -> Telemetry:
        v = [0.0] * DIM

        def put(name, value):
            v[INDEX[name]] = float(value)

        n = max(1, stats.candidates)
        put("p_top1", stats.p_top1)
        put("entropy_norm", stats.entropy / math.log(n) if n > 1 else 0.0)
        put("margin", stats.margin)
        put("log_candidates", math.log1p(stats.candidates) / 6)
        put("p_default", stats.p_default)
        limit = max(1, self.step_limit)
        put("steps_used", o.step / limit)
        put("steps_remaining", o.remaining_steps / limit)
        put("work_remaining", o.remaining_work / self.work0 if self.work0 else 0.0)
        put("travel_remaining", o.remaining_travel / self.travel0)
        put("steps_remaining_abs", o.remaining_steps / 96)
        put("public_cost", 10 * self.public_cost(o))
        put("compute_units", self.compute_units / 100)
        put("calls", self.calls / 10)
        put("inspections", self.inspections / 20)
        put(f"class_last_{self.last_class or 'none'}", 1.0)
        for c in STEP_CLASSES:
            put(f"count_{c}", math.log1p(self.counts[c]) / 4)
        put("flagged_state", stats.flagged_state)
        put("masked_keys", stats.masked / 8)
        put("default_not_greedy", stats.default_not_greedy)
        put("no_progress_run", self.np_run / 6)
        put("latent_stalls", (tracker.latent_stalls if tracker is not None else 0) / 4)
        put("repeat_rejections", (tracker.repeat_rejections if tracker is not None else 0) / 4)
        fb = o.feedback or {}
        put(f"status_{_status(fb)}", 1.0)
        put(f"reason_{_reason(fb)}", 1.0)
        put("new_information", self.new_information)
        put("event_now", self.event_now)
        put("events", len(o.events))
        put("revoked", self.revoked)
        put("requirements_version", o.requirements_version / 5)
        put("requirements_known", o.requirements is not None)
        put("map_known", o.known_edges is not None)
        put("items_known", len(o.known_items) / self.items)
        put("selection_committed", o.selection_id is not None)
        put("assignment_committed", o.assignment_id is not None)
        put("verified", o.verified)
        put("pending_items", len(o.pending) / self.categories)
        put("pending_assignment", bool(o.pending_assignment))
        put("at_destination", o.position == o.goal.get("destination"))
        put("drafts", len(o.problems) / 10)
        put("records", len(o.records) / 10)
        put("retrieved", len(o.retrieved) / 10)
        put(f"u_last_{self.last_u or 'none'}", 1.0)
        put("interventions", self.interventions / 10)
        return Telemetry(TELEMETRY_VERSION, o.step, tuple(v))

    def update(self, after, action, step_class, u: str, compute_units: float) -> None:
        """After the step: ``step_class`` is the tracker's StepClass (or None), ``u`` the
        intervention that chose the action, ``compute_units`` the task/actor compute the
        deployment charged for this step (metacontrol charges excluded)."""
        if u not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {u!r}")
        fb = after.feedback or {}
        cls = step_class.cls if step_class is not None else None
        if cls in self.counts:
            self.counts[cls] += 1
        self.last_class = cls if cls in self.counts else None
        self.np_run = self.np_run + 1 if cls in NO_PROGRESS else 0
        if action.kind == "inspect" and fb.get("status") == "success":
            self.inspections += 1
        if action.kind == "call" and "return" in fb:
            self.calls += 1
        self.compute_units += compute_units
        self.event_now = "event" in fb
        self.revoked = self.revoked or bool((fb.get("event") or {}).get("revoked"))
        new_record = action.kind == "retrieve" and fb.get("status") == "success" and cls not in NO_PROGRESS
        self.new_information = cls in ("useful_inspection", "solver_work") or new_record or self.event_now
        self.last_u = u
        self.interventions += u != "default"
