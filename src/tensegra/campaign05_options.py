"""Track A headroom tooling (A-HR) for extended-05 (design §1 + v2 revisions 1-4, 8; registry A-HR).

Deployment contract D
---------------------
D = the extended-04 "greedy + R-mask" deployment (``campaign04_deploy`` mode
``r_mask``), bit-for-bit: the frozen actor scores the public candidate set; the
progress-diagnostic-v1 R-mask removes flagged stagnant action keys at the current
relevant signature (verify/abstain never masked; all-masked falls back to the
unmasked argmax); greedy argmax with ties to the lowest candidate index (torch
``argmax`` returns the first maximal index); one actor forward charged per decision;
the tracker updates after each step; the episode stops at verify success, abstain
or the decision cap (96 = the actors' ``max_steps``). ``run_policy`` with the
``d_choice`` chooser is tested identical to ``deploy_episodes(mode="r_mask")`` and to
``campaign04_meta.run(FixedAgent("r_mask"))`` (actions, utility, tracker summary).
Registered device: CPU, 1 thread, fp32 (as extended-04 Track C) for every A-HR job.

Decision-anchored option set O(I) (public; ``option_set``; design v2 revision 3)
------------------------------------------------------------------------------
Computed from the public observation, the public candidate catalog (the candidate
list IS ``action_catalog(o)``) and D's choice (a function of public information and
the frozen actor). Every option is a candidate index. The anchor is D's action kind:

- ``call`` (D chooses ``call(p, b)``): ``budget`` = the same problem at every other
  allowed budget; ``call_other`` = every allowed budget of every *other open* problem;
  ``not_call`` = D's best non-call candidate (R-mask masked argmax over the candidates
  that are neither calls nor abstain; review F3 "not calling");
- ``reuse_recompute`` (D chooses ``use_return``, ``retrieve``, ``start_subset``,
  ``start_assign`` or ``build_route``; primitive P = the primitive D's action serves):
  ``reuse`` = ``use_return(h, use(P))`` for every type-applicable record h of P (record
  primitive = P, publicly usable status + valid certificate) that is retrieved;
  ``reuse_retrieve`` = ``retrieve(h)`` for such records not yet retrieved (reuse's
  first step: use_return of an unretrieved record is rejected); ``recompute`` = the
  ``start_*``/``build_route`` of P (a fresh draft);
- ``commit_revise`` (D chooses ``commit_pending``, ``commit_assignment`` or ``uncommit``):
  ``revise_assign`` / ``revise_select`` = ``uncommit(assign)`` / ``uncommit(select)``
  when that target is committed (otherwise a rejected no-op);
- ``other`` (any other D action): ``call_now`` = every allowed budget of every open
  problem (a problem is open when its primitive's stage is uncommitted, public:
  constrained_subset while no selection is committed; csp while a selection is
  committed and no assignment; shortest_path while an assignment is committed and
  the agent is not at the destination);
- ``abstain``: always, its own class (never pooled into G1; revision 1).

D's own choice is always included (``is_d``). Duplicates are merged (first type
kept, all types recorded). ``mode="full"`` is the entire candidate catalog (types by
action kind; ``in_oi`` marks the anchored options), used on a subsample only.

State sampling (public, online; ``HRConfig``)
---------------------------------------------
At every D decision one uniform is drawn from the per-world stream
``("e05-hr-sample", seed)``; the decision is a *sampled point* if the uniform is below
the anchor-stratum probability (call .35, reuse_recompute .35, commit_revise .5,
other .08) and fewer than 8 points were sampled earlier in the episode. The rule
uses public information only, so the single-deviation policy class (deviate at most
once, at a sampled point, then D; revision 2) is deployable. In addition, a 1/16
subsample of worlds (hash ``("e05-hr-all", seed)``, the same worlds for every base)
branches *every* decision state (hindsight-headroom distribution; not used by the
single-deviation estimator except through its sampled points).

Branch labels
-------------
Q^D(I_t, a) = U_final(branch) − U_t: clone the environment (sharing the exact solver
cache), the actor's recurrent state (if any) and the progress tracker; charge the
step-t actor forward; take a; follow D to termination (remaining decision cap).
U_t = ``DepWorkshop.current_utility()`` before the step-t charges, U_final =
``evaluate()["utility"]`` (the evaluation utility; every charge lands exactly once).
For D's own choice the main D line after t IS the continuation, so Q(I_t, D(I_t)) is
taken from the main line (a registered fraction of points also branches it and
checks equality). These are **hidden-state (true-world) labels**: an optimistic,
non-deployable privilege of offline analysis, never an input at deployment.

Multi-step option class (A-HR2; registry A-HR2; ``option_mode="multi"``)
-----------------------------------------------------------------------
Same worlds, sampler, points, main line and labels as A-HR; the option set at a point
is {D, ``delegate``} instead of O(I):

- ``delegate`` (option T): at the point a fresh public teacher instance
  (``make_reference("dep_reuse")`` = ``DepReference("reuse")``, stateless: it reads only
  the public ``DepObservation`` and the public catalog) chooses the step-t action and
  every following action until the first step whose feedback is a *successful commit*
  (the environment's own completion rule: status success with a new selection_id or
  assignment_id, i.e. ``commit_pending`` / ``commit_assignment`` / a committing
  ``use_return``) or the episode ends (verify success, abstain, step limit), or 12
  teacher steps (incl. step t) have been taken; then D continues to termination.
  The progress tracker is updated on every step (teacher steps included), so D's
  R-mask after the delegation sees the whole history. Every decision (teacher or D) is
  charged one controller forward (``neural_work_per_forward``; the extended-04
  reference tariff 1.0), so T with a teacher that happens to pick D's actions IS the
  main line. A teacher action outside the catalog is never repaired: the delegation
  ends and D takes that decision (counted; at step t the option is then unavailable).
- Budget override (option B: for the next solver call use b in {16, 1024, remaining}
  at points where D calls or at call_now points, then D) is **identical to existing
  A-HR options**: at a D-call point it is ``call(p, b)`` then D = the ``budget`` option;
  at an ``other`` point it is a ``call_now`` option (every allowed budget, incl. the
  remaining-work budget, of every open problem). ``budget_override_options`` lists
  those indices; they are always a subset of O(I), so B is not re-branched (its
  hindsight value is read from the A-HR labels by the analysis).

Torch is imported lazily (actor forwards only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time
from typing import Any, Callable

from .campaign03_depworld import PRIMITIVE_OF_USE, USE_OF_PRIMITIVE, record_usable, relations
from .campaign04_branch import clone_branch
from .campaign04_progress import ProgressTracker

HR_VERSION = "e05-hr-v2"
OPTION_SET_VERSION = "e05-oi-anchored-v1"
ANCHORS = ("call", "reuse_recompute", "commit_revise", "other")
OPTION_TYPES = ("budget", "call_other", "not_call", "reuse", "reuse_retrieve", "recompute", "revise_assign",
                "revise_select", "call_now", "abstain")
CONTINUATION = "D = greedy + R-mask (campaign04_deploy r_mask semantics, progress-diagnostic-v1), cap 96, CPU"
MULTI_VERSION = "e05-hr2-v1"
TEACHER = "dep_reuse"
DELEGATE_MAX_STEPS = 12
B_BUDGETS = (16, 1024, "remaining")
RECOMPUTE_OF = {"start_subset": "constrained_subset", "start_assign": "csp", "build_route": "shortest_path"}
START_OF = {v: k for k, v in RECOMPUTE_OF.items()}


def digest_seed(*parts) -> int:
    return int(hashlib.sha256(json.dumps(list(parts), separators=(",", ":")).encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# Option set O(I) (public only)
# ---------------------------------------------------------------------------

def stage_open(o) -> dict[str, bool]:
    """Which stages are uncommitted, from public state only."""
    select_committed = o.selection_id is not None
    assign_committed = o.assignment_id is not None
    at_destination = o.position == o.goal.get("destination")
    return {"constrained_subset": not select_committed,
            "csp": select_committed and not assign_committed,
            "shortest_path": assign_committed and not at_destination}


def action_type(action) -> str:
    """Type of any catalog action by kind (full mode, and D's own action)."""
    k, a = action.kind, action.arguments
    if k == "call":
        return "call"
    if k == "use_return":
        return "reuse"
    if k == "retrieve":
        return "retrieve"
    if k in RECOMPUTE_OF:
        return "recompute"
    if k == "uncommit":
        return "revise_assign" if a.get("target") == "assign" else "revise_select"
    if k == "abstain":
        return "abstain"
    return f"other:{k}"


def anchor_of(o, action) -> tuple[str, str | None]:
    """(anchor kind, primitive for reuse_recompute) of D's action."""
    k, a = action.kind, action.arguments
    if k == "call":
        return "call", None
    if k == "use_return":
        return "reuse_recompute", PRIMITIVE_OF_USE.get(a.get("as"))
    if k == "retrieve":
        r = next((r for r in o.records if r["handle"] == a.get("handle")), None)
        return "reuse_recompute", None if r is None else r["primitive"]
    if k in RECOMPUTE_OF:
        return "reuse_recompute", RECOMPUTE_OF[k]
    if k in ("commit_pending", "commit_assignment", "uncommit"):
        return "commit_revise", None
    return "other", None


def option_set(o, actions, d_index: int, mode: str = "anchored", scores=None, masked=None) -> list[dict[str, Any]]:
    """[{index, type, types, is_d, anchor}] over candidate indices, sorted by index. Public only:
    ``o`` is the public observation, ``actions`` the public catalog, ``d_index`` D's choice,
    ``scores``/``masked`` the actor's logits and D's R-mask for the candidates (for ``not_call``)."""
    if mode not in ("anchored", "full"):
        raise ValueError(mode)
    d_action = actions[d_index]
    anchor, prim = anchor_of(o, d_action)
    found: dict[int, list[str]] = {}

    def add(i, t):
        found.setdefault(i, [])
        if t not in found[i]:
            found[i].append(t)

    if mode == "full":
        for i, a in enumerate(actions):
            add(i, action_type(a))
    else:
        open_ = stage_open(o)
        problems = o.problems
        records = {r["handle"]: r for r in o.records}
        if anchor == "call":
            p0 = d_action.arguments.get("problem")
            for i, a in enumerate(actions):
                if a.kind != "call" or i == d_index:
                    continue
                p = a.arguments.get("problem")
                if p == p0:
                    add(i, "budget")
                elif p in problems and open_.get(problems[p]["primitive"], False):
                    add(i, "call_other")
            if scores is not None:
                allowed = [i for i, a in enumerate(actions) if a.kind not in ("call", "abstain")]
                unmasked = [i for i in allowed if not (masked and masked[i])] or allowed
                if unmasked:
                    add(max(unmasked, key=lambda i: (float(scores[i]), -i)), "not_call")
        elif anchor == "reuse_recompute" and prim is not None:
            use = USE_OF_PRIMITIVE[prim]
            for i, a in enumerate(actions):
                if i == d_index:
                    continue
                k, arg = a.kind, a.arguments
                if k == "use_return" and arg.get("as") == use:
                    r = records.get(arg.get("handle"))
                    if r is not None and r["primitive"] == prim and record_usable(r) and r["handle"] in o.retrieved:
                        add(i, "reuse")
                elif k == "retrieve":
                    r = records.get(arg.get("handle"))
                    if r is not None and r["primitive"] == prim and record_usable(r) and r["handle"] not in o.retrieved:
                        add(i, "reuse_retrieve")
                elif k == START_OF[prim]:
                    add(i, "recompute")
        elif anchor == "commit_revise":
            for i, a in enumerate(actions):
                if a.kind == "uncommit" and i != d_index:
                    if a.arguments.get("target") == "assign" and o.assignment_id is not None:
                        add(i, "revise_assign")
                    elif a.arguments.get("target") == "select" and o.selection_id is not None:
                        add(i, "revise_select")
        elif anchor == "other":
            for i, a in enumerate(actions):
                if a.kind == "call" and i != d_index:
                    p = problems.get(a.arguments.get("problem"))
                    if p is not None and open_.get(p["primitive"], False):
                        add(i, "call_now")
        for i, a in enumerate(actions):
            if a.kind == "abstain" and i != d_index:
                add(i, "abstain")
    found.pop(d_index, None)
    out = [{"index": i, "type": ts[0], "types": list(ts), "is_d": False, "anchor": anchor}
           for i, ts in found.items()]
    out.append({"index": d_index, "type": "D", "types": ["D", action_type(d_action)], "is_d": True, "anchor": anchor})
    return sorted(out, key=lambda x: x["index"])


def option_features(o, actions, options, logits_row, probabilities, d_index) -> list[dict[str, float]]:
    """Public per-option features for the same-information regression (actor numbers,
    budget facts, public record applicability relations). Never labels."""
    n = len(actions)
    d_logit = float(logits_row[d_index])
    order = sorted(range(n), key=lambda i: (-float(logits_row[i]), i))
    rank = {i: r for r, i in enumerate(order)}
    records = {r["handle"]: r for r in o.records}
    by_problem: dict[str, list] = {}
    for r in o.records:
        by_problem.setdefault(r.get("problem"), []).append(r)
    latest = {}
    for h, p in o.problems.items():
        latest[p["primitive"]] = max(latest.get(p["primitive"], -1), p.get("created_step", 0))
    d_action = actions[d_index]
    d_budget = d_action.arguments.get("budget") if d_action.kind == "call" else None
    out = []
    for opt in options:
        i = opt["index"]
        a = actions[i]
        f = {"prob": float(probabilities[i]), "logit_gap": float(logits_row[i]) - d_logit,
             "rank_frac": rank[i] / max(1, n - 1), "is_d": float(opt["is_d"])}
        if a.kind == "call":
            b = a.arguments["budget"]
            p = o.problems.get(a.arguments["problem"], {})
            prev = by_problem.get(a.arguments["problem"], [])
            f.update({"budget_log": math.log2(b) / 12, "budget_frac": b / max(1, o.remaining_work),
                      "budget_vs_d_log": (math.log2(b) - math.log2(d_budget)) / 12 if d_budget else 0.0,
                      "problem_latest": float(p.get("created_step", -1) == latest.get(p.get("primitive"), -2)),
                      "prev_calls": len(prev) / 4,
                      "prev_timeouts": sum(r["status"] == "timeout" for r in prev) / 4,
                      "prev_max_budget_log": math.log2(max([r.get("budget") or 1 for r in prev] or [1])) / 12})
        elif a.kind in ("use_return", "retrieve"):
            r = records.get(a.arguments["handle"])
            rel = relations(o, r, r["primitive"])
            f.update({f"rel_{k}": float(v) for k, v in rel.items()})
            f.update({"rec_timeout": float(r["status"] == "timeout"), "rec_age": (o.step - r["created_step"]) / 20,
                      "rec_foreign": float(r.get("created_step", 0) == 0 and r.get("budget", 1) == 0)})
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Runner (D and arbitrary choosers)
# ---------------------------------------------------------------------------

@dataclass
class Ep:
    env: Any
    tracker: ProgressTracker
    hidden: Any = None
    cap: int = 96
    taken: int = 0
    recorder: Any = None
    actions_taken: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    unsupported: str | None = None


@dataclass
class Dec:
    index: int
    observation: Any
    actions: list
    greedy: int
    default: int
    masked: list
    flagged: bool
    probabilities: list      # float64 softmax over the padded row
    logits: Any              # float cpu tensor row
    telemetry: Any = None


def new_ep(env, cap=96, telemetry=False, info=None) -> Ep:
    o = env.observe()
    rec = None
    if telemetry:
        from .campaign04_telemetry import TelemetryRecorder
        rec = TelemetryRecorder(o)
    return Ep(env, ProgressTracker(o), cap=cap, recorder=rec, info=dict(info or {}))


def _stack_hidden(actor, eps, active):
    import torch
    rows = [eps[i].hidden for i in active]
    if all(r is None for r in rows):
        return None
    init = getattr(actor, "initial", None)
    return torch.cat([r if r is not None else init.unsqueeze(0) for r in rows])


def d_choice(decisions, eps):
    """D: the R-mask masked argmax (``campaign04_meta.run`` default / deploy r_mask)."""
    return [d.default for d in decisions]


def run_policy(actor, eps, chooser: Callable = d_choice, *, device="cpu", neural_work_per_forward=1.0,
               hook=None, timing=None):
    """Advance every episode until done / cap / unsupported. ``chooser(decisions, eps) ->
    action indices``; ``hook(ep, decision, choice)`` runs after the choice and before any
    step-t charge (branch snapshots). Decisions (greedy, R-mask default, fallbacks,
    intervention notes) are exactly ``campaign04_meta.run``'s; the D chooser is D."""
    import torch
    from .campaign02_training import PublicInterfaceCapacityError, policy_batch, prepare_frame, score_batch
    from .campaign04_telemetry import policy_stats
    timing = timing if timing is not None else {}
    for k in ("actor_s", "env_s", "decisions"):
        timing.setdefault(k, 0.0)
    observations = [ep.env.observe() for ep in eps]
    while True:
        active, public = [], []
        for i, ep in enumerate(eps):
            if observations[i].done or ep.taken >= ep.cap or ep.unsupported:
                continue
            try:
                public.append(prepare_frame(observations[i], actor))
            except PublicInterfaceCapacityError as exc:
                ep.unsupported = str(exc)
                continue
            active.append(i)
        if not active:
            break
        t0 = time.perf_counter()
        batch = policy_batch(actor, [frame for _, frame in public], device)
        with torch.no_grad():
            logits, _, next_hidden = score_batch(actor, batch, _stack_hidden(actor, eps, active))
            greedy = logits.argmax(-1).cpu().tolist()
            distribution = logits.double().softmax(-1).cpu().tolist()
            logits_cpu = logits.float().cpu()
        timing["actor_s"] += time.perf_counter() - t0
        decisions = []
        for row, i in enumerate(active):
            ep, actions = eps[i], public[row][0]
            masked = ep.tracker.mask(actions)
            flagged = ep.tracker.is_flagged_state()
            n_masked = sum(masked)
            default = greedy[row]
            if n_masked and n_masked < len(actions):
                row_logits = logits_cpu[row, :len(actions)].clone()
                row_logits[[k for k, m in enumerate(masked) if m]] = float("-inf")
                default = int(row_logits.argmax().item())
            d = Dec(i, observations[i], actions, greedy[row], default, masked, flagged, distribution[row],
                    logits_cpu[row])
            if ep.recorder is not None:
                d.telemetry = ep.recorder.features(observations[i], ep.tracker, policy_stats(
                    distribution[row][:len(actions)], default, greedy[row], n_masked, flagged))
            decisions.append(d)
        if next_hidden is not None:   # post-forward state first: snapshots taken by the chooser/hook carry it
            for row, i in enumerate(active):
                eps[i].hidden = next_hidden[row:row + 1]
        choices = chooser(decisions, eps)
        for d, choice in zip(decisions, choices):
            ep = eps[d.index]
            if hook is not None:
                hook(ep, d, choice)
            n_masked = sum(d.masked)
            if choice == d.default:
                if n_masked and n_masked < len(d.actions):
                    ep.tracker.note_intervention(ep.taken, "mask", n_masked, d.default != d.greedy)
                elif n_masked:
                    ep.tracker.note_intervention(ep.taken, "fallback", n_masked, False)
            else:
                ep.tracker.note_intervention(ep.taken, "hr_option", n_masked, True)
            t0 = time.perf_counter()
            ep.env.charge_compute(neural_work_per_forward)
            action = d.actions[choice]
            after = ep.env.step(action)
            step_class = ep.tracker.update(after, action)
            if ep.recorder is not None:
                ep.recorder.update(after, action, step_class, "default" if choice == d.default else "mask_top",
                                   neural_work_per_forward)
            timing["env_s"] += time.perf_counter() - t0
            ep.actions_taken.append(choice)
            observations[d.index] = after
            ep.taken += 1
            timing["decisions"] += 1
    return eps


def outcome(env) -> dict:
    out = env.evaluate()
    return {"success": bool(out["verified_success"]), "utility": out["utility"], "cost": out["cost"],
            "steps": out["steps"], "compute_units": out["compute_units"]}


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """A pre-decision snapshot (before the step-t charges; hidden = post-forward state)."""
    key: Any
    step: int
    env: Any
    tracker: Any
    hidden: Any
    actions: list
    d_index: int
    utility: float          # U_t
    cap: int                # decisions left incl. the branched one
    options: list
    info: dict = field(default_factory=dict)


def snapshot(ep: Ep, d: Dec, key, options, info=None) -> Point:
    env, hidden, tracker = clone_branch(ep.env, ep.hidden, ep.tracker)
    return Point(key, ep.taken, env, tracker, hidden, d.actions, d.default, env.current_utility(), ep.cap - ep.taken,
                 options, dict(info or {}))


def make_teacher():
    """A fresh public dep_reuse teacher (``campaign02_references.make_reference``)."""
    from .campaign02_references import make_reference
    return make_reference(TEACHER)


def committed(o) -> bool:
    """The step that produced observation ``o`` was a successful selection/assignment commit
    (the environment's completion rule; direct or via use_return). Public feedback only."""
    fb = o.feedback or {}
    return fb.get("status") == "success" and ("selection_id" in fb or "assignment_id" in fb)


def teacher_index(teacher, o, actions):
    """Catalog index of the teacher's action (public observation + public catalog), or None
    when the proposal is outside the catalog (never repaired)."""
    action = teacher.choose(o, actions)
    try:
        return actions.index(action)
    except ValueError:
        return None


def delegate_choice(decisions, eps):
    """D, except for episodes with an active delegation (``ep.info["delegate"]``): the teacher
    chooses until the previous teacher step committed, or ``max_steps`` teacher steps were taken
    (or its proposal is outside the catalog); from then on D."""
    out = []
    for d in decisions:
        g = eps[d.index].info.get("delegate")
        choice = d.default
        if g is not None and g["active"]:
            if g["steps"] and committed(d.observation):
                g["active"], g["end"] = False, "commit"
            elif g["steps"] >= g["max_steps"]:
                g["active"], g["end"] = False, "max_steps"
            else:
                i = teacher_index(g["teacher"], d.observation, d.actions)
                if i is None:
                    g["active"], g["end"] = False, "out_of_catalog"
                    g["out_of_catalog"] += 1
                else:
                    choice = i
                    g["steps"] += 1
                    g["agree"] += int(i == d.default)
        out.append(choice)
    return out


def _delegate_end(g, ep):
    if g["end"] is not None:
        return g["end"]
    o = ep.env.observe()       # the delegation was active to the end: the last step was the teacher's
    if committed(o):
        return "commit"
    return "episode_end" if o.done else "cap"


def run_branches(actor, jobs, *, neural_work_per_forward=1.0, device="cpu", batch=128, timing=None,
                 teacher_factory=make_teacher, delegate_max_steps=DELEGATE_MAX_STEPS):
    """jobs: [(Point, action index | "delegate")] -> [{utility, success, cost, steps, dU, dsteps, truncated}]
    (take the action at the point, then D to termination; "delegate": the teacher from the point until
    its next successful commit / max steps, then D), in job order."""
    out = []
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        eps = []
        for p, a in chunk:
            env, hidden, tracker = clone_branch(p.env, p.hidden, p.tracker)
            info = {}
            if a == "delegate":
                teacher = teacher_factory()     # fresh instance, constructed at the point
                a = teacher_index(teacher, env.observe(), p.actions)
                if a is None:
                    raise ValueError("delegate option without an in-catalog first teacher action")
                info["delegate"] = {"teacher": teacher, "active": True, "steps": 1, "agree": int(a == p.d_index),
                                    "max_steps": delegate_max_steps, "end": None, "out_of_catalog": 0,
                                    "first": a}
            env.charge_compute(neural_work_per_forward)   # the step-t actor forward, charged once as on the main line
            action = p.actions[a]
            after = env.step(action)
            tracker.update(after, action)
            ep = Ep(env, tracker, hidden=hidden, cap=p.cap - 1, info=info)
            eps.append(ep)
        delegated = any("delegate" in ep.info for ep in eps)
        run_policy(actor, eps, delegate_choice if delegated else d_choice, device=device,
                   neural_work_per_forward=neural_work_per_forward, timing=timing)
        for (p, a), ep in zip(chunk, eps):
            o = outcome(ep.env)
            r = {**o, "dU": o["utility"] - p.utility, "dsteps": ep.taken + 1, "truncated": not ep.env.observe().done}
            g = ep.info.get("delegate")
            if g is not None:
                r.update({"teacher_steps": g["steps"], "teacher_agree": g["agree"], "delegate_end": _delegate_end(g, ep),
                          "teacher_first": g["first"], "teacher_out_of_catalog": g["out_of_catalog"]})
            out.append(r)
    return out


def budget_override_options(o, actions, d_index) -> list[int]:
    """Option B's actions at a point (public): D-call point -> call(p_D, b), b in B_BUDGETS minus D's budget;
    ``other`` point -> call(p, b) for every open problem p, b in B_BUDGETS ("remaining" = the remaining-work
    budget, when allowed). Empty elsewhere. Always a subset of the anchored O(I) (tested)."""
    anchor, _ = anchor_of(o, actions[d_index])
    budgets = {b if b != "remaining" else o.remaining_work for b in B_BUDGETS}
    open_ = stage_open(o)
    out = []
    for i, a in enumerate(actions):
        if a.kind != "call" or i == d_index or a.arguments.get("budget") not in budgets:
            continue
        if anchor == "call" and a.arguments.get("problem") == actions[d_index].arguments.get("problem"):
            out.append(i)
        elif anchor == "other":
            p = o.problems.get(a.arguments.get("problem"))
            if p is not None and open_.get(p["primitive"], False):
                out.append(i)
    return out


# ---------------------------------------------------------------------------
# State sampling (registered, public, online)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HRConfig:
    """Registered A-HR sampling constants."""
    p_call: float = 0.35
    p_reuse_recompute: float = 0.35
    p_commit_revise: float = 0.5
    p_other: float = 0.08
    max_points: int = 8            # sampled points per episode (online: the first 8 sampled)
    p_all_states: float = 1 / 16   # worlds (seed hash, same for every base) branching EVERY decision state
    all_states: bool = True        # False: sampled points only (e.g. the full-catalog subsample)
    check_fraction: float = 0.1    # points that also branch D's own choice (checked equal to the main line)
    max_steps: int = 96
    option_mode: str = "anchored"

    def p_of(self, anchor: str) -> float:
        return {"call": self.p_call, "reuse_recompute": self.p_reuse_recompute,
                "commit_revise": self.p_commit_revise, "other": self.p_other}[anchor]

    def describe(self) -> str:
        return (f"online public Bernoulli per D decision, one uniform per decision from the world stream "
                f"('e05-hr-sample', seed): sampled if u < p(anchor of D's action) with p = call {self.p_call}, "
                f"reuse_recompute {self.p_reuse_recompute}, commit_revise {self.p_commit_revise}, other {self.p_other}, "
                f"and fewer than {self.max_points} earlier sampled points; plus, on a {self.p_all_states:.4f} fraction "
                f"of worlds (hash 'e05-hr-all'; identical across bases; all_states={self.all_states}), every decision "
                f"state is branched (p_include = 1 there)")


def all_states_episode(seed: int, cfg: HRConfig) -> bool:
    return cfg.all_states and digest_seed("e05-hr-all", seed) / 2 ** 64 < cfg.p_all_states


def check_point(seed: int, step: int, cfg: HRConfig) -> bool:
    return digest_seed("e05-hr-check", seed, step) / 2 ** 64 < cfg.check_fraction


class Sampler:
    """The registered online public sampler for one episode."""

    def __init__(self, seed: int, cfg: HRConfig):
        self.rng = random.Random(digest_seed("e05-hr-sample", seed))
        self.cfg, self.count = cfg, 0
        self.all_states = all_states_episode(seed, cfg)

    def __call__(self, anchor: str) -> tuple[bool, float, bool]:
        """(sampled, inclusion probability, branch this state)."""
        u = self.rng.random()          # always drawn: the stream stays aligned with decisions
        p = self.cfg.p_of(anchor)
        sampled = self.count < self.cfg.max_points and u < p
        self.count += sampled
        return sampled, p, sampled or self.all_states


# ---------------------------------------------------------------------------
# States (D rollouts) and branch labels
# ---------------------------------------------------------------------------

def d_rollouts(actor, envs, seeds, cfg: HRConfig = HRConfig(), *, neural_work_per_forward=1.0, device="cpu",
               telemetry=False, hook=None, timing=None):
    eps = [new_ep(env, cfg.max_steps, telemetry=telemetry, info={"seed": s}) for env, s in zip(envs, seeds)]
    run_policy(actor, eps, d_choice, device=device, neural_work_per_forward=neural_work_per_forward, hook=hook,
               timing=timing)
    return eps


def multi_option_set(o, actions, d_index, teacher_factory=make_teacher) -> list[dict[str, Any]]:
    """A-HR2 options at a point: D and ``delegate`` (index = the fresh teacher's step-t action; omitted
    when that proposal is outside the catalog). Public only."""
    anchor, _ = anchor_of(o, actions[d_index])
    out = [{"index": d_index, "type": "D", "types": ["D", action_type(actions[d_index])], "is_d": True,
            "anchor": anchor}]
    i = teacher_index(teacher_factory(), o, actions)
    if i is not None:
        out.append({"index": i, "type": "delegate", "types": ["delegate", action_type(actions[i])], "is_d": False,
                    "anchor": anchor, "teacher_first_is_d": i == d_index})
    return out


DELEGATE_FIRST_KINDS = ("call", "reuse", "retrieve", "recompute", "commit", "revise", "abstain", "verify", "other")


def delegate_features(o, actions, opt) -> dict[str, float]:
    """Extra public features of the delegate option (teacher's first action, D's anchor)."""
    a = actions[opt["index"]]
    t = action_type(a)
    kind = ("commit" if a.kind in ("commit_pending", "commit_assignment") else "revise" if a.kind == "uncommit"
            else "verify" if a.kind == "verify" else t if t in DELEGATE_FIRST_KINDS else "other")
    f = {"teacher_first_is_d": float(opt["teacher_first_is_d"])}
    f.update({f"teacher_first_{k}": float(kind == k) for k in DELEGATE_FIRST_KINDS})
    f.update({f"anchor_{k}": float(opt["anchor"] == k) for k in ANCHORS})
    return f


DELEGATE_FIELDS = ("teacher_steps", "teacher_agree", "delegate_end", "teacher_first", "teacher_out_of_catalog")


def _job_key(opt):
    return "delegate" if opt["type"] == "delegate" else opt["index"]


def hr_labels(actor, envs, seeds, cfg: HRConfig = HRConfig(), *, neural_work_per_forward=1.0, device="cpu",
              branch_batch=128, features=True, branch=True, teacher_factory=make_teacher):
    """A-HR states and (``branch=True``) branch labels for one batch of worlds.
    ``cfg.option_mode``: "anchored" (O(I)), "full" (catalog) or "multi" (A-HR2: D + delegate).
    Returns (episodes, points, stats)."""
    t_all = time.process_time()
    timing_main, timing_b = {}, {}
    samplers = {s: Sampler(s, cfg) for s in seeds}
    points: list[Point] = []
    descriptors = {s: [] for s in seeds}

    def hook(ep, d, choice):
        s = ep.info["seed"]
        anchor, _ = anchor_of(d.observation, d.actions[d.default])
        sampled, p, take = samplers[s](anchor)
        descriptors[s].append({"step": ep.taken, "anchor": anchor, "sampled": sampled, "p": p,
                               "d_type": action_type(d.actions[d.default])})
        if not take or not branch:
            return
        n = len(d.actions)
        scores = d.logits[:n].tolist()
        if cfg.option_mode == "multi":
            options = multi_option_set(d.observation, d.actions, d.default, teacher_factory)
        else:
            options = option_set(d.observation, d.actions, d.default, cfg.option_mode, scores, d.masked)
        if cfg.option_mode == "full":
            in_oi = {o["index"] for o in option_set(d.observation, d.actions, d.default, "anchored", scores, d.masked)}
            for o in options:
                o["in_oi"] = o["index"] in in_oi
        pinfo = {"seed": s, "sampled": sampled, "p_sample": p, "p_include": 1.0 if samplers[s].all_states else p,
                 "anchor": anchor, "candidates": n, "d_is_greedy": d.default == d.greedy, "flagged": d.flagged}
        if cfg.option_mode == "multi":
            b = budget_override_options(d.observation, d.actions, d.default)
            oi = {o["index"] for o in option_set(d.observation, d.actions, d.default, "anchored", scores, d.masked)}
            pinfo["b_options"] = b
            pinfo["b_subset_of_oi"] = set(b) <= oi
        if features:
            pinfo["telemetry"] = list(d.telemetry.values)
            pinfo["option_features"] = option_features(d.observation, d.actions, options, scores,
                                                       d.probabilities[:n], d.default)
            for o, f in zip(options, pinfo["option_features"]):
                if o["type"] == "delegate":
                    f.update(delegate_features(d.observation, d.actions, o))
        points.append(snapshot(ep, d, (s, ep.taken), options, pinfo))

    c0 = time.process_time()
    eps = d_rollouts(actor, envs, seeds, cfg, neural_work_per_forward=neural_work_per_forward, device=device,
                     telemetry=features and branch, hook=hook, timing=timing_main)
    main_cpu = time.process_time() - c0
    episodes = []
    for ep in eps:
        s = ep.info["seed"]
        episodes.append({"seed": s, "T": ep.taken, **outcome(ep.env), "truncated": not ep.env.observe().done,
                         "unsupported": ep.unsupported, "actions": list(ep.actions_taken),
                         "all_states": samplers[s].all_states, "decisions": descriptors[s],
                         "sampled_steps": [x["step"] for x in descriptors[s] if x["sampled"]]})
    info = {e["seed"]: e for e in episodes}
    jobs, check = [], []
    for p in points:
        do_check = check_point(p.info["seed"], p.step, cfg)
        check.append(do_check)
        for opt in p.options:
            if opt["is_d"] and not do_check:
                continue
            jobs.append((p, _job_key(opt)))
    c0 = time.process_time()
    results = run_branches(actor, jobs, neural_work_per_forward=neural_work_per_forward, device=device,
                           batch=branch_batch, timing=timing_b, teacher_factory=teacher_factory)
    branch_cpu = time.process_time() - c0
    by = {(id(p), a): r for (p, a), r in zip(jobs, results)}
    out_points, checks = [], [0, 0]
    for p, do_check in zip(points, check):
        e = info[p.info["seed"]]
        main = {"success": e["success"], "utility": e["utility"], "cost": e["cost"], "steps": e["steps"],
                "dU": e["utility"] - p.utility, "dsteps": e["T"] - p.step, "truncated": e["truncated"]}
        opts = []
        for opt, f in zip(p.options, p.info.get("option_features") or [None] * len(p.options)):
            r = by.get((id(p), _job_key(opt)))
            if opt["type"] == "delegate" and r["teacher_first"] != opt["index"]:
                raise AssertionError("delegate: the branch's fresh teacher disagrees with the point's proposal")
            if opt["is_d"]:
                if r is not None:
                    checks[0] += 1
                    checks[1] += int((r["dU"], r["success"], r["cost"]) == (main["dU"], main["success"], main["cost"]))
                r = main
            action = p.actions[opt["index"]]
            opts.append({**opt, "action": {"kind": action.kind, "arguments": dict(action.arguments)},
                         "dU": r["dU"], "success": r["success"], "cost_rem": r["cost"] - (-p.utility),
                         "dsteps": r["dsteps"], "truncated": r["truncated"], **({"features": f} if f else {}),
                         **{k: r[k] for k in DELEGATE_FIELDS if k in r}})
        q_d = main["dU"]
        best = max(opts, key=lambda x: (x["dU"], x["is_d"], -x["index"]))
        non_abstain = [x for x in opts if x["type"] != "abstain"]
        best_na = max(non_abstain, key=lambda x: (x["dU"], x["is_d"], -x["index"]))
        d_opt = next(o for o in opts if o["is_d"])
        out_points.append({"seed": p.info["seed"], "step": p.step, "T": e["T"], "U_t": p.utility,
                           "d_success": e["success"], "sampled": p.info["sampled"], "p_sample": p.info["p_sample"],
                           "p_include": p.info["p_include"], "anchor": p.info["anchor"],
                           "candidates": p.info["candidates"], "d_index": p.d_index, "d_type": d_opt["types"][1],
                           "d_is_greedy": p.info["d_is_greedy"], "flagged": p.info["flagged"],
                           "q_d": q_d, "headroom": best["dU"] - q_d, "headroom_excl_abstain": best_na["dU"] - q_d,
                           "best_index": best["index"], "best_type": best["type"], "options": opts,
                           **({"telemetry": p.info["telemetry"]} if "telemetry" in p.info else {}),
                           **{k: p.info[k] for k in ("b_options", "b_subset_of_oi") if k in p.info}})
    branch_steps = [r["dsteps"] for r in results]
    stats = {"episodes": len(episodes), "points": len(points), "sampled_points": sum(p.info["sampled"] for p in points),
             "branches": len(jobs), "default_checks": checks[0], "default_check_matches": checks[1],
             "decisions": sum(len(e["decisions"]) for e in episodes),
             "cpu": {"main_s": main_cpu, "branch_s": branch_cpu, "total_s": time.process_time() - t_all},
             "branch_steps": branch_steps, "timing_branch": timing_b}
    return episodes, out_points, stats


def summarize_states(points: list[dict], thr: float = 0.01) -> dict:
    """Quick per-file headroom summary (the analysis tool does the full statistics)."""
    if not points:
        return {}
    h = [p["headroom"] for p in points]
    hx = [p["headroom_excl_abstain"] for p in points]
    return {"points": len(points), "mean_headroom": sum(h) / len(h), "mean_headroom_excl_abstain": sum(hx) / len(hx),
            "frac_ge_thr": sum(x >= thr for x in h) / len(h), "frac_ge_thr_excl_abstain": sum(x >= thr for x in hx) / len(hx),
            "mean_options": sum(len(p["options"]) for p in points) / len(points)}
