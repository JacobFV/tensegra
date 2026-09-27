"""A-PI-T learned delegation controller (extended-05 Track A; registry A-PI-T).

The learned component decides only WHEN to hand control to the supplied public
``dep_reuse`` teacher (option T of A-HR2); the teacher and the option are supplied
(disclosed), and D (greedy + R-mask, ``campaign05_options.d_choice``) is the base policy.

Controller (``Controller``; artifact written by ``research/tools/campaign05_apit.py train``)
------------------------------------------------------------------------------------------
Per-anchor regression of the delegate advantage  Q^D(I, delegate) - Q^D(I, D(I))  on the
public feature vector of the A-HR2 G1b estimator (``FEATURE_VERSION``):
``telemetry d_t (c-telemetry-v1, 89)`` + the delegate option's public features
(``FEATURE_KEYS`` actor numbers / budget facts / record relations of the teacher's first
action, ``DELEGATE_KEYS`` teacher-first-action kind and D's anchor) + a constant 1.
Families: ridge (as G1b) or a small tanh MLP on standardized inputs, one model per
trigger anchor. The family and the margin m are chosen by the training tool on held-out
world folds of the A-HR2 screening worlds (development only) and frozen in the artifact.
Inputs are public only: the public observation, the public catalog, the frozen actor's
numbers on that catalog, the progress diagnostic's public state, the telemetry recorder
(public bookkeeping) and the teacher's proposal (a function of the public observation).

Deployed policy pi_T (``Delegator("pit")``)
---------------------------------------------
Follow D. At every decision where D has the turn (no active delegation) and D's anchor
(``campaign05_options.anchor_of``) is in ``TRIGGERS`` = {call, reuse_recompute,
commit_revise}, a fresh teacher (``make_teacher()``) proposes its action; if the proposal
is in the public catalog the decision is *eligible*, the controller predicts the
delegate advantage and, if it exceeds m, the teacher takes this and every following
decision until the first decision whose observation reports a successful commit
(``campaign05_options.committed``), or 12 teacher steps, or the episode ends; a teacher
proposal outside the catalog ends the delegation (never repaired) and D takes that
decision. These are exactly option T's semantics (``delegate_choice``). After a
delegation ends, D has the turn again at that same decision (so the controller may
delegate again: multiple delegations per episode). The progress tracker and the
telemetry recorder are updated on every step, teacher steps included.

Charges (every charge lands once in the environment's evaluation utility):
- every decision, D's or the teacher's, is charged one actor/controller forward
  (``neural_work_per_forward``, the extended-04 reference tariff 1.0) by ``run_policy``:
  a teacher step costs exactly one forward, as in A-HR2 and as the dep_reuse reference
  evaluation (``run_episode(..., model_compute_tariff=1.0)``);
- every controller evaluation (eligible decision) is additionally charged
  ``controller_units`` = neural_work_per_forward x (controller parameters / actor
  parameters): the controller's forward priced in proportion to its size relative to the
  frozen actor. The teacher proposal used as a feature is part of the controller forward;
  the number of consultations is recorded so a reader can re-price them
  (``consults`` x compute_price).
Controller charges are excluded from the telemetry (metacontrol charges never change
behaviour, as in extended-04 Track C).

Comparators (same machinery): ``d`` (never delegate; = D), ``r1`` (delegate at every
eligible call anchor), ``r2`` (every eligible commit_revise anchor), ``random`` (every
eligible trigger decision with probability q, one uniform per eligible decision from a
per-(world, base) stream), ``always`` (the teacher chooses every decision; D only where
the proposal is outside the catalog). The evaluation's always-teacher comparator is
the dep_reuse reference evaluation itself (``run_episode``), tested equal to ``always``.

Torch is imported lazily (actor forwards only); numpy for the controller.
"""
from __future__ import annotations

from copy import copy as _copy
import hashlib
import json
import math
import random
from typing import Any

from .campaign05_options import (ANCHORS, Ep, action_type, anchor_of, committed, delegate_features, digest_seed,
                                 make_teacher, option_features, outcome, run_policy, teacher_index)
from .campaign04_branch import clone_branch

APIT_VERSION = "e05-apit-v1"
FEATURE_VERSION = "e05-apit-features-v1 (= A-HR2 G1b delegate features: c-telemetry-v1 + FEATURE_KEYS + DELEGATE_KEYS + 1)"
TRIGGERS = ("call", "reuse_recompute", "commit_revise")
DELEGATE_MAX_STEPS = 12
POLICIES = ("d", "pit", "r1", "r2", "random", "always")
RULE_TRIGGERS = {"d": (), "pit": TRIGGERS, "random": TRIGGERS, "r1": ("call",), "r2": ("commit_revise",)}
# identical to research/tools/campaign05_hr.py FEATURE_KEYS / DELEGATE_KEYS (tested)
FEATURE_KEYS = ("prob", "logit_gap", "rank_frac", "budget_log", "budget_frac", "budget_vs_d_log", "problem_latest",
                "prev_calls", "prev_timeouts", "prev_max_budget_log", "rel_type_match", "rel_request_match",
                "rel_canonical_match", "rel_dependency_match", "rel_requirements_match", "rel_selection_match",
                "rel_usable", "rec_timeout", "rec_age", "rec_foreign")
DELEGATE_KEYS = ("teacher_first_is_d",) + tuple(f"teacher_first_{k}" for k in (
    "call", "reuse", "retrieve", "recompute", "commit", "revise", "abstain", "verify", "other")) + tuple(
    f"anchor_{k}" for k in ANCHORS)


def feature_names() -> list[str]:
    from .campaign04_telemetry import LAYOUT
    return [f"telemetry:{g}:{n}" for g, n in LAYOUT] + list(FEATURE_KEYS) + list(DELEGATE_KEYS) + ["bias"]


def feature_vector(telemetry, features: dict) -> list[float]:
    """[telemetry d_t, option features (FEATURE_KEYS + DELEGATE_KEYS; missing = 0), 1] (A-HR2 ``_x``)."""
    return [float(v) for v in telemetry] + [float(features.get(k, 0.0)) for k in FEATURE_KEYS + DELEGATE_KEYS] + [1.0]


def delegate_option(o, actions, d_index, teacher_idx) -> dict[str, Any]:
    """The delegate option dict exactly as ``campaign05_options.multi_option_set`` builds it."""
    anchor, _ = anchor_of(o, actions[d_index])
    return {"index": teacher_idx, "type": "delegate", "types": ["delegate", action_type(actions[teacher_idx])],
            "is_d": False, "anchor": anchor, "teacher_first_is_d": teacher_idx == d_index}


def decision_features(d, teacher_idx) -> list[float]:
    """Public feature vector of the delegate option at decision ``d`` (a ``campaign05_options.Dec`` with
    telemetry): the same computation as ``hr_labels`` (option_features + delegate_features)."""
    if d.telemetry is None:
        raise ValueError("the controller needs telemetry (new_ep(..., telemetry=True))")
    n = len(d.actions)
    opt = delegate_option(d.observation, d.actions, d.default, teacher_idx)
    f = option_features(d.observation, d.actions, [opt], d.logits[:n].tolist(), d.probabilities[:n], d.default)[0]
    f.update(delegate_features(d.observation, d.actions, opt))
    return feature_vector(d.telemetry.values, f)


# ---------------------------------------------------------------------------
# Models (numpy)
# ---------------------------------------------------------------------------

def fit_ridge(X, y, lam):
    """Ridge with an unpenalized last (bias) column (the G1b estimator)."""
    import numpy as np
    X, y = np.asarray(X, float), np.asarray(y, float)
    reg = lam * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    return {"kind": "ridge", "lam": lam, "w": np.linalg.solve(X.T @ X + reg, X.T @ y).tolist()}


def fit_mlp(X, y, hidden=16, l2=1e-3, epochs=400, lr=1e-2, seed=0):
    """One tanh hidden layer on standardized inputs (bias column dropped), full-batch Adam on MSE + L2."""
    import numpy as np
    X, y = np.asarray(X, float)[:, :-1], np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    rng = np.random.default_rng(seed)
    params = {"W1": rng.normal(0, 1 / math.sqrt(Z.shape[1]), (Z.shape[1], hidden)), "b1": np.zeros(hidden),
              "W2": rng.normal(0, 1 / math.sqrt(hidden), hidden), "b2": np.array(float(y.mean()))}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    n = len(y)
    for t in range(1, epochs + 1):
        H = np.tanh(Z @ params["W1"] + params["b1"])
        err = H @ params["W2"] + params["b2"] - y
        g_out = 2 * err / n
        grads = {"W2": H.T @ g_out + 2 * l2 * params["W2"], "b2": np.array(g_out.sum())}
        gH = np.outer(g_out, params["W2"]) * (1 - H ** 2)
        grads["W1"] = Z.T @ gH + 2 * l2 * params["W1"]
        grads["b1"] = gH.sum(0)
        for k in params:
            m[k] = .9 * m[k] + .1 * grads[k]
            v2[k] = .999 * v2[k] + .001 * grads[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - .9 ** t)) / (np.sqrt(v2[k] / (1 - .999 ** t)) + 1e-8)
    return {"kind": "mlp", "hidden": hidden, "l2": l2, "epochs": epochs, "lr": lr, "seed": seed,
            "mu": mu.tolist(), "sd": sd.tolist(), **{k: np.asarray(v).tolist() for k, v in params.items()}}


def model_parameters(model) -> int:
    """Numbers used by one forward (weights, biases; for the MLP also the input standardization)."""
    if model["kind"] == "ridge":
        return len(model["w"])
    n_in, hidden = len(model["mu"]), len(model["b1"])
    return 2 * n_in + n_in * hidden + hidden + hidden + 1


def predict_model(model, X):
    """Predictions for rows X (full feature vectors incl. the bias column)."""
    import numpy as np
    X = np.atleast_2d(np.asarray(X, float))
    if model["kind"] == "ridge":
        return X @ np.asarray(model["w"])
    Z = (X[:, :-1] - np.asarray(model["mu"])) / np.asarray(model["sd"])
    H = np.tanh(Z @ np.asarray(model["W1"]) + np.asarray(model["b1"]))
    return H @ np.asarray(model["W2"]) + float(model["b2"])


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Controller:
    """Frozen delegation controller: per-anchor models + margin (from the artifact)."""

    def __init__(self, artifact: dict, margin: float | None = None, verify: bool = True):
        if verify and canonical_hash(artifact["controller"]) != artifact["controller_sha256"]:
            raise ValueError("controller hash mismatch")
        c = artifact["controller"]
        if c["feature_version"] != FEATURE_VERSION or c["feature_names"] != feature_names():
            raise ValueError("controller feature definition differs from this code")
        self.artifact = artifact
        self.models = c["models"]
        self.family = c["family"]
        self.margin = float(c["margin"] if margin is None else margin)
        self.n_parameters = sum(model_parameters(m) for m in self.models.values())

    @classmethod
    def from_file(cls, path, **kw):
        with open(path) as f:
            return cls(json.load(f), **kw)

    def predict(self, anchor: str, x) -> float:
        return float(predict_model(self.models[anchor], x)[0])


# ---------------------------------------------------------------------------
# Delegating policy (a run_policy chooser)
# ---------------------------------------------------------------------------

def _new_state(rng=None):
    return {"g": None, "eligible": 0, "starts": 0, "teacher_steps": 0, "controller_evals": 0, "consults": 0,
            "consult_out_of_catalog": 0, "delegation_out_of_catalog": 0, "eligible_by_anchor": {},
            "starts_by_anchor": {}, "ends": {}, "last": None, "rng": rng, "preds": []}


class Delegator:
    """run_policy chooser for pi_T and the comparators (see module docstring)."""

    def __init__(self, policy: str, *, controller: Controller | None = None, controller_units: float = 0.0,
                 rate: float | None = None, rng_key: str = "", teacher_factory=make_teacher,
                 max_steps: int = DELEGATE_MAX_STEPS, record_predictions: bool = False):
        if policy not in POLICIES:
            raise ValueError(policy)
        if policy == "pit" and controller is None:
            raise ValueError("pit needs a controller")
        if policy == "random" and rate is None:
            raise ValueError("random needs the matched rate")
        if controller_units < 0 or not math.isfinite(controller_units):
            raise ValueError("controller_units must be finite and >= 0")
        self.policy, self.controller, self.controller_units = policy, controller, float(controller_units)
        self.rate, self.rng_key, self.teacher_factory, self.max_steps = rate, rng_key, teacher_factory, max_steps
        self.record_predictions = record_predictions
        self.triggers = RULE_TRIGGERS.get(policy, ())

    def attach(self, ep: Ep, seed: int) -> Ep:
        rng = random.Random(digest_seed("e05-apit-random", self.rng_key, seed)) if self.policy == "random" else None
        ep.info["seed"] = seed
        ep.info["apit"] = _new_state(rng)
        if self.policy == "always":
            ep.info["apit"]["teacher"] = self.teacher_factory()
        return ep

    def _end(self, st, why):
        st["g"]["active"] = False
        st["ends"][why] = st["ends"].get(why, 0) + 1
        st["g"] = None

    def __call__(self, decisions, eps):
        return [self.choose(d, eps[d.index]) for d in decisions]

    def choose(self, d, ep) -> int:
        st = ep.info["apit"]
        st["last"] = None
        if self.policy == "always":
            i = teacher_index(st["teacher"], d.observation, d.actions)
            if i is None:
                st["delegation_out_of_catalog"] += 1
                return d.default
            st["teacher_steps"] += 1
            return i
        g = st["g"]
        if g is not None:
            if g["steps"] and committed(d.observation):
                self._end(st, "commit")
            elif g["steps"] >= g["max_steps"]:
                self._end(st, "max_steps")
            else:
                i = teacher_index(g["teacher"], d.observation, d.actions)
                if i is None:
                    st["delegation_out_of_catalog"] += 1
                    self._end(st, "out_of_catalog")
                else:
                    g["steps"] += 1
                    st["teacher_steps"] += 1
                    return i
        # D has the turn
        anchor, _ = anchor_of(d.observation, d.actions[d.default])
        if anchor not in self.triggers:
            return d.default
        teacher = self.teacher_factory()          # fresh instance at the point
        ti = teacher_index(teacher, d.observation, d.actions)
        st["consults"] += 1
        if ti is None:
            st["consult_out_of_catalog"] += 1
            return d.default
        st["eligible"] += 1
        st["eligible_by_anchor"][anchor] = st["eligible_by_anchor"].get(anchor, 0) + 1
        pred = None
        if self.policy == "pit":
            pred = self.controller.predict(anchor, decision_features(d, ti))
            st["controller_evals"] += 1
            if self.controller_units:
                ep.env.charge_compute(self.controller_units)
            fire = pred > self.controller.margin
            if self.record_predictions:
                st["preds"].append([ep.taken, anchor, pred, bool(fire)])
        elif self.policy == "random":
            fire = st["rng"].random() < self.rate
        else:
            fire = True
        st["last"] = {"step": ep.taken, "anchor": anchor, "teacher_index": ti, "fired": bool(fire), "pred": pred}
        if not fire:
            return d.default
        st["g"] = {"teacher": teacher, "active": True, "steps": 1, "max_steps": self.max_steps}
        st["starts"] += 1
        st["teacher_steps"] += 1
        st["starts_by_anchor"][anchor] = st["starts_by_anchor"].get(anchor, 0) + 1
        return ti


def episode_row(ep: Ep, policy: str) -> dict[str, Any]:
    st = ep.info.get("apit") or _new_state()
    o = outcome(ep.env)
    ends = dict(st["ends"])
    if st["g"] is not None:          # delegation active to the end
        last = ep.env.observe()
        why = "commit" if committed(last) else ("episode_end" if last.done else "cap")
        ends[why] = ends.get(why, 0) + 1
    return {"seed": ep.info.get("seed"), "policy": policy, **o, "decisions": ep.taken,
            "truncated": not ep.env.observe().done, "unsupported": ep.unsupported,
            "delegations": st["starts"], "eligible": st["eligible"], "teacher_steps": st["teacher_steps"],
            "controller_evals": st["controller_evals"], "consults": st["consults"],
            "consult_out_of_catalog": st["consult_out_of_catalog"],
            "delegation_out_of_catalog": st["delegation_out_of_catalog"],
            "eligible_by_anchor": st["eligible_by_anchor"], "delegations_by_anchor": st["starts_by_anchor"],
            "delegation_ends": ends, **({"predictions": st["preds"]} if st["preds"] else {})}


def run_delegator(actor, envs, seeds, delegator: Delegator, *, cap=96, neural_work_per_forward=1.0, device="cpu",
                  hook=None, timing=None):
    from .campaign05_options import new_ep
    telemetry = delegator.policy == "pit"
    eps = [delegator.attach(new_ep(env, cap, telemetry=telemetry), s) for env, s in zip(envs, seeds)]
    run_policy(actor, eps, delegator, device=device, neural_work_per_forward=neural_work_per_forward, hook=hook,
               timing=timing)
    return eps


# ---------------------------------------------------------------------------
# Delegation-decision regret (branch-evaluated subsample)
# ---------------------------------------------------------------------------

def regret_world(seed: int, p: float) -> bool:
    return digest_seed("e05-apit-regret", seed) / 2 ** 64 < p


class RegretHook:
    """run_policy hook for pi_T: snapshot every eligible decision (after the choice and the controller
    charge, before the step-t forward charge) on subsampled worlds."""

    def __init__(self, p_world: float, max_points: int = 16):
        self.p_world, self.max_points, self.points = p_world, max_points, []
        self._n = {}

    def __call__(self, ep, d, choice):
        st = ep.info["apit"]
        last = st.get("last")
        seed = ep.info["seed"]
        if last is None or last["step"] != ep.taken or not regret_world(seed, self.p_world):
            return
        if self._n.get(seed, 0) >= self.max_points:
            return
        self._n[seed] = self._n.get(seed, 0) + 1
        env, hidden, tracker = clone_branch(ep.env, ep.hidden, ep.tracker)
        self.points.append({"seed": seed, "step": ep.taken, "anchor": last["anchor"], "fired": last["fired"],
                            "pred": last["pred"], "teacher_index": last["teacher_index"], "d_index": d.default,
                            "choice": choice, "actions": d.actions, "env": env, "hidden": hidden, "tracker": tracker,
                            "recorder": ep.recorder.copy(), "U_t": env.current_utility(), "cap": ep.cap - ep.taken,
                            "ep": ep})


def run_regret_branches(actor, delegator: Delegator, points, *, neural_work_per_forward=1.0, device="cpu",
                        batch=64, check=False):
    """For each snapshot, branch the alternative (fired -> D now; not fired -> delegate now) and continue
    with pi_T; the main line is the chosen action's continuation. ``check`` also branches the chosen
    action (determinism check). Returns rows with Q(delegate), Q(D) (utility-to-go from U_t)."""
    jobs = []
    for p in points:
        jobs.append((p, "delegate" if not p["fired"] else "D"))
        if check:
            jobs.append((p, "delegate" if p["fired"] else "D"))
    results = []
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        eps = []
        for p, alt in chunk:
            env, hidden, tracker = clone_branch(p["env"], p["hidden"], p["tracker"])
            recorder = p["recorder"].copy()
            ep = Ep(env, tracker, hidden=hidden, cap=p["cap"] - 1, recorder=recorder, info={"seed": p["seed"]})
            st = _new_state()
            a = p["d_index"]
            if alt == "delegate":
                a = p["teacher_index"]
                st["g"] = {"teacher": delegator.teacher_factory(), "active": True, "steps": 1,
                           "max_steps": delegator.max_steps}
            ep.info["apit"] = st
            env.charge_compute(neural_work_per_forward)
            action = p["actions"][a]
            after = env.step(action)
            step_class = tracker.update(after, action)
            recorder.update(after, action, step_class, "default" if a == p["d_index"] else "mask_top",
                            neural_work_per_forward)
            eps.append(ep)
        run_policy(actor, eps, delegator, device=device, neural_work_per_forward=neural_work_per_forward)
        results += [ep.env.evaluate()["utility"] for ep in eps]
    rows, k = [], 0
    for p in points:
        main = p["ep"].env.evaluate()["utility"] - p["U_t"]
        alt = results[k] - p["U_t"]
        k += 1
        chk = None
        if check:
            chk = results[k] - p["U_t"]
            k += 1
        q_del, q_d = (main, alt) if p["fired"] else (alt, main)
        rows.append({"seed": p["seed"], "step": p["step"], "anchor": p["anchor"], "fired": p["fired"],
                     "pred": p["pred"], "q_delegate": q_del, "q_d": q_d, "adv_delegate": q_del - q_d,
                     "regret": max(q_del, q_d) - (q_del if p["fired"] else q_d),
                     **({"check_equal": chk == main} if check else {})})
    return rows
