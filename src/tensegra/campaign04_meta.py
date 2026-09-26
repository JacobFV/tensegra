"""Track C: learned metacognitive appraisal and control over a FROZEN depworld actor.

extended-04 design Track C with v2 revisions 5-8 (review F7-F12, F16). Everything
here acts on action *selection* only: the frozen actor, the recorded facts, the
costs and the evaluator are never changed.

Deployment procedures
---------------------
The **declared continuation / default deployment D0 = greedy + R-mask**
(``campaign04_deploy`` mode ``r_mask``; identical choices, tested). At each
decision the controller picks an intervention u:

- ``default``   the D0 choice (R-mask masked argmax; no intervention);
- ``sample``    one action sampled from the full policy π (float64 softmax, inverse
                CDF, one uniform from the episode's intervention stream);
- ``mask_top``  the best action (by logit) other than the default's choice;
- ``stop``      abstain (ends the episode).

Every Q̂ label is "u at this decision, then D0 to the end of the episode"; the
learned controller re-applies argmax Q̂ at every step, so it is a **one-step
rollout improvement over greedy + R-mask** (review F9).

Counterfactual labels (review F7, F8, F12)
------------------------------------------
Along base rollouts, at a subset of decision points, the episode is cloned
(``campaign04_branch.clone_branch``: the environment sharing the executor, the
actor's recurrent state row if any, the progress tracker) and every intervention
is branched, then D0 runs to the end. Label = realized external utility from that
point: ΔU = U_final(branch) − U_t, where U_t = utility before the step-t charges
(the same ``DepWorkshop`` utility as evaluation; every charge lands in the
environment exactly once). ``sample`` uses K draws (mean and variance recorded).
Branch returns are single-world outcomes: they are **regression targets only** for
E[ΔU | visible history, u, D0] (never argmax class labels). The environment copy
is an oracle-simulation privilege of label generation; it is charged in the
offline ledger (CPU recorded) and never available at evaluation.

Appraisal model (``MetaAppraisal``)
-----------------------------------
A GRU cell (≤ 64 hidden) over the telemetry d_t (``campaign04_telemetry``), updated
at every decision (the automatic path). Heads: P(verified success | D0), expected
remaining cost under D0, Q̂(I_t, default) and advantages Q̂(I_t, u) − Q̂(I_t,
default) for u ∈ {sample, mask_top, stop}. Inputs are telemetry only (the actor's
distribution enters as numbers computed under no_grad: stop-gradient by
construction). Losses are proper scoring rules (BCE, squared error); no loss
rewards higher predicted values.

Controller: u_t = argmax_u [Q̂(u) − c_meta(u)] if the best predicted gain over
default exceeds the margin m, else default. c_meta(u) = 0 for every u because the
label already contains every downstream charge of u, and the metacontroller's own
forward is u-independent (review F16); the forward is charged in the episode as
neural work units ∝ its parameter count relative to the actor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any

from .campaign04_branch import clone_branch
from .campaign04_progress import NO_PROGRESS, ProgressTracker
from .campaign04_telemetry import DIM, INTERVENTIONS, TELEMETRY_VERSION, TelemetryRecorder, group_mask, policy_stats

META_VERSION = "c-meta-v1"
LABEL_VERSION = "c-labels-v1"
CONTINUATION = "D0 = greedy + R-mask (campaign04_deploy r_mask semantics, progress-diagnostic-v1)"
FIXED_MODES = ("greedy", "sampled", "r_mask", "r_sample")
ADV_US = ("sample", "mask_top", "stop")   # advantage head order
BASE_KINDS = ("d0", "eps", "sampled")


def digest_seed(*parts) -> int:
    import hashlib
    import json
    return int(hashlib.sha256(json.dumps(list(parts), separators=(",", ":")).encode()).hexdigest()[:16], 16)


@dataclass
class Episode:
    """Per-episode running state for ``run``."""
    env: Any
    tracker: ProgressTracker
    recorder: TelemetryRecorder | None = None
    hidden: Any = None            # actor recurrent state row [1, ...] or None (lightweight family)
    rng: random.Random | None = None      # intervention / sampling stream (one uniform per sample)
    aux_rng: random.Random | None = None  # agent's own stream (eps base, random-rate arm)
    cap: int = 96                 # remaining decisions allowed (the actor's max_steps budget)
    taken: int = 0
    meta_h: Any = None            # appraisal GRU state [1, H]
    meta_forwards: int = 0
    log: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    unsupported: str | None = None


@dataclass
class Decision:
    """What the runner knows at one decision, before the action is applied."""
    index: int                    # episode index in the run
    row: int                      # batch row
    observation: Any
    actions: list
    greedy: int
    default: int
    masked: list
    flagged: bool
    probabilities: list           # float64 softmax over the full padded row (padding = 0)
    logits: Any                   # float tensor row (valid candidates first)
    telemetry: Any = None
    prediction: dict | None = None

    def mask_top(self) -> int | None:
        n = len(self.actions)
        if n < 2:
            return None
        import torch
        row = self.logits[:n].clone()
        row[self.default] = -torch.inf
        return int(row.argmax().item())

    def stop(self) -> int:
        return next(i for i, a in enumerate(self.actions) if a.kind == "abstain")


# ---------------------------------------------------------------------------
# Agents: map decisions to interventions / fixed-mode choices
# ---------------------------------------------------------------------------

class FixedAgent:
    """Fixed deployments (A1 semantics): greedy, sampled, r_mask (= D0), r_sample."""
    meta_forward = False
    telemetry = False

    def __init__(self, mode: str):
        if mode not in FIXED_MODES:
            raise ValueError(mode)
        self.mode = mode

    def decide(self, decisions, episodes):
        return [self.mode for _ in decisions]


class LabelBaseAgent:
    """Base rollouts for label states: per-episode kind d0 (D0), eps (D0 with probability-eps
    random sample/mask_top interventions) or sampled (every step samples from π)."""
    meta_forward = False
    telemetry = True

    def __init__(self, eps: float):
        self.eps = eps

    def decide(self, decisions, episodes):
        out = []
        for d in decisions:
            ep = episodes[d.index]
            kind = ep.info["base_kind"]
            if kind == "d0":
                out.append("default")
            elif kind == "sampled":
                out.append("sample")
            else:
                u = "default"
                if ep.aux_rng.random() < self.eps:
                    u = ep.aux_rng.choice(("sample", "mask_top"))
                out.append(u)
        return out


class MetaAgent:
    """Metacontroller arms over the appraisal model.

    mode: appraisal_only (predict, behave D0), learned (argmax rule with margin),
    threshold (sample-step when P(success) < tau), clamp:<u> (learned timing, u replaced),
    random (matched rate; no model). ``zero_groups`` zeroes telemetry groups at
    evaluation; ``shuffle`` feeds each episode another active episode's telemetry."""
    telemetry = True

    def __init__(self, model=None, mode="learned", *, margin=math.inf, tau=0.0, rate=0.0, u_probs=None,
                 zero_groups=(), shuffle=False):
        if mode not in ("appraisal_only", "learned", "threshold", "random") and not mode.startswith("clamp:"):
            raise ValueError(mode)
        if mode.startswith("clamp:") and mode[6:] not in ADV_US:
            raise ValueError(mode)
        self.model, self.mode, self.margin, self.tau = model, mode, margin, tau
        self.rate, self.u_probs = rate, dict(u_probs or {"sample": 1.0})
        self.meta_forward = mode != "random"
        drop = tuple(zero_groups) + tuple(getattr(getattr(model, "spec", None), "drop_groups", ()) or ())
        self.input_mask = group_mask(drop)
        self.shuffle = shuffle

    def decide(self, decisions, episodes):
        if self.mode == "random":
            out = []
            for d in decisions:
                rng = episodes[d.index].aux_rng
                u = "default"
                if rng.random() < self.rate:
                    x, acc = rng.random(), 0.0
                    for name, p in sorted(self.u_probs.items()):
                        acc += p
                        u = name
                        if x < acc:
                            break
                out.append(u)
            return [self._available(u, d) for u, d in zip(out, decisions)]
        import torch
        x = torch.tensor([d.telemetry.vector() for d in decisions], dtype=torch.float32)
        x = x * torch.tensor(self.input_mask, dtype=torch.float32)
        if self.shuffle and len(decisions) > 1:
            x = torch.roll(x, 1, 0)
        h = torch.cat([episodes[d.index].meta_h if episodes[d.index].meta_h is not None
                       else self.model.initial_state(1) for d in decisions])
        with torch.no_grad():
            h, out = self.model.step(x, h)
        preds = {k: v.tolist() for k, v in out.items()}
        us = []
        for row, d in enumerate(decisions):
            ep = episodes[d.index]
            ep.meta_h = h[row:row + 1]
            ep.meta_forwards += 1
            adv = preds["adv"][row]
            d.prediction = {"p": preds["p_success"][row], "cost": preds["cost"][row], "q0": preds["q0"][row],
                            "adv": adv}
            u = "default"
            if self.mode == "threshold":
                u = "sample" if d.prediction["p"] < self.tau else "default"
            elif self.mode != "appraisal_only":
                gains = {name: (adv[i] if self._available(name, d) == name else -math.inf)
                         for i, name in enumerate(ADV_US)}
                best = max(gains, key=lambda k: (gains[k], -ADV_US.index(k)))
                if gains[best] > self.margin:
                    u = best if not self.mode.startswith("clamp:") else self.mode[6:]
            us.append(self._available(u, d))
        return us

    @staticmethod
    def _available(u, d):
        if u == "mask_top" and len(d.actions) < 2:
            return "default"
        return u


# ---------------------------------------------------------------------------
# Batched runner
# ---------------------------------------------------------------------------

def _stack_hidden(actor, episodes, active):
    import torch
    rows = [episodes[i].hidden for i in active]
    if all(r is None for r in rows):
        return None
    init = getattr(actor, "initial", None)
    filled = [r if r is not None else init.unsqueeze(0) for r in rows]
    return torch.cat(filled)


def run(actor, episodes, agent, *, device="cpu", neural_work_per_forward=1.0, meta_units=0.0,
        charge_meta=True, hook=None, timing=None):
    """Advance every episode until done or its decision cap. Choices (tested):
    greedy/sampled/r_mask/r_sample are identical to ``campaign04_deploy.deploy_episodes``;
    u = default is the r_mask choice. ``hook(episode, decision, u)`` runs after the agent
    decided and before any charge or environment step (label/branch snapshots)."""
    import torch
    from .campaign02_training import PublicInterfaceCapacityError, inverse_cdf_choice, policy_batch, prepare_frame, \
        score_batch
    timing = timing if timing is not None else {}
    for k in ("actor_s", "diagnostic_s", "telemetry_s", "meta_s", "env_s", "decisions"):
        timing.setdefault(k, 0.0)
    observations = [ep.env.observe() for ep in episodes]
    while True:
        active, public = [], []
        for i, ep in enumerate(episodes):
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
            logits, _, next_hidden = score_batch(actor, batch, _stack_hidden(actor, episodes, active))
            greedy = logits.argmax(-1).cpu().tolist()
            distribution = logits.double().softmax(-1).cpu().tolist()
            logits_cpu = logits.float().cpu()
        timing["actor_s"] += time.perf_counter() - t0
        decisions = []
        for row, i in enumerate(active):
            ep, actions = episodes[i], public[row][0]
            t0 = time.perf_counter()
            masked = ep.tracker.mask(actions)
            flagged = ep.tracker.is_flagged_state()
            n_masked = sum(masked)
            default = greedy[row]
            if n_masked and n_masked < len(actions):
                row_logits = logits_cpu[row, :len(actions)].clone()
                row_logits[[k for k, m in enumerate(masked) if m]] = float("-inf")
                default = int(row_logits.argmax().item())
            timing["diagnostic_s"] += time.perf_counter() - t0
            d = Decision(i, row, observations[i], actions, greedy[row], default, masked, flagged,
                         distribution[row], logits_cpu[row])
            if agent.telemetry:
                t0 = time.perf_counter()
                stats = policy_stats(distribution[row][:len(actions)], default, greedy[row], n_masked, flagged)
                d.telemetry = ep.recorder.features(observations[i], ep.tracker, stats)
                timing["telemetry_s"] += time.perf_counter() - t0
            decisions.append(d)
        t0 = time.perf_counter()
        us = agent.decide(decisions, episodes)
        timing["meta_s"] += time.perf_counter() - t0
        if next_hidden is not None:
            for row, i in enumerate(active):
                episodes[i].hidden = next_hidden[row:row + 1]
        for d, u in zip(decisions, us):
            ep = episodes[d.index]
            if hook is not None:
                hook(ep, d, u)
            choice, u_record = _choose(d, u, ep, inverse_cdf_choice)
            t0 = time.perf_counter()
            ep.env.charge_compute(neural_work_per_forward)
            if agent.meta_forward and charge_meta and meta_units:
                ep.env.charge_compute(meta_units)
            action = d.actions[choice]
            after = ep.env.step(action)
            timing["env_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            step_class = ep.tracker.update(after, action)
            timing["diagnostic_s"] += time.perf_counter() - t0
            if agent.telemetry:
                ep.recorder.update(after, action, step_class, u_record, neural_work_per_forward)
            ep.log.append({"u": u_record, "a": choice, "cls": step_class.cls,
                           **({"pred": d.prediction} if d.prediction is not None else {})})
            observations[d.index] = after
            ep.taken += 1
            timing["decisions"] += 1
    return episodes


def _choose(d: Decision, u: str, ep: Episode, inverse_cdf_choice):
    """(action index, intervention recorded in telemetry)."""
    tracker = ep.tracker
    n_masked = sum(d.masked)
    if u in ("default", "r_mask"):
        if n_masked and n_masked < len(d.actions):
            tracker.note_intervention(ep.taken, "mask", n_masked, d.default != d.greedy)
        elif n_masked:
            tracker.note_intervention(ep.taken, "fallback", n_masked, False)
        return d.default, "default"
    if u == "greedy":
        return d.greedy, "default"
    if u == "sampled":
        return inverse_cdf_choice(d.probabilities, ep.rng.random()), "sample"
    if u == "r_sample":
        if d.flagged:
            choice = inverse_cdf_choice(d.probabilities, ep.rng.random())
            tracker.note_intervention(ep.taken, "sample", n_masked, choice != d.greedy)
            return choice, "sample"
        return d.greedy, "default"
    if u == "sample":
        choice = inverse_cdf_choice(d.probabilities, ep.rng.random())
    elif u == "mask_top":
        choice = d.mask_top()
    elif u == "stop":
        choice = d.stop()
    else:
        raise ValueError(f"unknown intervention {u!r}")
    tracker.note_intervention(ep.taken, f"meta_{u}", n_masked, choice != d.default)
    return choice, u


def new_episode(env, *, cap=96, rng=None, aux_rng=None, telemetry=True, info=None) -> Episode:
    o = env.observe()
    return Episode(env, ProgressTracker(o), TelemetryRecorder(o) if telemetry else None, rng=rng, aux_rng=aux_rng,
                   cap=cap, info=dict(info or {}))


def outcome(env) -> dict:
    out = env.evaluate()
    return {"success": bool(out["verified_success"]), "utility": out["utility"], "cost": out["cost"],
            "steps": out["steps"], "compute_units": out["compute_units"]}


# ---------------------------------------------------------------------------
# Counterfactual branching
# ---------------------------------------------------------------------------

@dataclass
class BranchPoint:
    """A decision snapshot: the pre-decision environment/tracker/hidden clones plus the
    branch actions derived from the actor's distribution at that decision."""
    episode: int
    step: int
    stratum: str
    p_include: float
    env: Any
    tracker: Any
    hidden: Any
    actions: list
    u_index: dict               # u -> action index (default, mask_top (or None), stop)
    sample_indices: list        # K draws from π
    utility: float              # U_t (before the step-t charges)
    cost: float
    cap: int                    # decisions left for the branch (incl. the branched one)
    prediction: dict | None = None
    u_taken: str | None = None
    check_default: bool = False


def snapshot(ep: Episode, d: Decision, *, stratum, p_include, k_draws, draw_rng) -> BranchPoint:
    """Clone the episode at a decision (before any step-t charge). U_t = current utility;
    the episode is not verified yet (verification ends it), so cost_t = -U_t."""
    from .campaign02_training import inverse_cdf_choice
    env, hidden, tracker = clone_branch(ep.env, ep.hidden, ep.tracker)
    u_index = {"default": d.default, "mask_top": d.mask_top(), "stop": d.stop()}
    draws = [inverse_cdf_choice(d.probabilities, draw_rng.random()) for _ in range(k_draws)]
    utility = env.current_utility()
    return BranchPoint(d.index, ep.taken, stratum, p_include, env, tracker, hidden, d.actions, u_index, draws,
                       utility, -utility, ep.cap - ep.taken, prediction=d.prediction)


def run_branches(actor, points, *, neural_work_per_forward=1.0, device="cpu", batch=128, want_default=None,
                 timing=None):
    """Evaluate every needed branch of each point. Returns one dict per point with
    ΔU per u (default / mask_top / stop / per-draw sample), success and remaining cost
    under the default branch. ``want_default(point)`` decides whether the default branch
    is run (else the caller supplies it from the main line; D0 base episodes)."""
    jobs = []   # (point index, action index)
    for p_i, p in enumerate(points):
        need = set()
        if want_default is None or want_default(p):
            need.add(p.u_index["default"])
        if p.u_index["mask_top"] is not None:
            need.add(p.u_index["mask_top"])
        need.add(p.u_index["stop"])
        need.update(p.sample_indices)
        if want_default is not None and not want_default(p):
            need.discard(p.u_index["default"])  # supplied from the main line
        jobs += [(p_i, a) for a in sorted(need)]
    results = {}
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        eps = []
        for p_i, a in chunk:
            p = points[p_i]
            env, hidden, tracker = clone_branch(p.env, p.hidden, p.tracker)
            env.charge_compute(neural_work_per_forward)   # the step-t actor forward (charged once, as on the main line)
            action = p.actions[a]
            after = env.step(action)
            tracker.update(after, action)
            eps.append(Episode(env, tracker, None, hidden=hidden, cap=p.cap - 1))
        run(actor, eps, FixedAgent("r_mask"), device=device, neural_work_per_forward=neural_work_per_forward,
            timing=timing)
        for (p_i, a), ep in zip(chunk, eps):
            o = outcome(ep.env)
            results[(p_i, a)] = o
    out = []
    for p_i, p in enumerate(points):
        def du(a):
            r = results.get((p_i, a))
            return None if r is None else r["utility"] - p.utility
        d_idx = p.u_index["default"]
        rd = results.get((p_i, d_idx))
        out.append({
            "default": du(d_idx),
            "mask_top": du(p.u_index["mask_top"]) if p.u_index["mask_top"] is not None else None,
            "stop": du(p.u_index["stop"]),
            "sample_actions": list(p.sample_indices),
            "sample": [du(a) for a in p.sample_indices],   # None where a == default and default not run
            "success_default": None if rd is None else rd["success"],
            "cost_default": None if rd is None else rd["cost"] - p.cost,
            "branches": len({a for (q, a) in results if q == p_i}),
        })
    return out


def fill_default(result: dict, point: BranchPoint, du_default: float, success: bool, cost_rem: float) -> dict:
    """Supply the default branch from the main line (D0 base episodes: the main line after
    t IS the default continuation; tested equal to the cloned branch)."""
    r = dict(result)
    r["default"], r["success_default"], r["cost_default"] = du_default, success, cost_rem
    r["sample"] = [du_default if a == point.u_index["default"] else v
                   for a, v in zip(point.sample_indices, result["sample"])]
    return r


# ---------------------------------------------------------------------------
# Appraisal model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaSpec:
    input_dim: int = DIM
    hidden: int = 64
    drop_groups: tuple = ()
    telemetry_version: str = TELEMETRY_VERSION
    cost_scale: float = 10.0


def make_model(spec: MetaSpec):
    import torch
    from torch import nn

    class MetaAppraisal(nn.Module):
        def __init__(self, spec):
            super().__init__()
            if spec.hidden > 64:
                raise ValueError("appraisal GRU is registered at <= 64 hidden units")
            self.spec = spec
            self.gru = nn.GRU(spec.input_dim, spec.hidden, batch_first=True)
            self.success = nn.Linear(spec.hidden, 1)
            self.cost = nn.Linear(spec.hidden, 1)
            self.q0 = nn.Linear(spec.hidden, 1)
            self.adv = nn.Linear(spec.hidden, len(ADV_US))
            self.register_buffer("mask", torch.tensor(group_mask(spec.drop_groups), dtype=torch.float32))

        def initial_state(self, batch):
            return torch.zeros(batch, self.spec.hidden)

        def heads(self, h):
            logit = self.success(h).squeeze(-1)
            return {"p_success": torch.sigmoid(logit), "logit": logit,
                    "cost": self.cost(h).squeeze(-1) / self.spec.cost_scale, "q0": self.q0(h).squeeze(-1),
                    "adv": self.adv(h)}

        def step(self, x, h):
            """One decision for a batch: x [B, D], h [B, H] -> (h', heads)."""
            y, _ = self.gru((x * self.mask)[:, None], h[None].contiguous())
            h = y[:, 0]
            out = self.heads(h)
            out.pop("logit")
            return h, out

        def sequence(self, x):
            """x [B, T, D] (episodes padded at the end; the GRU is causal) -> heads per step."""
            y, _ = self.gru(x * self.mask)
            return self.heads(y)

    return MetaAppraisal(spec)


def parameter_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def meta_units_per_forward(meta_model, actor, neural_work_per_forward=1.0) -> float:
    """protocol-A1: the metacontroller forward is charged as neural work units in proportion
    to its parameter count relative to the actor's."""
    return neural_work_per_forward * parameter_count(meta_model) / parameter_count(actor)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelConfig:
    """Registered label-generation constants (trackc.md §3)."""
    k_draws: int = 4                      # sample-step draws per labelled point
    eps: float = 0.15                     # eps base: per-decision probability of a random sample/mask_top
    base_mix: tuple = (("d0", 0.5), ("eps", 0.25), ("sampled", 0.25))
    p_flagged: float = 0.5                # inclusion probability per stratum (online Bernoulli)
    p_low_margin: float = 0.12
    p_other: float = 0.04
    low_margin: float = 0.2               # top1 - top2 probability margin
    max_points: int = 4                   # per episode (points beyond the cap are skipped and counted)
    check_fraction: float = 0.1           # D0 episodes: also branch the default at this fraction of points
    max_steps: int = 96

    def p_include(self, stratum):
        return {"flagged": self.p_flagged, "low_margin": self.p_low_margin, "other": self.p_other}[stratum]


def base_kind(seed: int, cfg: LabelConfig) -> str:
    x = digest_seed("c-label-base-kind", seed) / 2 ** 64
    acc = 0.0
    for kind, p in cfg.base_mix:
        acc += p
        if x < acc:
            return kind
    return cfg.base_mix[-1][0]


def stratum_of(ep: Episode, d: Decision, cfg: LabelConfig) -> str:
    """Public-only stratum: flagged (R-mask holds a mask here, or the last step was no-progress),
    low_margin (actor top1-top2 < cfg.low_margin), other."""
    if d.flagged or (ep.recorder is not None and ep.recorder.last_class in NO_PROGRESS):
        return "flagged"
    if d.telemetry is not None and d.telemetry.get("margin") < cfg.low_margin:
        return "low_margin"
    return "other"


def generate_labels(actor, envs, seeds, cfg: LabelConfig = LabelConfig(), *, neural_work_per_forward=1.0,
                    device="cpu", branch_batch=128):
    """Counterfactual labels for one batch of worlds. Returns (episodes, points, stats).

    Per episode: telemetry sequence, interventions taken, U_t per decision, the final
    outcome (for D0 base episodes the main line after any t IS the default continuation,
    so every step carries default targets). Per point: ΔU for default / mask_top / stop and
    K sample draws (continuation D0), success and remaining cost under the default branch,
    stratum and inclusion probability."""
    timing = {}
    cpu0 = time.perf_counter()
    episodes = []
    for env, seed in zip(envs, seeds):
        kind = base_kind(seed, cfg)
        ep = new_episode(env, cap=cfg.max_steps, rng=random.Random(digest_seed("c-label-sample", seed)),
                         aux_rng=random.Random(digest_seed("c-label-eps", seed)),
                         info={"seed": seed, "base_kind": kind, "feats": [], "U": [], "points": [], "capped": 0,
                               "select_rng": random.Random(digest_seed("c-label-select", seed)),
                               "draw_rng": random.Random(digest_seed("c-label-draws", seed)),
                               "check_rng": random.Random(digest_seed("c-label-check", seed))})
        episodes.append(ep)
    points: list[BranchPoint] = []

    def hook(ep, d, u):
        info = ep.info
        info["feats"].append(d.telemetry.values)
        info["U"].append(ep.env.current_utility())
        s = stratum_of(ep, d, cfg)
        p = cfg.p_include(s)
        if info["select_rng"].random() < p:
            if len(info["points"]) >= cfg.max_points:
                info["capped"] += 1
                return
            bp = snapshot(ep, d, stratum=s, p_include=p, k_draws=cfg.k_draws, draw_rng=info["draw_rng"])
            bp.u_taken = u
            bp.check_default = info["base_kind"] == "d0" and info["check_rng"].random() < cfg.check_fraction
            info["points"].append(len(points))
            points.append(bp)

    run(actor, episodes, LabelBaseAgent(cfg.eps), device=device, neural_work_per_forward=neural_work_per_forward,
        hook=hook, timing=timing)
    finals = [outcome(ep.env) for ep in episodes]
    base_wall = time.perf_counter() - cpu0
    cpu1 = time.perf_counter()
    kinds = [ep.info["base_kind"] for ep in episodes]
    branch = run_branches(actor, points, neural_work_per_forward=neural_work_per_forward, device=device,
                          batch=branch_batch, want_default=lambda p: kinds[p.episode] != "d0" or p.check_default)
    branch_wall = time.perf_counter() - cpu1
    out_points = []
    checks = [0, 0]
    for p, r in zip(points, branch):
        check = -1
        if kinds[p.episode] == "d0":
            f = finals[p.episode]
            main = (f["utility"] - p.utility, f["success"], f["cost"] - p.cost)
            if p.check_default:
                check = int((r["default"], r["success_default"], r["cost_default"]) == main)
                checks[0] += 1
                checks[1] += check
            r = fill_default(r, p, *main)
        out_points.append({"episode": p.episode, "step": p.step, "stratum": p.stratum, "p_include": p.p_include,
                           "u_taken": p.u_taken,
                           "dU": {"default": r["default"], "mask_top": r["mask_top"], "stop": r["stop"]},
                           "dU_sample": r["sample"], "sample_actions": r["sample_actions"],
                           "sample_is_default": [a == p.u_index["default"] for a in p.sample_indices],
                           "success_default": r["success_default"], "cost_default": r["cost_default"],
                           "default_check": check, "branches": r["branches"]})
    out_eps = []
    for ep, f in zip(episodes, finals):
        info = ep.info
        out_eps.append({"seed": info["seed"], "base_kind": info["base_kind"], "feats": info["feats"],
                        "u": [x["u"] for x in ep.log], "cls": [x["cls"] for x in ep.log], "U": info["U"],
                        "final": f, "truncated": not ep.env.observe().done, "capped_points": info["capped"]})
    stats = {"episodes": len(episodes), "decisions": int(timing["decisions"]), "points": len(points),
             "branches": sum(r["branches"] for r in branch), "default_checks": checks[0],
             "default_check_matches": checks[1], "base_wall_seconds": base_wall, "branch_wall_seconds": branch_wall,
             "base_timing": timing}
    return out_eps, out_points, stats
