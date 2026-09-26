"""Extended-03 P2a Part D: collapse audit on archived X1 RL checkpoints (no training).

Registered in research/campaigns/extended-03/protocol-P2a.md (Part D). Descriptive
only: it localizes where the collapse happens and makes no causal claim.

Subjects (read-only): for lineages X1-r0/r1/r2, the shared bootstrap bank endpoint
and the RL tranche endpoints at attempts 18-29 (attempt 17 is loaded only as the
previous-tranche reference of attempt 18). Every checkpoint is hash-verified
against configs/campaign03/p1-banks.json (bootstrap) or the run's state.json.

Probe worlds: 64 fixed IID-group worlds (P1 SEALED iid_f0 / iid_f2 definitions,
32 each) on fresh probe seeds 1,990,000,000+ (iid_f0) and 1,990,100,000+ (iid_f2),
address namespace ``e03p2a-partd-probe``. World order interleaves f0/f2 so that
every 8-world sampled batch holds 4 of each.

Measurements per checkpoint (protocol Part D items 1-4):
 1. greedy (argmax, as P1 evaluation) and sampled (fixed torch seeds, 4 samples
    per world, training's ``batched_on_policy`` in batches of 8 = training
    batch_size) rollouts: success, utility, action-kind mix, idempotent-repeat
    rate, short-cycle rate, no-progress rate, steps-to-cap rate;
 2. KL(pi_boot || pi_ckpt) and KL(pi_prev || pi_ckpt) with training's masked KL,
    on the states of the bootstrap's greedy rollouts and on the checkpoint's own
    greedy rollouts (plus greedy-argmax disagreement on the same states);
 3. critic: value vs realized undiscounted return-to-go on the checkpoint's own
    sampled rollouts (MSE, bias) and the mean raw / batch-standardized advantage
    (training's computation) of loop vs productive vs other actions;
 4. gradient norms of the actor, critic, entropy and KL terms of
    ``actor_critic_objective`` (weighted as in the loss) on fixed sampled batches,
    on shared (observation+context encoder) and head parameters, plus pairwise
    cosines on shared parameters. No optimizer step.

Operational definitions (protocol-P2a.md "Metrics"; this module's reading):
 - accepted action: feedback status not in {"rejected", "invalid_input"};
 - public state: drafts (primitive, problem, depends_on), record handles, retrieved
   handles, inspected knowledge (items, requirements, map), pending choices,
   committed selection/assignment, pending assignment, position, verified flag,
   requirements_version. Step counters, remaining budgets, travel, feedback,
   the attempt log and prices are excluded;
 - public decision state (short cycles): drafts (primitive, problem), pending
   choices, committed selection/assignment, pending assignment, retrieved handles,
   position;
 - idempotent repeat at step t: action_key(a_t) equals the action_key of the most
   recent accepted action before t, and the public state after t equals the one
   before t;
 - short-cycle step t: a_t accepted and the decision state after t equals the
   decision state before some step j in [t-5, t] (the 6 preceding states,
   including the state a_t was taken from), with no solver call and no new
   inspection (a target not previously inspected) among steps j..t;
 - no-progress step: idempotent repeat or short-cycle step;
 - step-cap episode: not verified and the world step limit was exhausted.
 - productive action: call, use_return, move, or a successful commit_pending /
   commit_assignment. Action classes for item 3 are "loop" (a no-progress step),
   "productive" (productive and not loop) and "other".

CLI (full run; the root launches it under the job wrapper):
  python research/tools/campaign03_p2a_collapse_audit.py --output DIR [--processes 3]
Smoke: add --lineages 2 --attempts 29 --worlds 8 --samples 2.
Budget reduction (protocol: fewer sampled repeats first): --samples 2 or 1.
``--report DIR`` rebuilds partd.json / partd.md from existing lineage files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = "/home/brand/tensegra-campaign03/results"
PROBE_SEED_START = 1_990_000_000
PROBE_CONDITION_STRIDE = 100_000
PROBE_CONDITIONS = ("iid_f0", "iid_f2")
PROBE_WORLDS_PER_CONDITION = 32
SAMPLE_SEED_BASE = 1_990_500_000  # torch.manual_seed for sampled batches (not world seeds)
ADDRESS_NAMESPACE = "e03p2a-partd-probe"
ATTEMPTS = tuple(range(18, 30))
LINEAGES = (0, 1, 2)
SAMPLES = 4
BATCH = 8  # P1 RL batch_size; sampled rollouts and advantage standardization use it
CYCLE_WINDOW = 6
REJECTED = ("rejected", "invalid_input")
PRODUCTIVE_KINDS = ("call", "use_return", "move")
COMMIT_KINDS = ("commit_pending", "commit_assignment")


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Probe worlds
# ---------------------------------------------------------------------------

def probe_worlds(per_condition: int = PROBE_WORLDS_PER_CONDITION):
    """[(condition, seed, world kwargs)] interleaved f0, f2, f0, f2, ..."""
    sealed = dict(_load_tool("campaign03_p1_configs").SEALED)
    columns = [[(name, PROBE_SEED_START + PROBE_CONDITION_STRIDE * i + k, dict(sealed[name]))
                for k in range(per_condition)] for i, name in enumerate(PROBE_CONDITIONS)]
    return [w for pair in zip(*columns) for w in pair]


# ---------------------------------------------------------------------------
# Loop definitions (pure; operate on per-step records)
# ---------------------------------------------------------------------------

def public_signature(o) -> str:
    return _digest({
        "problems": {n: [p["primitive"], p["problem"], p.get("depends_on")] for n, p in o.problems.items()},
        "records": sorted(r["handle"] for r in o.records), "retrieved": sorted(o.retrieved),
        "known": sorted(o.known_items), "requirements_known": o.requirements is not None,
        "map_known": o.known_edges is not None, "pending": sorted(o.pending), "selected": list(o.selected),
        "pending_assignment": o.pending_assignment, "assignment": o.assignment, "position": o.position,
        "verified": o.verified, "requirements_version": o.requirements_version})


def decision_signature(o) -> str:
    return _digest({
        "drafts": {n: [p["primitive"], p["problem"]] for n, p in o.problems.items()},
        "pending": sorted(o.pending), "selected": list(o.selected), "pending_assignment": o.pending_assignment,
        "assignment": o.assignment, "retrieved": sorted(o.retrieved), "position": o.position})


def is_new_inspection(before, action) -> bool:
    if action.kind != "inspect":
        return False
    target = action.arguments["target"]
    if target == "requirements":
        return before.requirements is None
    if target == "map":
        return before.known_edges is None
    return target not in before.known_items


def step_record(before, action, after) -> dict:
    from tensegra.campaign03_depworld import action_key
    status = after.feedback.get("status")
    return {"kind": action.kind, "key": action_key(action), "status": status,
            "accepted": status not in REJECTED,
            "success": status == "success", "pub_before": public_signature(before), "pub_after": public_signature(after),
            "dec_before": decision_signature(before), "dec_after": decision_signature(after),
            "call": action.kind == "call", "new_inspection": is_new_inspection(before, action)}


def loop_flags(steps: list[dict], window: int = CYCLE_WINDOW) -> list[dict]:
    """Idempotent-repeat / short-cycle / productive flags per step (see module doc)."""
    flags, last_accepted_key = [], None
    for t, s in enumerate(steps):
        idem = (last_accepted_key is not None and s["key"] == last_accepted_key
                and s["pub_after"] == s["pub_before"])
        cycle = False
        if s["accepted"]:
            for j in range(max(0, t - window + 1), t + 1):
                if steps[j]["dec_before"] == s["dec_after"] and not any(
                        steps[k]["call"] or steps[k]["new_inspection"] for k in range(j, t + 1)):
                    cycle = True
                    break
        productive = s["kind"] in PRODUCTIVE_KINDS or (s["kind"] in COMMIT_KINDS and s["success"])
        loop = idem or cycle
        flags.append({"idempotent": idem, "cycle": cycle, "loop": loop, "productive": productive,
                      "action_class": "loop" if loop else ("productive" if productive else "other")})
        if s["accepted"]:
            last_accepted_key = s["key"]
    return flags


class Recording:
    """Environment proxy: records public step records for the loop definitions."""

    def __init__(self, env):
        self.env, self.steps, self._last = env, [], None

    def observe(self):
        self._last = self.env.observe()
        return self._last

    def charge_compute(self, units):
        self.env.charge_compute(units)

    def step(self, action):
        before = self._last
        after = self.env.step(action)
        self.steps.append(step_record(before, action, after))
        self._last = after
        return after

    def evaluate(self):
        return self.env.evaluate()

    @property
    def remaining_steps(self):
        return self._last.remaining_steps


def episode_summary(env: Recording, outcome: dict) -> dict:
    flags = loop_flags(env.steps)
    kinds = {}
    for s in env.steps:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    success = bool(outcome["verified_success"])
    return {"success": success, "utility": float(outcome["utility"]), "steps": len(env.steps),
            "cap": (not success) and env.remaining_steps == 0, "kinds": kinds,
            "idempotent": sum(f["idempotent"] for f in flags), "cycle": sum(f["cycle"] for f in flags),
            "no_progress": sum(f["loop"] for f in flags), "flags": flags}


def aggregate_episodes(episodes: list[dict]) -> dict:
    n, steps = len(episodes), sum(e["steps"] for e in episodes)
    kinds = {}
    for e in episodes:
        for k, v in e["kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    rate = (lambda x: x / steps) if steps else (lambda x: None)
    return {"episodes": n, "success": sum(e["success"] for e in episodes) / n,
            "utility": sum(e["utility"] for e in episodes) / n,
            "mean_steps": steps / n, "steps_to_cap_rate": sum(e["cap"] for e in episodes) / n,
            "idempotent_repeat_rate": rate(sum(e["idempotent"] for e in episodes)),
            "short_cycle_rate": rate(sum(e["cycle"] for e in episodes)),
            "no_progress_rate": rate(sum(e["no_progress"] for e in episodes)),
            "no_progress_per_episode": sum(e["no_progress"] for e in episodes) / n,
            "action_mix": {k: v / steps for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])} if steps else {}}


# ---------------------------------------------------------------------------
# Training-identical computations
# ---------------------------------------------------------------------------

def masked_kl(reference_logits, logits, valid):
    """KL(reference || current) per row, exactly as batched_on_policy computes it."""
    p_ref = reference_logits.log_softmax(-1).masked_fill(~valid, 0.0)
    p_cur = logits.log_softmax(-1).masked_fill(~valid, 0.0)
    return (p_ref.exp() * (p_ref - p_cur)).sum(-1)


def training_advantages(episodes_terms):
    """Per-decision (value, return-to-go, raw advantage, standardized advantage).

    Mirrors actor_critic_objective: undiscounted return-to-go of utility deltas,
    raw advantage = G - V (detached), standardization over all decisions of the
    batch with population std + 1e-6."""
    rows = []
    for episode in episodes_terms:
        future, part = 0.0, []
        for logp, value, entropy, reward_delta in reversed(episode):
            future += reward_delta
            v = float(value.detach())
            part.append({"value": v, "return_to_go": future, "raw": future - v})
        rows.append(list(reversed(part)))
    flat = [x["raw"] for part in rows for x in part]
    if flat:
        mean = sum(flat) / len(flat)
        std = math.sqrt(sum((x - mean) ** 2 for x in flat) / len(flat)) + 1e-6
        for part in rows:
            for x in part:
                x["normalized"] = (x["raw"] - mean) / std
    return rows


def critic_summary(decisions: list[dict]) -> dict:
    """decisions: {value, return_to_go, raw, normalized, action_class}."""
    def stats(rows):
        if not rows:
            return {"n": 0}
        return {"n": len(rows), "mean_raw_advantage": statistics.fmean(r["raw"] for r in rows),
                "mean_normalized_advantage": statistics.fmean(r["normalized"] for r in rows),
                "positive_raw_fraction": sum(r["raw"] > 0 for r in rows) / len(rows),
                "mean_value": statistics.fmean(r["value"] for r in rows),
                "mean_return_to_go": statistics.fmean(r["return_to_go"] for r in rows)}
    errors = [r["value"] - r["return_to_go"] for r in decisions]
    out = {"decisions": len(decisions),
           "critic_mse": statistics.fmean(e * e for e in errors) if errors else None,
           "critic_bias": statistics.fmean(errors) if errors else None,
           "by_class": {c: stats([r for r in decisions if r["action_class"] == c]) for c in ("loop", "productive", "other")}}
    loop, prod = out["by_class"]["loop"], out["by_class"]["productive"]
    out["loop_minus_productive_normalized"] = (loop["mean_normalized_advantage"] - prod["mean_normalized_advantage"]
                                               if loop["n"] and prod["n"] else None)
    out["loop_minus_productive_raw"] = (loop["mean_raw_advantage"] - prod["mean_raw_advantage"]
                                        if loop["n"] and prod["n"] else None)
    return out


PARAMETER_GROUPS = {"shared": ("observation.", "context."), "actor_head": ("candidate.", "scorer."),
                    "critic_head": ("value.",)}


def parameter_group(name: str) -> str:
    for group, prefixes in PARAMETER_GROUPS.items():
        if name.startswith(prefixes):
            return group
    raise ValueError(f"Unassigned parameter {name}")


def gradient_terms(model, episodes_terms, kls, config) -> dict:
    """Gradient norms of each weighted objective term; no optimizer step."""
    import torch
    from tensegra.campaign02_training import actor_critic_objective
    loss, parts = actor_critic_objective(episodes_terms, config, kls)
    terms = {"actor": parts["policy_loss"], "critic": config.value_weight * parts["critic_decision_mean_loss"],
             "entropy": -config.entropy_weight * parts["entropy"]}
    if "kl_to_round_start" in parts:
        terms["kl"] = config.kl_weight * parts["kl_to_round_start"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = {n: parameter_group(n) for n, _ in named}
    vectors = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
        vectors[name] = {g: torch.cat([(gr if gr is not None else torch.zeros_like(p)).reshape(-1)
                                       for (n, p), gr in zip(named, grads) if groups[n] == g])
                         for g in PARAMETER_GROUPS}
    # The loss is exactly the sum of the weighted terms, so its gradient is the sum of theirs.
    vectors["total"] = {g: sum(vectors[name][g] for name in terms) for g in PARAMETER_GROUPS}
    norms = {name: {**{g: float(v.norm()) for g, v in vec.items()},
                    "all": float(torch.sqrt(sum(v.square().sum() for v in vec.values())))}
             for name, vec in vectors.items()}
    cos = {}
    names = list(terms) + ["total"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = vectors[a]["shared"], vectors[b]["shared"]
            denominator = float(x.norm() * y.norm())
            cos[f"{a}|{b}"] = float(x @ y) / denominator if denominator > 0 else None
    return {"loss": float(loss.detach()), "parts": {k: float(v.detach()) for k, v in parts.items()},
            "norms": norms, "shared_cosine": cos, "would_clip": norms["total"]["all"] > config.gradient_clip,
            "gradient_clip": config.gradient_clip}


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def greedy_probe(driver, environments, models: dict, pairs, *, device="cpu", max_steps=96,
                 neural_work_per_forward=1.0, driver_name=None):
    """Greedy (argmax) episodes of ``driver`` as P1 evaluation, plus per-state
    masked KL(models[ref] || models[cur]) and argmax disagreement for each pair
    on the visited public states. Lightweight (non-recurrent) policies only."""
    import torch
    from tensegra.campaign02_training import policy_batch, prepare_active, score_batch
    for m in [driver, *models.values()]:
        if m.config.family != "lightweight":
            raise ValueError("State-wise KL on stored states needs a non-recurrent policy")
    observations = [env.observe() for env in environments]
    unsupported, kls, disagree = {}, {p: [] for p in pairs}, {p: [] for p in pairs}
    with torch.no_grad():
        for step in range(max_steps):
            active, public = prepare_active(driver, observations, unsupported)
            if not active:
                break
            batch = policy_batch(driver, [frame for _, frame in public], device)
            logits, _, _ = score_batch(driver, batch)
            chosen = logits.argmax(-1).cpu().tolist()
            scored = {name: logits if name == driver_name else score_batch(m, batch)[0]
                      for name, m in models.items() if any(name in p for p in pairs)}
            for ref, cur in pairs:
                kls[(ref, cur)].extend(masked_kl(scored[ref], scored[cur], batch[2]).cpu().tolist())
                disagree[(ref, cur)].extend((scored[ref].argmax(-1) != scored[cur].argmax(-1)).cpu().tolist())
            for row, index in enumerate(active):
                environments[index].charge_compute(neural_work_per_forward)
                observations[index] = environments[index].step(public[row][0][chosen[row]])
    if unsupported:
        raise ValueError(f"Unsupported public interface in probe: {unsupported}")
    return [env.evaluate() for env in environments], kls, disagree


def kl_summary(values, disagreements) -> dict:
    if not values:
        return {"states": 0}
    ordered = sorted(values)
    return {"states": len(values), "mean": statistics.fmean(values), "median": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(.9 * len(ordered)))], "max": ordered[-1],
            "argmax_disagreement": sum(disagreements) / len(disagreements)}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def subject_bindings(lineage: int, attempts, results=RESULTS, banks_path=ROOT / "configs/campaign03/p1-banks.json"):
    """{name: {path, sha256}} for bootstrap, attempts and their previous tranches."""
    bank = json.loads(Path(banks_path).read_text())[f"x1-r{lineage}"]
    run = Path(results) / f"p1-rl-x1-r{lineage}"
    state = json.loads((run / "state.json").read_text())
    by_attempt = {}
    for row in state["allocations"]:
        attempt = int(Path(row["checkpoint"]).stem.rsplit("-", 1)[1])
        by_attempt[attempt] = {"path": str(run / row["checkpoint"]), "sha256": row["checkpoint_sha256"]}
    needed = sorted(set(attempts) | {a - 1 for a in attempts if a - 1 >= 0})
    missing = [a for a in needed if a not in by_attempt]
    if missing:
        raise ValueError(f"Attempts missing from state.json: {missing}")
    out = {"boot": {"path": bank["path"], "sha256": bank["sha256"]}}
    out.update({f"a{a}": by_attempt[a] for a in needed})
    return out


RL_KEYS = ("method", "learning_rate", "entropy_weight", "kl_weight", "value_weight", "advantage_normalization",
           "policy_loss_reduction", "gradient_clip", "batch_size", "max_steps", "rollout_mode",
           "neural_work_per_forward", "bptt_steps")


def rl_train_config(lineage: int, config_dir=ROOT / "configs/campaign03"):
    """The P1 X1 RL TrainConfig (train block + member row 0), on CPU."""
    from tensegra.campaign02_training import TrainConfig
    raw = json.loads((Path(config_dir) / f"p1-rl-x1-r{lineage}.json").read_text())
    return TrainConfig(**{**raw["train"], **raw["member_hyperparameters"][0], "method": "actor_critic", "device": "cpu"})


def load_policy(binding, verify=True):
    import torch
    from tensegra.campaign02_population import build_policy
    from tensegra.campaign02_training import TrainConfig
    if verify and file_sha256(binding["path"]) != binding["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {binding['path']}")
    saved = torch.load(binding["path"], map_location="cpu", weights_only=False)
    model = build_policy(saved["policy_config"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    config = TrainConfig(**{**saved["config"], "device": "cpu"})
    return model, config, {"updates": saved["updates"], "config": saved["config"]}


# ---------------------------------------------------------------------------
# Per-lineage audit
# ---------------------------------------------------------------------------

def audit_lineage(lineage: int, *, attempts=ATTEMPTS, worlds=None, samples=SAMPLES, grad_batches=1,
                  results=RESULTS, make_executor=None, bindings=None, log=print, partial_path=None) -> dict:
    import torch
    from tensegra.campaign02_training import batched_on_policy, independent_address_seed
    from tensegra.campaign03_depworld import DepWorkshop, generate_depworld
    torch.set_num_threads(1)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    worlds = probe_worlds() if worlds is None else worlds
    bindings = bindings or subject_bindings(lineage, attempts, results)
    models, configs, meta = {}, {}, {}
    for name, binding in bindings.items():
        models[name], configs[name], meta[name] = load_policy(binding)
    rl_config = rl_train_config(lineage)
    if rl_config.batch_size != BATCH or rl_config.family != "lightweight":
        raise ValueError("Audit assumes the P1 lightweight policy with training batch size 8")
    max_steps, nwpf = rl_config.max_steps, rl_config.neural_work_per_forward
    # Every audited RL checkpoint must carry the registered RL hyperparameters.
    config_check = {name: {k: [getattr(c, k), getattr(rl_config, k)] for k in RL_KEYS if getattr(c, k) != getattr(rl_config, k)}
                    for name, c in configs.items() if name != "boot"}
    if any(config_check.values()):
        raise ValueError(f"Checkpoint RL configuration differs from the P1 RL config: {config_check}")
    executor = make_executor()
    specs = [generate_depworld(seed, **kwargs) for _, seed, kwargs in worlds]

    def envs(indices):
        return [Recording(DepWorkshop(specs[i], executor=executor,
                                      address_seed=independent_address_seed(worlds[i][1], ADDRESS_NAMESPACE)))
                for i in indices]

    subjects = ["boot"] + [f"a{a}" for a in attempts]
    previous = {f"a{a}": f"a{a - 1}" for a in attempts}

    # (a) states of the bootstrap's greedy rollouts, scored by every subject/previous model.
    t0 = time.process_time()
    boot_envs = envs(range(len(worlds)))
    pairs = [("boot", s) for s in subjects] + [(previous[s], s) for s in subjects if s != "boot"]
    boot_outcomes, boot_kls, boot_dis = greedy_probe(models["boot"], boot_envs, models, pairs, max_steps=max_steps,
                                                     neural_work_per_forward=nwpf, driver_name="boot")
    boot_state_cpu = time.process_time() - t0
    log(f"lineage r{lineage}: bootstrap-state KL pass {boot_state_cpu:.1f}s cpu")

    records = []
    for name in subjects:
        t0, w0 = time.process_time(), time.perf_counter()
        model = models[name]
        prev = previous.get(name)
        # 1+2(b): own greedy rollouts with KL to bootstrap / previous tranche on own states.
        # The bootstrap's own greedy rollout is the bootstrap-state pass itself (deterministic).
        if name == "boot":
            greedy_envs, outcomes, own_kls, own_dis = boot_envs, boot_outcomes, boot_kls, boot_dis
        else:
            greedy_envs = envs(range(len(worlds)))
            own_pairs = [("boot", name), (prev, name)]
            outcomes, own_kls, own_dis = greedy_probe(model, greedy_envs, models, own_pairs, max_steps=max_steps,
                                                      neural_work_per_forward=nwpf, driver_name=name)
        greedy = [episode_summary(e, o) for e, o in zip(greedy_envs, outcomes)]
        greedy_cpu, sampled_cpu, gradient_cpu = time.process_time() - t0, 0.0, 0.0
        # 1+3+4: sampled rollouts, training rollout function, batches of 8.
        reference = models[prev] if prev else model  # tranche-start reference (attempt k-1); boot: itself
        sampled, decisions, gradients = [], [], []
        batches = [list(range(b, min(b + BATCH, len(worlds)))) for b in range(0, len(worlds), BATCH)]
        for s in range(samples):
            for b, indices in enumerate(batches):
                with_grad = s == 0 and b < grad_batches
                phase = time.process_time()
                torch.manual_seed(SAMPLE_SEED_BASE + 1000 * s + b)
                batch_envs = envs(indices)
                if with_grad:
                    for p in model.parameters():
                        p.requires_grad_(True)
                with torch.set_grad_enabled(with_grad):
                    # The KL reference only enters the gradient measurement; it does not change sampling.
                    rollout = batched_on_policy(model, batch_envs, device="cpu", max_steps=max_steps,
                        neural_work_per_forward=nwpf, bptt_steps=rl_config.bptt_steps, sample=True,
                        reference=reference if with_grad else None)
                results_, terms = rollout[:2]
                kls = rollout[2] if with_grad else None
                if with_grad:
                    middle = time.process_time()
                    sampled_cpu += middle - phase
                    gradients.append(gradient_terms(model, terms, kls, rl_config))
                    for p in model.parameters():
                        p.requires_grad_(False)
                    gradient_cpu += time.process_time() - middle
                    phase = time.process_time()
                advantage_rows = training_advantages(terms)
                for env, result, rows in zip(batch_envs, results_, advantage_rows):
                    summary = episode_summary(env, result["outcome"])
                    if len(rows) != len(summary["flags"]):
                        raise AssertionError("Decision terms and recorded steps are misaligned")
                    for row, flag in zip(rows, summary["flags"]):
                        decisions.append({**row, "action_class": flag["action_class"]})
                    sampled.append(summary)
                del terms, kls
                sampled_cpu += time.process_time() - phase
        record = {"subject": name, "attempt": None if name == "boot" else int(name[1:]),
                  "checkpoint": {k: v for k, v in bindings[name].items()}, "checkpoint_updates": meta[name]["updates"],
                  "previous_tranche": prev,
                  "checkpoint_hyperparameters": {k: meta[name]["config"].get(k) for k in RL_KEYS},
                  "measurement_hyperparameters": {k: getattr(rl_config, k) for k in RL_KEYS},
                  "greedy": aggregate_episodes(greedy), "sampled": aggregate_episodes(sampled),
                  "kl": {"bootstrap_states": {
                             "boot_to_ckpt": kl_summary(boot_kls[("boot", name)], boot_dis[("boot", name)]) if name != "boot" else None,
                             "prev_to_ckpt": kl_summary(boot_kls[(prev, name)], boot_dis[(prev, name)]) if prev else None},
                         "own_greedy_states": {
                             "boot_to_ckpt": kl_summary(own_kls[("boot", name)], own_dis[("boot", name)]),
                             "prev_to_ckpt": kl_summary(own_kls[(prev, name)], own_dis[(prev, name)]) if prev else None}},
                  "critic": critic_summary(decisions),
                  "gradients": {"batches": gradients, "mean": mean_gradients(gradients),
                                "kl_reference": prev or "boot (self; tranche-start reference of attempt 0)",
                                "batch_worlds": [[worlds[i][1] for i in batches[b]] for b in range(min(grad_batches, len(batches)))]},
                  "cost": {"process_cpu_seconds": time.process_time() - t0, "wall_seconds": time.perf_counter() - w0,
                           "greedy_cpu": greedy_cpu, "sampled_cpu": sampled_cpu, "gradient_cpu": gradient_cpu,
                           "greedy_steps": sum(e["steps"] for e in greedy), "sampled_steps": sum(e["steps"] for e in sampled)}}
        records.append(record)
        if partial_path is not None:
            temporary = Path(partial_path).with_suffix(".tmp")
            temporary.write_text(json.dumps({"lineage": lineage, "partial": True, "subjects": records,
                                             "bindings": bindings}, sort_keys=True))
            temporary.replace(partial_path)
        log(f"lineage r{lineage} {name}: greedy {record['greedy']['success']:.3f} sampled "
            f"{record['sampled']['success']:.3f} cost {json.dumps(record['cost'])}")
    return {"lineage": lineage, "subjects": records, "bindings": bindings,
            "probe": {"worlds": [[c, s] for c, s, _ in worlds], "samples_per_world": samples,
                      "sample_seed_base": SAMPLE_SEED_BASE, "address_namespace": ADDRESS_NAMESPACE,
                      "grad_batches": grad_batches, "batch_size": BATCH},
            "measurement_config": asdict(rl_config),
            "cost": {"process_cpu_seconds": time.process_time() - start_cpu,
                     "wall_seconds": time.perf_counter() - start_wall, "bootstrap_state_pass_cpu": boot_state_cpu,
                     "scope": "this process only; persistent solver worker CPU excluded (job receipt is inclusive)"}}


def mean_gradients(batches):
    if not batches:
        return None
    def mean(values):
        values = [v for v in values if v is not None]
        return statistics.fmean(values) if values else None
    norms = {term: {g: mean([b["norms"][term][g] for b in batches]) for g in batches[0]["norms"][term]}
             for term in batches[0]["norms"]}
    cos = {k: mean([b["shared_cosine"][k] for b in batches]) for k in batches[0]["shared_cosine"]}
    return {"norms": norms, "shared_cosine": cos, "would_clip_fraction": mean([float(b["would_clip"]) for b in batches]),
            "parts": {k: mean([b["parts"][k] for b in batches]) for k in batches[0]["parts"]}}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def _g(x):
    return "—" if x is None else f"{x:.2e}"


def locate(lineage: dict) -> dict:
    """Plot-free reading aids (descriptive thresholds, not registered tests)."""
    rows = lineage["subjects"]
    boot = rows[0]
    attempts = [r for r in rows if r["attempt"] is not None]
    def first(pred):
        return next((r["attempt"] for r in attempts if pred(r)), None)
    greedy_drop = first(lambda r: r["greedy"]["success"] < boot["greedy"]["success"] - .20)
    sampled_drop = first(lambda r: r["sampled"]["success"] < boot["sampled"]["success"] - .20)
    diverge = first(lambda r: abs(r["greedy"]["success"] - r["sampled"]["success"]) >= .10)
    kl_series = [(r["attempt"], r["kl"]["bootstrap_states"]["boot_to_ckpt"].get("mean")) for r in attempts]
    first_kl = kl_series[0][1] if kl_series else None
    kl_double = next((a for a, v in kl_series if first_kl and v is not None and v >= 2 * first_kl), None)
    jumps = [(b[0], b[1] - a[1]) for a, b in zip(kl_series, kl_series[1:]) if a[1] is not None and b[1] is not None]
    largest_jump = max(jumps, key=lambda x: x[1])[0] if jumps else None
    collapse = greedy_drop
    before = [r for r in attempts if collapse is None or r["attempt"] < collapse]
    favour = [r["critic"]["loop_minus_productive_normalized"] for r in before
              if r["critic"]["loop_minus_productive_normalized"] is not None]
    return {"greedy_success_drop_attempt": greedy_drop, "sampled_success_drop_attempt": sampled_drop,
            "greedy_sampled_divergence_attempt": diverge, "kl_boot_doubles_attempt": kl_double,
            "kl_boot_largest_increase_attempt": largest_jump,
            "pre_collapse_attempts": [r["attempt"] for r in before],
            "pre_collapse_loop_minus_productive_normalized_advantage": favour,
            "pre_collapse_loop_advantage_exceeds_productive": sum(v > 0 for v in favour) if favour else None,
            "thresholds": "descriptive reading aids: drop = success < bootstrap - 0.20; divergence = |greedy - sampled| >= 0.10; KL doubling vs the first audited attempt"}


def lineage_markdown(lineage: dict) -> str:
    r = lineage["lineage"]
    out = [f"## X1-r{r}", "", "### Rollouts (greedy | sampled)", "",
           "| ckpt | g succ | g util | g cap | g idem | g cycle | s succ | s util | s cap | s idem | s cycle | g top kinds |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in lineage["subjects"]:
        g, s = x["greedy"], x["sampled"]
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(g["action_mix"].items(), key=lambda kv: -kv[1])[:3])
        out.append(f"| {x['subject']} | {_f(g['success'])} | {_f(g['utility'])} | {_f(g['steps_to_cap_rate'])} | "
                   f"{_f(g['idempotent_repeat_rate'])} | {_f(g['short_cycle_rate'])} | {_f(s['success'])} | "
                   f"{_f(s['utility'])} | {_f(s['steps_to_cap_rate'])} | {_f(s['idempotent_repeat_rate'])} | "
                   f"{_f(s['short_cycle_rate'])} | {top} |")
    out += ["", "### KL (masked, as training); mean [argmax disagreement]", "",
            "| ckpt | boot-states KL(boot‖ck) | boot-states KL(prev‖ck) | own-states KL(boot‖ck) | own-states KL(prev‖ck) |",
            "|---|---|---|---|---|"]
    def kl(v):
        return "—" if not v or not v.get("states") else f"{v['mean']:.4f} [{v['argmax_disagreement']:.2f}]"
    for x in lineage["subjects"]:
        k = x["kl"]
        out.append(f"| {x['subject']} | {kl(k['bootstrap_states']['boot_to_ckpt'])} | {kl(k['bootstrap_states']['prev_to_ckpt'])} | "
                   f"{kl(k['own_greedy_states']['boot_to_ckpt'])} | {kl(k['own_greedy_states']['prev_to_ckpt'])} |")
    out += ["", "### Critic on own sampled rollouts (advantage: raw / batch-standardized)", "",
            "| ckpt | MSE | bias (V−G) | loop n | loop adv | prod n | prod adv | other adv | loop−prod (norm) |",
            "|---|---|---|---|---|---|---|---|---|"]
    def adv(c):
        return "—" if not c["n"] else f"{c['mean_raw_advantage']:+.3f} / {c['mean_normalized_advantage']:+.3f}"
    for x in lineage["subjects"]:
        c = x["critic"]
        b = c["by_class"]
        out.append(f"| {x['subject']} | {_f(c['critic_mse'], 4)} | {_f(c['critic_bias'], 4)} | {b['loop']['n']} | {adv(b['loop'])} | "
                   f"{b['productive']['n']} | {adv(b['productive'])} | {adv(b['other'])} | {_f(c['loop_minus_productive_normalized'])} |")
    out += ["", "### Gradient norms of weighted terms (mean over fixed batches): shared / actor head / critic head; shared cosines", "",
            "| ckpt | actor | critic | entropy | KL | total | cos(a,c) | cos(a,e) | cos(a,kl) | cos(c,kl) | clip frac |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in lineage["subjects"]:
        m = x["gradients"]["mean"]
        if not m:
            continue
        n = m["norms"]
        def trio(t):
            return "—" if t not in n else f"{_g(n[t]['shared'])} / {_g(n[t]['actor_head'])} / {_g(n[t]['critic_head'])}"
        c = m["shared_cosine"]
        out.append(f"| {x['subject']} | {trio('actor')} | {trio('critic')} | {trio('entropy')} | {trio('kl')} | {trio('total')} | "
                   f"{_f(c.get('actor|critic'))} | {_f(c.get('actor|entropy'))} | {_f(c.get('actor|kl'))} | "
                   f"{_f(c.get('critic|kl'))} | {_f(m['would_clip_fraction'], 2)} |")
    return "\n".join(out)


def _a(attempt):
    return "none audited" if attempt is None else str(attempt)


def summary_markdown(lineages: list[dict]) -> str:
    out = ["## Summary (descriptive; no causal claim)", ""]
    for lineage in lineages:
        loc = {k: (_a(v) if k.endswith("_attempt") else v) for k, v in locate(lineage).items()}
        fav = loc["pre_collapse_loop_advantage_exceeds_productive"]
        out.append(f"- **X1-r{lineage['lineage']}**: greedy success drop (≥.20 below bootstrap) at attempt "
                   f"{loc['greedy_success_drop_attempt']}; sampled drop at {loc['sampled_success_drop_attempt']}; "
                   f"greedy/sampled divergence (≥.10) first at {loc['greedy_sampled_divergence_attempt']}. "
                   f"KL(boot‖ck) on bootstrap states doubles vs the first audited attempt at {loc['kl_boot_doubles_attempt']}, "
                   f"largest single-tranche increase at {loc['kl_boot_largest_increase_attempt']}. "
                   f"Before the greedy drop, the standardized advantage of loop actions exceeds that of productive actions in "
                   f"{fav if fav is not None else '—'} of {len(loc['pre_collapse_loop_minus_productive_normalized_advantage'])} audited checkpoints.")
    out += ["", "Thresholds above are reading aids chosen for this table, not registered tests."]
    return "\n".join(out)


def write_report(output: Path) -> dict:
    """partd.json / partd.md from lineage-r{k}.json; a lineage without a final file
    falls back to its per-subject partial file (e.g. after a CPU-cap stop), marked partial."""
    lineages = []
    for lineage in LINEAGES:
        final, partial_file = output / f"lineage-r{lineage}.json", output / f"lineage-r{lineage}.partial.json"
        if final.exists():
            lineages.append(json.loads(final.read_text()))
        elif partial_file.exists():
            lineages.append(json.loads(partial_file.read_text()))
    lineages = [x for x in lineages if x["subjects"]]
    if not lineages:
        raise ValueError("No lineage files to report")
    manifest = output / "manifest.json"
    options = json.loads(manifest.read_text())["options"] if manifest.exists() else {}
    samples = next((x["probe"]["samples_per_world"] for x in lineages if "probe" in x), options.get("samples"))
    worlds = next((len(x["probe"]["worlds"]) for x in lineages if "probe" in x), None)
    report = {"protocol": "research/campaigns/extended-03/protocol-P2a.md#part-d", "descriptive_only": True,
              "partial_lineages": [x["lineage"] for x in lineages if x.get("partial")],
              "locations": {f"x1-r{x['lineage']}": locate(x) for x in lineages},
              "lineages": lineages}
    (output / "partd.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    text = ["# P2a Part D: collapse audit (X1 lineages)", "",
            f"Probe: {worlds or 'the'} IID-group worlds (iid_f0/iid_f2 interleaved, seeds 1,990,000,000+ / 1,990,100,000+), "
            f"greedy + {samples} fixed-seed samples per world. Definitions: module docstring of "
            "research/tools/campaign03_p2a_collapse_audit.py."
            + (f" **Partial lineages (stopped early): {report['partial_lineages']}.**" if report["partial_lineages"] else ""),
            "", summary_markdown(lineages), ""]
    text += [lineage_markdown(x) + "\n" for x in lineages]
    (output / "partd.md").write_text("\n".join(text))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_lineage(args):
    lineage, options, output = args
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign03_depworld import depworld_executor
    import torch
    torch.set_num_threads(1)
    log_path = output / f"lineage-r{lineage}.log"
    with BoundedSolver() as solver, log_path.open("a") as stream:
        def log(message):
            stream.write(message + "\n")
            stream.flush()
        worlds = probe_worlds()
        if options["worlds"]:
            worlds = worlds[:options["worlds"]]
        result = audit_lineage(lineage, attempts=options["attempts"], worlds=worlds, samples=options["samples"],
                               grad_batches=options["grad_batches"], results=options["results"],
                               make_executor=lambda: partial(depworld_executor, execute_call=solver.execute), log=log,
                               partial_path=output / f"lineage-r{lineage}.partial.json")
        result["solver_accounting"] = {"startup_wall_seconds": solver.startup_wall_seconds,
                                       "startup_child_cpu_seconds": solver.startup_child_cpu_seconds,
                                       "call_wall_seconds": solver.call_wall_seconds, "restarts": solver.restarts}
    result["options"] = {k: v for k, v in options.items()}
    path = output / f"lineage-r{lineage}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True))
    temporary.replace(path)
    return lineage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="Rebuild partd.json/partd.md from lineage files in this directory")
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--lineages", default="0,1,2")
    parser.add_argument("--attempts", default=",".join(map(str, ATTEMPTS)))
    parser.add_argument("--worlds", type=int, default=0, help="Smoke only: first N interleaved probe worlds (0 = all 64)")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--grad-batches", type=int, default=1, help="Fixed 8-episode batches for item 4 (protocol: one)")
    parser.add_argument("--processes", type=int, default=1, help="Lineages in parallel (single-thread each; <= 3)")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only (inherited by spawned lineage workers)
    if args.report:
        write_report(args.report)
        return
    if args.output is None:
        parser.error("--output required")
    lineages = [int(x) for x in args.lineages.split(",")]
    attempts = tuple(int(x) for x in args.attempts.split(","))
    if set(lineages) - set(LINEAGES) or not attempts or min(attempts) < 1 or args.samples < 1 or not 1 <= args.processes <= 3:
        parser.error("Invalid lineage/attempt/sample/process selection")
    if args.output.exists() and any(args.output.glob("lineage-r*.json")):
        raise FileExistsError("Refusing to overwrite existing lineage results")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {name: file_sha256(ROOT / name) for name in (
        "research/tools/campaign03_p2a_collapse_audit.py", "research/tools/campaign03_p1_configs.py",
        "src/tensegra/campaign02_training.py", "src/tensegra/campaign02_policy.py",
        "src/tensegra/campaign02_population.py", "src/tensegra/campaign03_depworld.py",
        "src/tensegra/campaign02_protocol.py", "src/tensegra/campaign02_world.py")}
    options = {"attempts": attempts, "worlds": args.worlds, "samples": args.samples,
               "grad_batches": args.grad_batches, "results": args.results}
    (args.output / "manifest.json").write_text(json.dumps({
        "argv": sys.argv, "sources": sources, "options": options, "lineages": lineages,
        "probe_worlds": [[c, s] for c, s, _ in probe_worlds()][:args.worlds or None],
        "started_unix": time.time()}, indent=1, sort_keys=True))
    jobs = [(lineage, options, args.output) for lineage in lineages]
    if args.processes > 1:
        # Non-daemonic spawn workers: each owns a persistent solver child process.
        import multiprocessing
        context, pending, running = multiprocessing.get_context("spawn"), list(jobs), []
        while pending or running:
            while pending and len(running) < args.processes:
                job = pending.pop(0)
                process = context.Process(target=_run_lineage, args=(job,))
                process.start()
                running.append((job[0], process))
            lineage, process = running.pop(0)
            process.join()
            if process.exitcode != 0:
                for _, other in running:
                    other.terminate()
                raise RuntimeError(f"lineage r{lineage} failed with exit code {process.exitcode}")
            print(f"lineage r{lineage} done", flush=True)
    else:
        for job in jobs:
            _run_lineage(job)
            print(f"lineage r{job[0]} done", flush=True)
    write_report(args.output)


if __name__ == "__main__":
    main()
