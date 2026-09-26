"""extended-04 Track C registered constants, bases, seed ranges and shared loaders.

Used by campaign04_c_labels.py / campaign04_c_train.py / campaign04_c_eval.py /
campaign04_c_analysis.py. See research/campaigns/extended-04/trackc.md.

    python research/tools/campaign04_c_configs.py      # print bases, ranges and the freshness check
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A1 = _load("campaign04_a1_configs")
SEALED = dict(_load("campaign03_p1_configs").SEALED)

# --- seeds --------------------------------------------------------------------------
LABEL_SEED_BASE = 150_000_000          # label worlds (the same worlds for every base: paired)
LABEL_SPAN = 1_000_000
EVAL_SEED_BASE = 160_000_000           # Track C evaluation worlds, + 100,000 * condition index
EVAL_STRIDE = 100_000
DEV_TEST_BASE = 2_170_000_000          # tests / smokes (tests/test_campaign04_meta.py)
DEV_TEST_SPAN = 1_000_000
DEV_LABEL_BASE = DEV_TEST_BASE + 200_000   # --dev-worlds smokes (non-protocol), inside the dev/test range
DEV_EVAL_BASE = DEV_TEST_BASE + 300_000    # + 100,000 * condition index (<= DEV_TEST_BASE + 700,000)
DEV_SPAN = 100_000
CONDITIONS = A1.CONDITIONS             # iid_f0, iid_f2, events_train_kinds_p1, foreign4
LABEL_CONDITIONS = ("iid_f0", "iid_f2")  # the actors' training mix (TRAIN_MIX); alternate by index
NAMESPACE_LABELS = "e04c-labels"
NAMESPACE_EVAL = "e04c-eval"
SAMPLING_SEED = A1.SAMPLING_SEED       # fixed sampled / r_sample arms (A1 semantics)
META_SAMPLING_SEED = 20260928          # sample-step interventions of the metacontrol arms
RANDOM_ARM_SEED = 20260929             # matched-rate random arm decisions

# Every registered world-seed range (half-open) of extended-03 and extended-04 before Track C.
OTHER_RANGES = {
    **{f"ext03 {k}": v for k, v in A1.EXT03_RANGES.items()},
    "ext04 A1 130M": (130_000_000, 130_000_000 + A1.SEED_STRIDE * len(A1.CONDITIONS)),
    "ext04 A2 screening 140M": (140_000_000, 140_000_000 + 100_000 * 4),
    "ext04 deploy/progress tests 2.15e9": (2_150_000_000, 2_151_000_000),
    "ext04 probeworld 4.1e9-4.7e9": (4_100_000_000, 4_700_000_000),
    "ext04 probeworld train 8e9": (8_000_000_000, 8_400_000_000),
}

# --- registered plan (trackc.md §8; set from the smoke before any main output) -------------
LABEL_EPISODES = 768                   # per base (24 chunks x 32); stop launching chunks at 12k core-s
CHUNK = 32
EVAL_EXAMPLES = 256                    # per condition (tier 1)
CAUSAL_EXAMPLES = 128                  # per IID condition (tier 2 causal arms, meta seed 0)
BRANCH_EVAL_WORLDS = 64                # per IID condition, appraisal_only + learned
META_SEEDS = (0, 1, 2)
EPOCHS = 30
TELEMETRY_GROUPS = ("policy", "budget", "cost", "diagnostic", "feedback", "info_events", "stage", "control")


def ranges(label_episodes=LABEL_EPISODES, eval_examples=EVAL_EXAMPLES):
    out = {"trackc labels": (LABEL_SEED_BASE, LABEL_SEED_BASE + label_episodes)}
    for i, c in enumerate(CONDITIONS):
        out[f"trackc eval {c}"] = (EVAL_SEED_BASE + EVAL_STRIDE * i, EVAL_SEED_BASE + EVAL_STRIDE * i + eval_examples)
    out["trackc dev/tests"] = (DEV_TEST_BASE, DEV_TEST_BASE + DEV_TEST_SPAN)
    return out


def check_fresh(label_episodes=LABEL_EPISODES, eval_examples=EVAL_EXAMPLES):
    if label_episodes > LABEL_SPAN or eval_examples > EVAL_STRIDE:
        raise ValueError("Track C ranges would overflow their spans")
    mine = ranges(label_episodes, eval_examples)
    for name, (lo, hi) in mine.items():
        for other, (a, b) in OTHER_RANGES.items():
            if lo < b and a < hi:
                raise ValueError(f"{name} overlaps {other}")
        for other, (a, b) in mine.items():
            if other != name and lo < b and a < hi:
                raise ValueError(f"{name} overlaps {other}")
    return mine


def bases(root=REPO):
    """X1 bootstraps r0-r2 (competent) and P1 RL finals X1 r0-r2 (loop-prone under greedy;
    r1 is the stable one, review F10), frozen, from the A1 policy list."""
    out = []
    for p in A1.policies(root):
        if p["family"] not in ("bootstrap", "p1_rl_final"):
            continue
        base_type = "competent" if p["family"] == "bootstrap" else "loop_prone"
        f10 = base_type if not (p["family"] == "p1_rl_final" and p["lineage"] == 1) else "stable"
        out.append({**p, "base_type": base_type, "f10_stratum": f10})
    return out


def base(name, root=REPO):
    for b in bases(root):
        if b["name"] == name:
            return b
    raise SystemExit(f"unknown base {name!r}; one of {[b['name'] for b in bases(root)]}")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_actor(binding, verify=True):
    """(policy, TrainConfig, info) for a frozen base; the checkpoint hash is verified."""
    import torch
    from tensegra.campaign02_population import build_policy
    from tensegra.campaign02_training import TrainConfig
    if verify and file_hash(binding["path"]) != binding["sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {binding['name']}")
    checkpoint = torch.load(binding["path"], map_location="cpu", weights_only=False)
    train_cfg = TrainConfig(**checkpoint["config"])
    policy = build_policy(checkpoint["policy_config"])
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, train_cfg, {"sha256": binding["sha256"], "updates": checkpoint.get("updates"),
                               "parameters": sum(p.numel() for p in policy.parameters()),
                               "family": policy.config.family, "width": policy.config.width}


def world(seed, condition, executor, namespace):
    from tensegra.campaign02_training import independent_address_seed
    from tensegra.campaign03_depworld import DepWorkshop, generate_depworld
    kwargs = dict(SEALED[condition])
    if "call_budgets" in kwargs:
        kwargs["call_budgets"] = tuple(kwargs["call_budgets"])
    return DepWorkshop(generate_depworld(seed, **kwargs), executor=executor,
                       address_seed=independent_address_seed(seed, namespace))


def label_condition(index):
    return LABEL_CONDITIONS[index % len(LABEL_CONDITIONS)]


def source_hashes():
    src = REPO / "src/tensegra"
    names = ("campaign04_meta", "campaign04_meta_train", "campaign04_telemetry", "campaign04_progress",
             "campaign04_branch", "campaign04_deploy", "campaign03_depworld", "campaign02_training")
    out = {n: file_hash(src / f"{n}.py") for n in names}
    for tool in ("campaign04_c_configs", "campaign04_c_labels", "campaign04_c_train", "campaign04_c_eval"):
        path = HERE / f"{tool}.py"
        if path.exists():
            out[tool] = file_hash(path)
    return out


if __name__ == "__main__":
    print(json.dumps({"ranges": check_fresh(), "bases": [{k: b[k] for k in ("name", "base_type", "f10_stratum", "sha256")}
                                                        for b in bases()]}, indent=2))
