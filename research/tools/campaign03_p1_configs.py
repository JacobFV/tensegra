"""Generate frozen P1 configs (bootstrap, RL, sealed evaluation) for extended-03.

Pairing: within lineage index r, arms share init seeds and training streams
(paired ablation); across lineage indices all streams/inits are disjoint.
X2 uses a different teacher, so its bootstrap stream is paired too but its data
differ by construction. Sealed seeds start at 110,000,000 (fresh, never used).
"""
import argparse, json
from pathlib import Path

TRAIN_KINDS = ["edge_closed", "capacity_reduced", "slot_closed"]
S3 = {"categories": 3, "choices": 3, "locations": 7, "slots": 6}
S4 = {"categories": 4, "choices": 3, "locations": 8, "slots": 7}
COMMON = {"compute_price": 0.0001, "event_trigger": "progress"}
TRAIN_MIX = [{**S3, **COMMON, "p_event": 0.5, "foreign_records": f, "event_kinds": TRAIN_KINDS} for f in (0, 2)]
ARMS = {"x1": ("dep_reuse", "d1"), "x2": ("dep_recompute", "d1"),
        "x3": ("dep_reuse", "d1-noapp"), "x4": ("dep_reuse", "d1-noattempt")}
HYPER = [{"learning_rate": 3e-5, "entropy_weight": 0.003, "kl_weight": 0.3}] * 6
TRAIN = {"width": 1024, "family": "lightweight", "device": "cuda", "batch_size": 8, "learning_rate": 3e-4,
         "rollout_mode": "batched", "max_steps": 96, "evaluation_batch": 32, "policy_loss_reduction": "decision_mean",
         "advantage_normalization": True}

SEALED = [
    ("iid_f0", {**S3, **COMMON, "p_event": 0.5, "foreign_records": 0, "event_kinds": TRAIN_KINDS}),
    ("iid_f2", {**S3, **COMMON, "p_event": 0.5, "foreign_records": 2, "event_kinds": TRAIN_KINDS}),
    ("noevent_f2", {**S3, **COMMON, "p_event": 0.0, "foreign_records": 2}),
    ("events_train_kinds_p1", {**S3, **COMMON, "p_event": 1.0, "foreign_records": 2, "event_kinds": TRAIN_KINDS}),
    ("heldout_deadline_moved", {**S3, **COMMON, "p_event": 1.0, "foreign_records": 2, "event_kinds": ["deadline_moved"]}),
    ("foreign4", {**S3, **COMMON, "p_event": 0.5, "foreign_records": 4, "event_kinds": TRAIN_KINDS}),
    ("larger_s4", {**S4, **COMMON, "p_event": 0.5, "foreign_records": 2, "event_kinds": TRAIN_KINDS}),
    ("work_price_x4", {**S3, **COMMON, "p_event": 0.5, "foreign_records": 2, "event_kinds": TRAIN_KINDS, "work_price": 0.0008}),
]


def base(arm, r):
    teacher, feat = ARMS[arm]
    return {"mode": "single", "population_seed": 30000 + r, "initialization_seeds": [300000 + 100*r + i for i in range(6)],
            "optimizer_policy": "inherit", "teacher": teacher, "threads": 1, "world_family": "depworld",
            "policy": {"interface": "legacy", "feature_version": feat}, "world_mix": TRAIN_MIX, "train": dict(TRAIN),
            "development_seed_start": 3_900_000_000 + 1_000_000*r, "development_examples": 128,
            "training_seed_stride": 1_000_000, "member_hyperparameters": HYPER}


def boot(arm, r):
    c = base(arm, r)
    c.update(rounds=1, updates_per_slot=100, methods=["supervised"], training_seed_start=3_000_000_000 + 20_000_000*r,
             address_namespace=f"e03p1-boot-r{r}")
    return c


def rl(arm, r, bank):
    c = base(arm, r)
    c.update(rounds=5, updates_per_slot=60, methods=["actor_critic"]*5, training_seed_start=3_400_000_000 + 20_000_000*r,
             address_namespace=f"e03p1-rl-r{r}", initial_checkpoints=[bank]*6)
    return c


def sealed(checkpoints, references, examples=256, start=110_000_000):
    return {"checkpoints": checkpoints, "evaluation_batch": 32, "references": references, "reference_compute_tariff": 1.0,
            "world_family": "depworld", "address_namespace": "e03p1-sealed",
            "conditions": [{"name": n, "world_family": "depworld", "seed_start": start + 100_000*i, "examples": examples, "world": w}
                           for i, (n, w) in enumerate(SEALED)]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["boot", "rl", "sealed"])
    p.add_argument("--banks", help="JSON {arm-r: {path, sha256}} for rl")
    p.add_argument("--checkpoints", help="JSON list for sealed")
    p.add_argument("--references", default="")
    p.add_argument("--output", default="configs/campaign03")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "boot":
        for arm in ARMS:
            for r in range(3):
                (out/f"p1-boot-{arm}-r{r}.json").write_text(json.dumps(boot(arm, r), indent=2))
    elif a.stage == "rl":
        banks = json.load(open(a.banks))
        for key, bank in banks.items():
            arm, r = key.split("-r")
            (out/f"p1-rl-{arm}-r{r}.json").write_text(json.dumps(rl(arm, int(r), bank), indent=2))
    else:
        refs = [x for x in a.references.split(",") if x]
        (out/"p1-sealed.json").write_text(json.dumps(sealed(json.load(open(a.checkpoints)), refs), indent=2))
