"""extended-03 P2a configs: Part C arms, the registered deployment rule, and the screening evaluation.

Registered design: research/campaigns/extended-03/protocol-P2a.md.

- ``arms``: C0 = the P1 X1-r2 RL config byte for byte (``p2a-c0-x1-r2.json``);
  C1 = C0 + the fixed bootstrap anchor (the X1-r2 bank checkpoint from
  ``p1-banks.json``, ``anchor_kl_weight`` 0.3; the tranche KL is kept).
- ``deploy``: the registered deployment rule for one finished run: the latest
  tranche checkpoint with development success >= bootstrap - 0.02 and development
  utility >= bootstrap - 0.02, else roll back to the bootstrap. Writes JSON with
  every candidate and the reasoning.
- ``screening``: fresh screening seeds 120,000,000 + 100,000*i, 256 worlds, the P1
  conditions iid_f0, iid_f2, events_train_kinds_p1, foreign4; greedy (as in P1) and
  sampled (one fixed-seed sample per world, conditions named ``<name>-sampled``);
  the P2a policies (checkpoints passed in) and the dep_reuse reference.

    python research/tools/campaign03_p2a_configs.py arms --output configs/campaign03
    python research/tools/campaign03_p2a_configs.py deploy --run <results>/p2a-c1-x1-r2 \\
        --bootstrap-run <results>/p1-boot-x1-r2 --output <results>/p2a-c1-x1-r2/deployment.json
    python research/tools/campaign03_p2a_configs.py screening --policies <policies.json> --output configs/campaign03
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_p1():
    spec = importlib.util.spec_from_file_location("campaign03_p1_configs", HERE / "campaign03_p1_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = _load_p1()
LINEAGE = "x1-r2"
ANCHOR_KL_WEIGHT = 0.3
DEPLOY_MARGIN = 0.02
SCREENING_START, SCREENING_STRIDE, SCREENING_EXAMPLES = 120_000_000, 100_000, 256
SCREENING_CONDITIONS = ("iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4")
SAMPLING_SEED = 20260926
SCREENING_REFERENCES = ("dep_reuse",)
POLICY_ROLES = ("bootstrap", "c0_final", "c0_deployed", "c1_final", "c1_deployed", "p1_rl")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Part C arms
# ---------------------------------------------------------------------------

def c0_config(p1_config_text: str, banks: dict) -> dict:
    """C0 is the P1 X1-r2 RL config, unchanged. Check it is what P1's generator produces."""
    c0 = json.loads(p1_config_text)
    regenerated = P1.rl("x1", 2, banks[LINEAGE])
    if c0 != regenerated:
        raise ValueError("p1-rl-x1-r2.json differs from the P1 generator output for the x1-r2 bank")
    return c0


def c1_config(c0: dict, banks: dict, weight: float = ANCHOR_KL_WEIGHT) -> dict:
    """C0 + the fixed bootstrap anchor. Only train.anchor_* differ; every seed, stream,
    bank, hyperparameter (including the tranche kl_weight) and namespace is C0's."""
    anchor = dict(banks[LINEAGE])
    if set(anchor) != {"path", "sha256"}:
        raise ValueError("Bank entry must be {path, sha256}")
    if any(c != anchor for c in c0["initial_checkpoints"]):
        raise ValueError("The anchor must be the run's own bootstrap bank checkpoint")
    c1 = json.loads(json.dumps(c0))
    c1["train"] = {**c1["train"], "anchor_kl_weight": weight, "anchor_checkpoint": anchor}
    return c1


def arm_difference(c0: dict, c1: dict) -> dict:
    """Every top-level/train key that differs between the arms (should be train anchor fields only)."""
    diff = {k: (c0.get(k), c1.get(k)) for k in set(c0) | set(c1) if k != "train" and c0.get(k) != c1.get(k)}
    diff.update({f"train.{k}": (c0["train"].get(k), c1["train"].get(k))
                 for k in set(c0["train"]) | set(c1["train"]) if c0["train"].get(k) != c1["train"].get(k)})
    return diff


def write_arms(configs: Path, output: Path):
    banks = json.loads((configs / "p1-banks.json").read_text())
    text = (configs / f"p1-rl-{LINEAGE}.json").read_text()
    c0 = c0_config(text, banks)
    c1 = c1_config(c0, banks)
    diff = arm_difference(c0, c1)
    if set(diff) != {"train.anchor_kl_weight", "train.anchor_checkpoint"}:
        raise ValueError(f"Arms differ beyond the anchor: {sorted(diff)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"p2a-c0-{LINEAGE}.json").write_text(text)  # byte-identical replay of P1's config
    (output / f"p2a-c1-{LINEAGE}.json").write_text(json.dumps(c1, indent=2))
    return c0, c1


# ---------------------------------------------------------------------------
# Registered deployment rule
# ---------------------------------------------------------------------------

def bootstrap_development(boot_run: Path) -> dict:
    """The bootstrap's development metrics: its population finalist row (P1's 128
    development worlds for this lineage; single mode: the last tranche)."""
    state = json.loads((boot_run / "state.json").read_text())
    if state.get("status") != "completed":
        raise ValueError(f"{boot_run} is not completed")
    f = state["finalist"]
    return {"source": str(boot_run / "state.json"), "checkpoint": str(boot_run / f["checkpoint"]),
            "sha256": f["checkpoint_sha256"], "success": f["success"], "utility": f["utility"],
            "examples": f.get("examples"), "seeds_hash": f.get("seeds_hash")}


def deployment(run: Path, boot: dict, margin: float = DEPLOY_MARGIN) -> dict:
    """Latest tranche checkpoint with dev success >= boot - margin and dev utility >= boot - margin;
    else roll back to the bootstrap. Development metrics only (never screening or sealed data)."""
    state = json.loads((run / "state.json").read_text())
    if state.get("status") != "completed":
        raise ValueError(f"{run} is not completed")
    rows = sorted(state["allocations"], key=lambda r: r["cumulative_slot_updates"])
    if not rows:
        raise ValueError("No tranche checkpoints")
    if any(r.get("seeds_hash") != rows[0].get("seeds_hash") for r in rows):
        raise ValueError("Tranche development seeds differ within the run")
    if boot.get("seeds_hash") and boot["seeds_hash"] != rows[0].get("seeds_hash"):
        raise ValueError("Bootstrap and run development worlds differ")
    s_min, u_min = boot["success"] - margin, boot["utility"] - margin
    candidates = []
    for r in rows:
        ok_s, ok_u = r["success"] >= s_min, r["utility"] >= u_min
        candidates.append({"label": Path(r["checkpoint"]).stem, "checkpoint": str(run / r["checkpoint"]),
                           "sha256": r["checkpoint_sha256"], "updates": r["cumulative_slot_updates"],
                           "round": r["round"], "slot": r["slot"], "success": r["success"], "utility": r["utility"],
                           "success_ok": ok_s, "utility_ok": ok_u, "qualifies": ok_s and ok_u})
    qualifying = [c for c in candidates if c["qualifies"]]
    final = candidates[-1]
    if qualifying:
        chosen = {**qualifying[-1], "kind": "tranche"}
        reason = (f"latest qualifying tranche checkpoint (update {chosen['updates']}); "
                  f"{len(qualifying)}/{len(candidates)} tranche checkpoints qualify")
    else:
        chosen = {"label": "bootstrap", "kind": "rollback", "checkpoint": boot["checkpoint"], "sha256": boot["sha256"],
                  "updates": 0, "success": boot["success"], "utility": boot["utility"]}
        reason = "no tranche checkpoint qualifies: roll back to the bootstrap"
    return {"rule": "deploy the latest tranche checkpoint whose development success >= bootstrap - 0.02 and "
                    "development utility >= bootstrap - 0.02; else roll back to the bootstrap "
                    "(protocol-P2a.md; never applied to screening or sealed data)",
            "run": str(run), "margin": margin, "bootstrap_development": boot,
            "thresholds": {"success": s_min, "utility": u_min},
            "development_seeds_hash": rows[0].get("seeds_hash"),
            "final": final, "deployed": chosen, "deployed_is_final": chosen.get("sha256") == final["sha256"],
            "reason": reason, "candidates": candidates,
            "note": "Bootstrap development metrics come from the bootstrap run, whose address namespace "
                    "(record-handle spelling) differs from the RL run's on the same development worlds."}


# ---------------------------------------------------------------------------
# Screening evaluation
# ---------------------------------------------------------------------------

def screening(policies, references=SCREENING_REFERENCES, examples=SCREENING_EXAMPLES, start=SCREENING_START,
              sampled=True, sampling_seed=SAMPLING_SEED):
    """policies: [{name, path, sha256, role}] (roles: POLICY_ROLES). Greedy conditions keep the
    P1 names; sampled ones are '<name>-sampled' on the same worlds, learned policies only."""
    roles = [p.get("role") for p in policies]
    if set(roles) - set(POLICY_ROLES) or len(set(roles)) != len(roles):
        raise ValueError(f"Policies need distinct roles from {POLICY_ROLES}")
    worlds = dict(P1.SEALED)
    conditions = []
    for name in SCREENING_CONDITIONS:
        i = SCREENING_CONDITIONS.index(name)
        base = {"world_family": "depworld", "seed_start": start + SCREENING_STRIDE * i, "examples": examples,
                "world": worlds[name]}
        conditions.append({"name": name, **base})
        if sampled:
            conditions.append({"name": f"{name}-sampled", **base, "policy_mode": "sampled",
                               "sampling_seed": sampling_seed, "references": []})
    return {"checkpoints": [dict(p) for p in policies], "evaluation_batch": 32, "references": list(references),
            "reference_compute_tariff": 1.0, "world_family": "depworld", "address_namespace": "e03p2a-screening",
            "conditions": conditions}


def verified_policies(path: Path):
    policies = json.loads(path.read_text())
    for p in policies:
        if _sha256(p["path"]) != p["sha256"]:
            raise ValueError(f"{p['name']}: checkpoint hash mismatch")
    return policies


def policies_from_runs(boot_run: Path, c0_run: Path, c1_run: Path, p1_rl_run: Path, d0: dict, d1: dict):
    """Policy bindings for the screening: bootstrap, C0/C1 final and deployed, P1's X1-rl-r2 endpoint."""
    boot = bootstrap_development(boot_run)
    p1 = json.loads((p1_rl_run / "state.json").read_text())["finalist"]
    out = [{"name": "p2a-bootstrap-x1-r2", "role": "bootstrap", "path": boot["checkpoint"], "sha256": boot["sha256"]}]
    for arm, d in (("c0", d0), ("c1", d1)):
        out.append({"name": f"p2a-{arm}-final", "role": f"{arm}_final", "path": d["final"]["checkpoint"],
                    "sha256": d["final"]["sha256"]})
        out.append({"name": f"p2a-{arm}-deployed", "role": f"{arm}_deployed", "path": d["deployed"]["checkpoint"],
                    "sha256": d["deployed"]["sha256"], "deployed_label": d["deployed"]["label"]})
    out.append({"name": "p1-rl-x1-r2", "role": "p1_rl", "path": str(p1_rl_run / p1["checkpoint"]),
                "sha256": p1["checkpoint_sha256"]})
    for p in out:
        p.update(arm="x1", lineage=2, feature_version="d1")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["arms", "deploy", "policies", "screening"])
    p.add_argument("--configs", type=Path, default=Path("configs/campaign03"), help="P1 config directory")
    p.add_argument("--output", type=Path, default=Path("configs/campaign03"))
    p.add_argument("--run", type=Path, help="deploy: finished P2a run directory")
    p.add_argument("--bootstrap-run", type=Path, help="deploy/policies: the p1-boot-x1-r2 run directory")
    p.add_argument("--c0-run", type=Path)
    p.add_argument("--c1-run", type=Path)
    p.add_argument("--p1-rl-run", type=Path)
    p.add_argument("--deployment-c0", type=Path)
    p.add_argument("--deployment-c1", type=Path)
    p.add_argument("--policies", type=Path, help="screening: JSON list of {name, path, sha256, role}")
    p.add_argument("--examples", type=int, default=SCREENING_EXAMPLES)
    p.add_argument("--seed-start", type=int, default=SCREENING_START)
    p.add_argument("--no-sampled", action="store_true", help="drop the sampled mode (budget fallback, registered)")
    a = p.parse_args()
    if a.stage == "arms":
        write_arms(a.configs, a.output)
        print(json.dumps({"c0": str(a.output / f"p2a-c0-{LINEAGE}.json"), "c1": str(a.output / f"p2a-c1-{LINEAGE}.json")}))
    elif a.stage == "deploy":
        if not (a.run and a.bootstrap_run):
            raise SystemExit("deploy needs --run and --bootstrap-run")
        result = deployment(a.run, bootstrap_development(a.bootstrap_run))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({"deployed": result["deployed"]["label"], "reason": result["reason"]}))
    elif a.stage == "policies":
        d0, d1 = (json.loads(x.read_text()) for x in (a.deployment_c0, a.deployment_c1))
        out = policies_from_runs(a.bootstrap_run, a.c0_run, a.c1_run, a.p1_rl_run, d0, d1)
        for x in out:
            if _sha256(x["path"]) != x["sha256"]:
                raise ValueError(f"{x['name']}: checkpoint hash mismatch")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2))
    else:
        policies = verified_policies(a.policies)
        protocol = a.examples == SCREENING_EXAMPLES and a.seed_start == SCREENING_START
        cfg = screening(policies, examples=a.examples, start=a.seed_start, sampled=not a.no_sampled)
        suffix = "" if not a.no_sampled else "-greedy-only"
        name = (f"p2a-screening{suffix}.json" if protocol
                else f"p2a-screening-nonprotocol-e{a.examples}-s{a.seed_start}{suffix}.json")
        if not protocol:
            cfg["address_namespace"] = "e03p2a-nonprotocol"
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / name).write_text(json.dumps(cfg, indent=2))
        print(a.output / name)


if __name__ == "__main__":
    main()
