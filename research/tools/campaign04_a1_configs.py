"""Generate the extended-04 A1 deployment-matrix evaluation configs (design §2 A1, v2 revisions).

Evaluation only (exploratory): every A0 frozen learned policy under every deployment
mode on fresh worlds, plus the supplied references.

- Policies (A0, identified by sha256; never retrained):
  X1 bootstraps r0-r2 (p1-boot-x1-r*), P1 RL finals X1 r0-r2 (p1-rl-x1-r*),
  P2a C1 final (anchored, r2). Paths/hashes come from configs/campaign03/
  p1-sealed-checkpoints.json and p2a-policies.json (remote results under
  /home/brand/tensegra-campaign03/results).
- References (supplied schedules): dep_reuse, dep_recompute, dep_greedy.
- Modes: greedy, sampled (T=1; one fixed-seed trajectory per world), r_mask, r_sample.
- Conditions (P1 definitions, campaign03_p1_configs.SEALED): iid_f0, iid_f2,
  events_train_kinds_p1, foreign4; 256 fresh worlds each.
- Fresh seeds: 130,000,000 + 100,000*i (i = condition index), disjoint from every
  extended-03 range (asserted below). Every mode of a condition uses the same worlds
  (world-paired), and sampled/r_sample share the registered sampling seed.

All conditions set progress_diagnostic (the v1 diagnostic runs alongside every
learned and reference episode; greedy/sampled choices are unchanged by it), the
exact solver cache (removes wall-deadline nondeterminism; result-identical on
deterministic calls) and trace_observations = false (size control; exact
action/feedback histories are kept).

    python research/tools/campaign04_a1_configs.py [--split] [--verify-hashes] [--output configs/campaign04]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED_BASE = 130_000_000
SEED_STRIDE = 100_000
EXAMPLES = 256
SAMPLING_SEED = 20260927
NAMESPACE = "e04a1"
CONDITIONS = ("iid_f0", "iid_f2", "events_train_kinds_p1", "foreign4")
MODES = ("greedy", "sampled", "r_mask", "r_sample")
REFERENCES = ("dep_reuse", "dep_recompute", "dep_greedy")
# Every world-seed range used by extended-03 (half-open intervals), to assert freshness.
EXT03_RANGES = {
    "training (boot 3.0e9+, rl 3.4e9+)": (3_000_000_000, 3_900_000_000),
    "development 3.9e9+": (3_900_000_000, 4_000_000_000),
    "P1 sealed 110M": (110_000_000, 111_000_000),
    "P2a screening 120M": (120_000_000, 121_000_000),
    "P2a Part D probe 1.99e9": (1_990_000_000, 2_000_000_000),
    "calibration/smoke 2.0-2.1e9": (2_000_000_000, 2_100_000_000),
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_ranges(examples=EXAMPLES, base=SEED_BASE):
    return {c: (base + SEED_STRIDE * i, base + SEED_STRIDE * i + examples) for i, c in enumerate(CONDITIONS)}


def check_fresh(examples=EXAMPLES, base=SEED_BASE):
    ranges = seed_ranges(examples, base)
    for c, (lo, hi) in ranges.items():
        if examples > SEED_STRIDE:
            raise ValueError("conditions would overlap")
        for label, (a, b) in EXT03_RANGES.items():
            if lo < b and a < hi:
                raise ValueError(f"A1 seeds for {c} overlap extended-03 {label}")
    return ranges


def policies(root=REPO):
    sealed = json.loads((root / "configs/campaign03/p1-sealed-checkpoints.json").read_text())
    p2a = json.loads((root / "configs/campaign03/p2a-policies.json").read_text())
    out = []
    for endpoint, family in (("boot", "bootstrap"), ("rl", "p1_rl_final")):
        for r in range(3):
            b = next(x for x in sealed if x["name"] == f"p1-{endpoint}-x1-r{r}")
            out.append({"name": b["name"], "path": b["path"], "sha256": b["sha256"], "family": family,
                        "lineage": r, "arm": "x1", "feature_version": b["feature_version"]})
    c1 = next(x for x in p2a if x["role"] == "c1_final")
    out.append({"name": "p2a-c1-final-x1-r2", "path": c1["path"], "sha256": c1["sha256"], "family": "p2a_c1_final",
                "lineage": 2, "arm": "x1", "feature_version": c1["feature_version"]})
    return out


def condition_entries(examples=EXAMPLES, base=SEED_BASE, modes=MODES, references=REFERENCES):
    sealed = dict(_load("campaign03_p1_configs").SEALED)
    ranges = check_fresh(examples, base)
    out = []
    for c in CONDITIONS:
        for mode in modes:
            entry = {"name": f"{c}-{mode}", "base_condition": c, "mode": mode, "world_family": "depworld",
                     "seed_start": ranges[c][0], "examples": examples, "world": sealed[c], "policy_mode": mode,
                     "progress_diagnostic": True, "references": list(references) if mode == "greedy" else []}
            if mode in ("sampled", "r_sample", "masked_sampled"):
                entry["sampling_seed"] = SAMPLING_SEED
            out.append(entry)
    return out


def config(checkpoints, examples=EXAMPLES, base=SEED_BASE, modes=MODES, references=REFERENCES):
    names = [c["name"] for c in checkpoints]
    if len(set(names)) != len(names) or len({c["sha256"] for c in checkpoints}) != len(names):
        raise ValueError("duplicate policy")
    return {"checkpoints": checkpoints, "evaluation_batch": 32, "references": [], "reference_compute_tariff": 1.0,
            "world_family": "depworld", "address_namespace": NAMESPACE, "progress_diagnostic": True,
            "solver_cache": True, "trace_observations": False, "diagnostic_compute_units": 0.0,
            "campaign": "extended-04", "experiment": "A1 deployment matrix (exploratory)",
            "conditions": condition_entries(examples, base, modes, references)}


def split_configs(checkpoints, examples=EXAMPLES, base=SEED_BASE):
    """One config per learned policy (no references) + one references-only config (greedy
    conditions only): identical worlds, namespaces and seeds, runnable in parallel."""
    out = {}
    for c in checkpoints:
        out[f"a1-{c['name']}"] = config([c], examples, base, references=())
    refs = config([], examples, base, modes=("greedy",))
    out["a1-references"] = refs
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "configs/campaign04")
    p.add_argument("--split", action="store_true", help="also write one config per policy + a references config")
    p.add_argument("--verify-hashes", action="store_true", help="check checkpoint files (remote host)")
    p.add_argument("--examples", type=int, default=EXAMPLES)
    p.add_argument("--seed-start", type=int, default=SEED_BASE)
    a = p.parse_args()
    cps = policies()
    if a.verify_hashes:
        for c in cps:
            if _sha256(c["path"]) != c["sha256"]:
                raise SystemExit(f"hash mismatch: {c['name']}")
    protocol = a.examples == EXAMPLES and a.seed_start == SEED_BASE
    suffix = "" if protocol else f"-nonprotocol-e{a.examples}-s{a.seed_start}"
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = config(cps, a.examples, a.seed_start)
    if not protocol:
        cfg["address_namespace"] = "e04a1-nonprotocol"
    (a.output / f"a1-deployment{suffix}.json").write_text(json.dumps(cfg, indent=2))
    written = [f"a1-deployment{suffix}.json"]
    if a.split:
        for name, c in split_configs(cps, a.examples, a.seed_start).items():
            if not protocol:
                c["address_namespace"] = "e04a1-nonprotocol"
            (a.output / f"{name}{suffix}.json").write_text(json.dumps(c, indent=2))
            written.append(f"{name}{suffix}.json")
    print(json.dumps({"written": written, "seed_ranges": seed_ranges(a.examples, a.seed_start)}))


if __name__ == "__main__":
    main()
