"""Protocol-A2 screening configs (registered: seeds 140,000,000 + 100,000*i, 256/condition,
conditions iid_f0/iid_f2/events_train_kinds_p1/foreign4, modes greedy/sampled/r_mask).

Condition blocks are copied from the A1 config template (identical world definitions and
evaluator flags) with the seed base moved from 130M to 140M and r_sample dropped.
Usage: campaign04_a2_screen_configs.py --policies policies.json --out configs/campaign04
policies.json: [{"name","path","sha256","family","lineage","arm","feature_version"}, ...]
"""
import argparse, copy, json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[2] / "configs/campaign04/a1-p1-rl-x1-r2.json"
MODES = ("greedy", "sampled", "r_mask")


def conditions(seed_base=140_000_000):
    t = json.loads(TEMPLATE.read_text())
    out = []
    for c in t["conditions"]:
        if c["mode"] not in MODES:
            continue
        c = copy.deepcopy(c)
        assert 130_000_000 <= c["seed_start"] < 131_000_000
        c["seed_start"] = c["seed_start"] - 130_000_000 + seed_base
        out.append(c)
    assert len(out) == 12, len(out)
    return t, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policies", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=140_000_000); ap.add_argument("--prefix", default="a2s"); ap.add_argument("--namespace", default="e04a2")
    a = ap.parse_args()
    t, conds = conditions(a.seed_base)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for p in json.loads(Path(a.policies).read_text()):
        cfg = {**{k: v for k, v in t.items() if k not in ("checkpoints", "conditions", "experiment", "address_namespace")},
               "checkpoints": [p] if p.get("path") else [], "references": [],
               "address_namespace": a.namespace, "experiment": "A2 screening (protocol-A2, exploratory)", "conditions": conds}
        if not p.get("path"):
            cfg["conditions"] = [{**c, "references": p["references"]} for c in conds if c["mode"] == "greedy"]
        (out / f"{a.prefix}-{p['name']}.json").write_text(json.dumps(cfg, indent=1))
        print(out / f"{a.prefix}-{p['name']}.json")


if __name__ == "__main__":
    main()
