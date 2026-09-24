"""E16 modular-composition conditions (training mixture, development profile, sealed evaluation).

Stage sequences are public goals. TRAIN contains singles and three ordered pairs;
HELD-OUT contains the three reversed/novel ordered pairs and all 3-stage
orders. Sizes: select 3x3 items (7 locations), assign 5 tasks x 3 slots.
"""
import argparse, itertools, json

BASE = {"categories": 3, "choices": 3, "locations": 7, "tasks": 5, "slots": 3, "forbid": 0.3,
        "step_limit": 64, "compute_price": 0.0001, "include_remaining_budget": True}
TRAIN = [("select",), ("route",), ("assign",), ("select", "route"), ("select", "assign"), ("assign", "route")]
TRAIN_MIX = TRAIN + TRAIN[3:]            # pairs weighted twice
HELD_PAIRS = [("route", "select"), ("assign", "select"), ("route", "assign")]
TRIPLES = list(itertools.permutations(("select", "route", "assign")))


def name(stages, prefix):
    return prefix + "_" + "-".join(s[0].upper() for s in stages)


def conditions(start, examples):
    rows = []
    groups = [("iid", TRAIN), ("heldpair", HELD_PAIRS), ("triple", TRIPLES)]
    for g, (prefix, seqs) in enumerate(groups):
        for i, st in enumerate(seqs):
            rows.append({"name": name(st, prefix), "world_family": "modular", "seed_start": start + 1_000_000*g + 10_000*i,
                         "examples": examples, "world": {**BASE, "stages": list(st)}})
    # budget/size shift on the new primitive (labelled budget-bound region)
    for i, st in enumerate([("assign",), ("select", "assign")]):
        rows.append({"name": name(st, "hardassign"), "world_family": "modular", "seed_start": start + 4_000_000 + 10_000*i,
                     "examples": examples, "world": {**BASE, "stages": list(st), "tasks": 6, "slots": 4, "forbid": 0.35}})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="[]")
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--examples", type=int, default=256)
    p.add_argument("--references", default="modular_cheap,modular_always_tool,modular_cheap_first")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    cks = json.loads(a.checkpoints) if a.checkpoints.startswith("[") else json.load(open(a.checkpoints))
    cfg = {"checkpoints": cks, "evaluation_batch": 32, "references": [r for r in a.references.split(",") if r],
           "reference_compute_tariff": 1.0, "world_family": "modular",
           "address_namespace": f"e16-modular-{a.seed_base}", "conditions": conditions(a.seed_base, a.examples)}
    json.dump(cfg, open(a.output, "w"), indent=2)
