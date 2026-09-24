"""Build the sealed E09/E10 evaluation config from frozen finalist checkpoints.

Seed ranges 70_000_000+ are reserved for phase-2 sealed evaluation and are never
used by training, development selection, or profiling. Conditions are fixed here
before any finalist is evaluated; checkpoint paths/hashes are bound at freeze.
"""
import argparse, json

BASE = {"step_limit": 48, "include_remaining_budget": True, "compute_price": 0.0001}
IID = [  # training-mixture components (fresh worlds)
    ("iid_2x2", dict(categories=2, choices=2, locations=5)),
    ("iid_3x3", dict(categories=3, choices=3, locations=7)),
    ("iid_4x4", dict(categories=4, choices=4, locations=9)),
    ("iid_4x4_tight128", dict(categories=4, choices=4, locations=9, work_limit=128)),
    ("iid_4x4_expensive_work", dict(categories=4, choices=4, locations=9, work_price=0.002)),
    ("iid_3x3_obstacle", dict(categories=3, choices=3, locations=7, obstacle=True)),
]
TRANSFER = [  # values or combinations absent from the training mixture
    ("xfer_5x5_larger", dict(categories=5, choices=5, locations=11, step_limit=64)),
    ("xfer_4x4_work256", dict(categories=4, choices=4, locations=9, work_limit=256)),
    ("xfer_4x4_work_price_mid", dict(categories=4, choices=4, locations=9, work_price=0.0005)),
    ("xfer_4x4_obstacle_tight128", dict(categories=4, choices=4, locations=9, obstacle=True, work_limit=128)),
    ("xfer_3x3_obstacle_expensive_work", dict(categories=3, choices=3, locations=7, obstacle=True, work_price=0.002)),
    ("xfer_3x3_expensive_travel", dict(categories=3, choices=3, locations=7, travel_price=0.01)),
    ("xfer_4x4_budget_menu_32_256_2048", dict(categories=4, choices=4, locations=9, call_budgets=[32, 256, 2048])),
]
INTERVENTIONS = [  # paired with a control of identical physical semantics (same seeds)
    ("int_4x4_control", dict(categories=4, choices=4, locations=9), None, None),
    ("int_4x4_no_tools", dict(categories=4, choices=4, locations=9, work_limit=0), "int_4x4_control", None),
    ("int_4x4_work_price_high", dict(categories=4, choices=4, locations=9, work_price=0.002), "int_4x4_control", None),
    ("int_4x4_work_limit_128", dict(categories=4, choices=4, locations=9, work_limit=128), "int_4x4_control", None),
    ("int_4x4_subset_wrong_value", dict(categories=4, choices=4, locations=9), "int_4x4_control",
     [{"kind": "wrong_value", "ordinal": 0, "primitive": "constrained_subset", "payload_path": [0, 0]}]),
    ("int_4x4_subset_absent", dict(categories=4, choices=4, locations=9), "int_4x4_control",
     [{"kind": "absent", "ordinal": 0, "primitive": "constrained_subset"}]),
    ("int_3x3_obstacle_control", dict(categories=3, choices=3, locations=7, obstacle=True), None, None),
    ("int_3x3_obstacle_route_stale", dict(categories=3, choices=3, locations=7, obstacle=True), "int_3x3_obstacle_control",
     [{"kind": "stale", "ordinal": 0, "primitive": "shortest_path"}]),
]


def build(checkpoints, examples, references):
    conditions, start = [], 70_000_000
    for i, (name, world) in enumerate(IID + TRANSFER):
        conditions.append({"name": name, "seed_start": start + 100_000*i, "examples": examples,
                           "world": {**BASE, **world}})
    for name, world, control, faults in INTERVENTIONS:
        # Paired conditions share seeds with their control by construction.
        group = "int_4x4" if name.startswith("int_4x4") else "int_3x3"
        row = {"name": name, "seed_start": start + 5_000_000 + (0 if group == "int_4x4" else 100_000),
               "examples": examples, "world": {**BASE, **world}}
        if control:
            row["paired_control"] = control
        if faults:
            row["return_faults"] = faults
        conditions.append(row)
    return {"checkpoints": checkpoints, "evaluation_batch": 32, "references": references,
            "reference_compute_tariff": 1.0, "address_namespace": "extended-02-phase2-sealed-v1",
            "conditions": conditions,
            "accounting": "Learned and reference controllers pay the same 1.0-unit modeled tariff per decision; actual neural/solver resources recorded separately."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True, help="JSON list of {name,path,sha256}")
    p.add_argument("--examples", type=int, default=256)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    refs = ["cheap", "always_tool", "cheap_first", "cheap_first_fallback_v2"]
    json.dump(build(json.load(open(a.checkpoints)), a.examples, refs), open(a.output, "w"), indent=2)
