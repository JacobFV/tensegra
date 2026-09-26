"""Compose review/f-independent-audit.json from the audit script outputs (f1_audit, f2_audit, fb_remote, extra)."""
import json, sys

d = sys.argv[1]
f1 = json.load(open(f"{d}/f1.json"))
f2 = json.load(open(f"{d}/f2_local.json"))
fb = json.load(open(f"{d}/fb.json"))
ex = json.load(open(f"{d}/extra.json"))
out = {
    "audit": "extended-04 Phase F (F1, F1b, F2) + Track B2 independent audit",
    "scripts": ["research/tools/audit_f/f1_audit.py", "research/tools/audit_f/f2_audit.py",
                "research/tools/audit_f/fb_remote.py", "research/tools/audit_f/extra.py"],
    "F1_F1b": {exp: {"verdict_per_lineage": f1["experiments"][exp]["verdict_per_lineage"],
                     "episode_clause_H": f1["experiments"][exp]["F_episode_H"],
                     "per_step_clause_H": f1["experiments"][exp]["F_step_H"],
                     "iid_group": f1["experiments"][exp]["iid"],
                     "cells": f1["experiments"][exp]["cells"],
                     "floor_all_pass": all(f1["experiments"][exp]["floor"].values()),
                     "mechanism": f1["experiments"][exp]["mechanism"],
                     "worlds": f1["experiments"][exp]["worlds"],
                     "paired_iid": {k.split("/", 1)[1]: v for k, v in ex["F"].items() if k.startswith(exp + "/")}}
               for exp in ("f1", "f1b")},
    "F1_integrity": {"f1_f1b_spec_overlap": f1["integrity"]["f1_f1b_spec_overlap"],
                     "other_eval_world_overlap": f1["integrity"]["other_eval_world_overlap"],
                     "checkpoints": {k: {kk: v[kk] for kk in v if kk not in ("state_excerpt", "state_keys")}
                                     for k, v in f1["integrity"]["checkpoints"].items()},
                     "receipts": f1["integrity"]["receipts"],
                     "diagnostic_class_counts": f1["class_letters"]},
    "F2": {"recomputed": f2, "world_freshness": fb["F2"]["world_freshness"], "train_meta": fb["F2"]["train_meta"],
           "remote_eval_sha256": fb["F2"]["eval_sha256"], "remote_f_brefs_sha256": fb["F2"]["f_brefs_sha256"],
           "commands": {k: v for k, v in fb["F2"]["commands"].items() if k.startswith("f-")}},
    "B2": {**fb["B2"], "pooled_commit_auroc_v_own": ex["B2"]},
}
json.dump(out, open(sys.argv[2], "w"), indent=1)
