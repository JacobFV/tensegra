#!/usr/bin/env python3
"""Collect the Track B independent-audit outputs into one JSON (review/b-independent-audit.json).
Usage: campaign05_b_audit_collect.py SCRATCH_DIR OUT_JSON
"""
import json
import os
import sys


def main(d, out):
    j = lambda f: json.load(open(os.path.join(d, f)))  # noqa: E731
    res = {
        "audit": "extended-05 Track B independent audit",
        "base": "campaign/extended-05 @ 88bd3b0c",
        "metrics_bx_bq_bxc": j("audit.json"),
        "split_integrity": j("split_full.json"),
        "bloc": j("bloc_audit.json"),
        "bloc_model_variation_control": j("bloc_control_summary.json"),
        "extras": j("extras.json"),
        "reproduction_vs_root": {"bx_bq_values_compared": 630, "bxc_values_compared": 120,
                                 "max_abs_diff": 1.1e-14, "bloc_class_agreement": 1.0,
                                 "bloc_count_diff": 0},
        "timeline_utc": {
            "bx_ni_registered_9226196a": "02:06:33", "bx_arms_launch": "02:43:34", "bx_scored_f30a1b0f": "02:56:25",
            "bq_registered_6a77eae7": "02:57:51", "bq_scored_d8ba0363": "03:07:54",
            "bxc_registered_0923cff2": "03:08:34", "bxc_trainings_start": "03:09:03",
            "b5c_tooling_f22ece8d": "03:22:14", "bxc_labels_start": "03:23:05", "bxc_evals_start": "03:35:48",
            "bxc_confirmed_22d326cc": "03:40:34",
            "registry_entries_unchanged_since_0923cff2": ["B-XC", "B-X", "B-Q", "B-SPLIT"]},
        "compute": {"metered_core_s": {"baudit-pools": 67.5, "baudit-bloc-control": 5.2, "total": 72.7},
                    "unmetered_remote_est_core_s": 10},
    }
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
