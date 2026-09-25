#!/usr/bin/env python3
"""Assemble phase3b-independent-audit.json from the three audit outputs plus verdicts.

    python3 phase3b_assemble.py RAW.json DETAIL.json STATIC.json > phase3b-independent-audit.json
"""
import json
import sys

raw, detail, static = (json.load(open(p)) for p in sys.argv[1:4])
e = raw["e20_e22"]
for k in ("e20", "e22"):
    for a in e["failures_nonhard"][k].values():
        a["list"] = a["list"][:10]  # keep the artifact small; full lists are reproducible
verdicts = {
    "1_E19": {"verdict": "CONFIRMED (mechanism QUALIFIED)",
              "e17_same_type_mean": [raw["e19"]["arm_mean"][f"e17rl-r{i}"] for i in range(3)],
              "teacher_cheap_first": raw["e19"]["arm_mean"]["reference-modular_cheap_first"],
              "no_tool_greedy_cheap": raw["e19"]["arm_mean"]["reference-modular_cheap"],
              "distractors_not_in_goal": raw["e19"]["distractors"]["not_in_goal"],
              "success_by_distractor_presence": raw["e19"]["success_by_distractor_presence"],
              "e17_success_vs_goal_valid_distractor_r0": detail["e19_r0_success_vs_any_goal_valid_distractor"]},
    "2_E20": {"verdict": "CONFIRMED", "groups_e20": e["groups"]["e20"], "groups_e17": e["groups"]["e17"],
              "e20_r2_failures": e["failures_nonhard"]["e20"]["e20rl-r2"]["failures"],
              "e20_r2_min_max_rejections": [e["failures_nonhard"]["e20"]["e20rl-r2"]["min_rej"],
                                             e["failures_nonhard"]["e20"]["e20rl-r2"]["max_rej"]],
              "rejected_kinds": detail["e20_r2_failure_rejected_kinds"]},
    "3_E20_configs_m2": {"verdict": "CONFIRMED"},
    "4_E21": {"verdict": "CONFIRMED", "arms": {k: v for k, v in raw["e21"].items() if k.startswith("halving")}},
    "5_E22": {"verdict": "CONFIRMED (two wording qualifications)",
              "failures": {k: {a: [v["failures"], v["perseveration"]] for a, v in e["failures_nonhard"][k].items()}
                           for k in ("e20", "e22")},
              "m3_trace_check": raw["m3_trace_check"], "e22_minus_e20_groups": e["e22_minus_e20"]},
    "6_seeds": {"verdict": "CONFIRMED", "overlaps": raw["seeds"]["overlaps"]},
}
cpu = {"remote_raw_pass": raw["cpu_seconds_remote"], "remote_detail_pass": detail["cpu_seconds_remote"],
       "local_static": static["cpu_seconds"]}
cpu["total_measured"] = sum(cpu.values())
json.dump({"audit": "phase3b independent audit (E19-E22)", "verdicts": verdicts, "cpu_seconds": cpu,
           "raw": raw, "detail": detail, "static": static}, sys.stdout, indent=1, default=str)
