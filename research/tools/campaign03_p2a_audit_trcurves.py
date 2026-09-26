#!/usr/bin/env python3
"""P2a auditor: per-tranche training curves from the final checkpoints' `curves`
(C0, C1, P1 X1-rl-r2): anchor KL, tranche KL, critic loss, gradient norm and the
fraction of updates whose pre-clip gradient norm exceeds the clip (1.0)."""
import json
import sys

import torch

R = "/home/brand/tensegra-campaign03/results/"
out = {}
for run in ["p2a-c0-x1-r2", "p2a-c1-x1-r2", "p1-rl-x1-r2"]:
    ck = torch.load(f"{R}{run}/checkpoints/round-4-slot-5-attempt-29.pt", map_location="cpu", weights_only=False)
    cur = [e for e in ck["curves"] if isinstance(e, dict) and e.get("update", 0) > 600]
    tr = []
    for t in range(30):
        seg = [e for e in cur if 600 + 60 * t < e["update"] <= 600 + 60 * (t + 1)]
        op = [e.get("objective_parts") or {} for e in seg]

        def m(key, src=op):
            v = [float(x[key]) for x in src if x.get(key) is not None]
            return sum(v) / len(v) if v else None
        gn = [float(e["gradient_norm"]) for e in seg if e.get("gradient_norm") is not None]
        tr.append({"attempt": t, "n": len(seg), "kl_to_anchor": m("kl_to_anchor"), "kl_to_round_start": m("kl_to_round_start"),
                   "critic_loss": m("critic_decision_mean_loss"), "entropy": m("entropy"), "policy_loss": m("policy_loss"),
                   "grad_norm_mean": sum(gn) / len(gn) if gn else None, "grad_norm_min": min(gn) if gn else None,
                   "clip_fraction": sum(g > 1.0 for g in gn) / len(gn) if gn else None,
                   "training_success": m("training_success", seg), "training_utility": m("mean_training_utility", seg)})
    allg = [float(e["gradient_norm"]) for e in cur if e.get("gradient_norm") is not None]
    out[run] = {"updates": len(cur), "clip_fraction_all": sum(g > 1 for g in allg) / len(allg),
                "grad_norm_min_all": min(allg), "tranches": tr}
    print(run, out[run]["updates"], out[run]["clip_fraction_all"], out[run]["grad_norm_min_all"])
    for x in tr:
        print("  ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
json.dump(out, open(sys.argv[1], "w"), indent=1)
