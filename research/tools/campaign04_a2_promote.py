"""Apply the protocol-A2 promotion rule (registered 2026-09-26T19:15Z) to the A2 screening IID-group table."""
import json, sys
a = json.load(open(sys.argv[1]))["iid_groups"]
MODES = ("greedy", "sampled", "r_mask")
REG = {"reh": "greedy", "imit": "greedy", "crit": "greedy", "ent": "greedy", "dep": "r_mask"}
boot = {m: a[f"boot-x1-r2/{m}"] for m in MODES}
best_mode = max(MODES, key=lambda m: boot[m]["utility"])
print("policy/mode".ljust(24), "succ  util   cost   work/s  noprog_ep")
for k in sorted(a):
    r = a[k]
    print(k.ljust(24), f"{r['success']:.3f} {r['utility']:.3f} {r['cost']:.4f} {r.get('work_per_success') or 0:7.1f} {r['no_progress_episode_rate']:.3f}")
print("bootstrap best fixed mode:", best_mode, round(boot[best_mode]["utility"], 4))
out = {}
for arm, mode in REG.items():
    r = a[f"a2-{arm}-final/{mode}"]; b = boot[mode]
    c1 = r["utility"] >= b["utility"] + .01 or (r["cost"] <= .9 * b["cost"] and r["success"] >= b["success"] - .02)
    c2 = r["utility"] >= boot[best_mode]["utility"] + .005
    c3 = r["no_progress_episode_rate"] <= b["no_progress_episode_rate"] + .02
    c4 = True if arm != "reh" else r["utility"] >= a[f"a2-imit-final/{mode}"]["utility"] + .01
    out[arm] = {"mode": mode, "utility": r["utility"], "boot_same_mode": b["utility"], "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                "promoted": c1 and c2 and c3 and c4}
    print(arm, json.dumps(out[arm]))
json.dump(out, open(sys.argv[2], "w"), indent=1) if len(sys.argv) > 2 else None
