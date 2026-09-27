"""Extended-05 machine-readable job plan: every planned job's dimensions x measured unit costs (from ext-04 receipts).

Unit costs (core-s), measured in extended-04:
  depworld eval episode (fast path, GPU actor)        0.10  (F1b 0.075; A1 0.17 incl. diagnostic, 12 concurrent)
  depworld branch continuation (clone + follow D)     0.15  (Track C labels: ~0.55/point at ~3.6 branches)
  depworld long-episode factor (tail, collapsed/96-step) x2 on 10% of branches
  depworld supervised/ranking-head training (1,800 upd) 1,800 ; RL 4,000
  probeworld labels 350 ; train 530/run ; eval 45/model (5 splits)
Contingency 25% on every line; audits listed explicitly.
"""
import json, sys

U = {"dep_ep": 0.10, "branch": 0.15, "tail": 1.10, "dep_train": 1800, "pw_labels": 350, "pw_train": 530, "pw_eval": 45}
jobs = []
def add(name, stage, dims, unit, per, note="", gated_by=None):
    n = 1
    for v in dims.values(): n *= v
    cost = n * per
    jobs.append({"name": name, "stage": stage, "dims": dims, "units": n, "unit": unit, "unit_cost": per,
                 "core_s": round(cost), "with_contingency": round(cost * 1.25), "gated_by": gated_by, "note": note})

# Track A
add("A-HR states (D rollouts, dev + 3 fresh lineages)", "B", {"lineages": 4, "conditions": 2, "worlds": 128}, "episode", U["dep_ep"])
add("A-HR branches over O(I)", "B", {"lineages": 4, "conditions": 2, "worlds": 128, "states_per_ep": 4, "options": 12}, "branch", U["branch"] * U["tail"])
add("A-HR full-catalog subsample", "B", {"lineages": 2, "states": 100, "options": 60}, "branch", U["branch"] * U["tail"])
add("A-HR oracle one-step-rollout policy episodes (H bound)", "B", {"lineages": 2, "conditions": 1, "worlds": 32, "steps": 30, "options": 12}, "branch", U["branch"] * U["tail"], "online rollout policy on a subsample; per-state headroom comes from A-HR branches")
add("A-PI labels (branch O(I) on training states)", "C", {"lineages": 1, "worlds": 768, "states_per_ep": 4, "options": 12}, "branch", U["branch"] * U["tail"], gated_by="G1")
add("A-PI ranking head training (+ teacher-sup comparator)", "C", {"arms": 2, "seeds": 1}, "run", 600, gated_by="G1")
add("A-PI screen eval", "C", {"policies": 5, "conditions": 4, "worlds": 256}, "episode", U["dep_ep"] * 1.3, "ranking head forward charged", gated_by="G1")
add("A-CF labels (3 fresh lineages)", "D", {"lineages": 3, "worlds": 768, "states_per_ep": 4, "options": 12}, "branch", U["branch"] * U["tail"], gated_by="A-PI promotion")
add("A-CF training", "D", {"lineages": 3, "arms": 2}, "run", 600, gated_by="A-PI promotion")
add("A-CF sealed eval", "D", {"lineages": 3, "policies": 3, "conditions": 4, "worlds": 512}, "episode", U["dep_ep"] * 1.3, gated_by="A-PI promotion")
# Track B
add("B-LOC exact localization (existing models, challenge sets)", "B", {"models": 12, "splits": 2, "episodes": 512}, "episode", 0.02, "pure Python DP + model forward")
add("B-SPLIT labels (new holds)", "B", {"label_sets": 1}, "labels", U["pw_labels"] * 1.5)
add("B-X training", "C", {"arms": 3, "seeds": 3}, "run", U["pw_train"] * 1.2, "BX2 adds belief features")
add("B-X eval + oracle diagnostic", "C", {"models": 9 + 3, "splits": 1}, "model", U["pw_eval"] * 1.5)
add("B-CF training (fresh seeds, promoted arm + B0)", "D", {"arms": 2, "seeds": 3}, "run", U["pw_train"] * 1.2, gated_by="B-X promotion")
add("B-CF eval", "D", {"models": 6}, "model", U["pw_eval"] * 1.5, gated_by="B-X promotion")
# dev/test/audit
add("dev/tests/smokes (metered)", "all", {"estimate": 1}, "lump", 8000)
add("independent audits (3)", "E", {"audits": 3}, "audit", 1500)

tot = sum(j["with_contingency"] for j in jobs)
ungated = sum(j["with_contingency"] for j in jobs if j["gated_by"] is None)
plan = {"campaign": "extended-05", "unit_costs_core_s": U, "contingency": 0.25, "jobs": jobs,
        "total_with_contingency_core_s": tot, "ungated_with_contingency_core_s": ungated,
        "ceiling_core_s": 172800, "usable_before_reserve_core_s": 138240,
        "gpu_note": "depworld eval/branch jobs use the GPU actor; occupancy tracked as union of main-job wall intervals"}
json.dump(plan, open(sys.argv[1] if len(sys.argv) > 1 else "jobplan.json", "w"), indent=1)
for j in jobs: print(f"{j['stage']:3s} {j['with_contingency']:7d}  {j['name']}" + (f"  [gated: {j['gated_by']}]" if j['gated_by'] else ""))
print("TOTAL (with 25% contingency):", tot, " ungated:", ungated, " usable:", 138240)
