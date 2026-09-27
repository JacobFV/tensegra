"""Run the campaign's own campaign06_bprimary.contrast_two_level with other bootstrap seeds / n_boot (gap + later)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tools"))
import campaign06_bprimary as bp

B = os.path.dirname(os.path.abspath(__file__)) + "/bfc"
t0 = time.process_time()
ld = lambda a, s: bp.load(f"{B}/e06-tb-{a}-s{s}/run", "b6c_hold_SCE")
A = [ld("lrn", s) for s in range(35, 40)]
R = [ld("rawf", s) for s in range(35, 40)]
full = dict(bp.ENDPOINTS)
# 1) exact reproduction of the staged run (all endpoints, n_boot 1000, seed 1000)
c = bp.contrast_two_level(A, R, 1000, seed=1000)
print("reproduce:", {k: (round(v["mean"], 4), [round(x, 4) for x in v["ci"]]) for k, v in c["endpoints"].items()})
print("verdict:", bp.verdict_two_level(c))
# 2) gap regret (and later) with other seeds / n_boot
bp.ENDPOINTS = {k: full[k] for k in ("gap_regret", "later_acc")}
res = {}
for nb, seeds in ((1000, range(1000, 1020)), (5000, range(1000, 1005)), (20000, (1000, 1001))):
    for sd in seeds:
        c = bp.contrast_two_level(A, R, nb, seed=sd)
        v = bp.verdict_two_level
        g = c["endpoints"]["gap_regret"]["ci"]; l = c["endpoints"]["later_acc"]["ci"]
        res.setdefault(nb, []).append((sd, g[1], l[0]))
        print(nb, sd, "gap CI", [round(x, 4) for x in g], "later CI", [round(x, 4) for x in l], flush=True)
for nb, xs in res.items():
    ub = [x[1] for x in xs]
    print(f"n_boot {nb}: gap upper bound range {min(ub):+.4f}..{max(ub):+.4f}; Q1-CI pass in {sum(u < 0 for u in ub)}/{len(ub)} bootstrap seeds; later lb min {min(x[2] for x in xs):+.4f}")
print("cpu s", round(time.process_time() - t0, 1))
