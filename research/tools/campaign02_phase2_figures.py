"""Phase-2 figures from archived analysis JSON (no inference). Usage: phase2_figures.py <phase2 dir> <out dir>"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C = {"robust-pbt": "#eda100", "pbt": "#2a78d6", "multistart": "#eb6834", "single": "#1baf7a", "bank0": "#8a8a85", "curriculum-pbt": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

a = json.load(open(P/"E09-analysis-all.json"))
modes = ["bank0", "pbt", "curriculum-pbt", "robust-pbt", "multistart", "single"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, key, title in ((axes[0], "iid_mean_utility", "Sealed IID (8 conditions × 256 worlds)"),
                       (axes[1], "transfer_mean_utility", "Sealed held-out transfer (10 conditions)")):
    for i, m in enumerate(modes):
        for r in range(3):
            v = a[key].get(f"{m}-r{r}")
            if v is not None:
                ax.plot(i + (r-1)*0.12, v, "o", ms=8, color=C[m], mec="white", mew=1.5)
    for ref, style, lab in (("reference-cheap_first", "-", "public teacher (cheap-first)"), ("reference-cheap", ":", "no-tool greedy")):
        ax.axhline(a[key][ref], color=MUTED, ls=style, lw=1)
        ax.text(-0.4, a[key][ref], lab, color=MUTED, fontsize=7, va="bottom", ha="left")
    ax.set_xticks(range(len(modes)), ["bank\n(no RL)", "PBT", "PBT+\ncurric.", "PBT+robust\nfitness", "multi-\nstart", "single"])
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
axes[0].set_ylabel("mean verified utility")
fig.text(0.01, 0.01, "Each dot = one replicate finalist (r0, r1, r2 left→right).", color=MUTED, fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT/"phase2-sealed-utility.png", dpi=160); plt.close(fig)

d = json.load(open(P/"E08-greedy-first-dynamics.json")); d.update(json.load(open(P/"E11-E12-greedy-first-dynamics.json")))
fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
panels = [("single", "e08-main-single"), ("multistart", "e08-main-multistart"), ("pbt", "e08-main-pbt"), ("curriculum-pbt", "e12-curriculum-pbt")]
for ax, (m, prefix) in zip(axes, panels):
    for r in range(3):
        series = d[f"{prefix}-r{r}"]
        rounds = sorted({x["round"] for x in series})
        mean = [sum(x["greedy_first"] for x in series if x["round"] == k)/6 for k in rounds]
        ax.plot(rounds, mean, "-o", color=C[m], lw=2, ms=4, alpha=[1, .7, .45][r])
        ax.text(rounds[-1]+0.1, mean[-1], f"r{r}", color=MUTED, fontsize=7, va="center")
    ax.set_title({"single": "single learner", "multistart": "multistart", "pbt": "PBT", "curriculum-pbt": "PBT + adaptive curriculum"}[m], fontsize=9, loc="left", color=INK)
    ax.set_xlabel("RL round (60 updates/slot)"); ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", color=GRID, lw=0.8)
axes[0].set_ylabel("greedy-first rate\n(mean over 6 slot evals, DEV)")
fig.tight_layout(); fig.savefig(OUT/"phase2-greedy-first-dynamics.png", dpi=160); plt.close(fig)

t = a["table"]
conds = [("int_4x4_control", "control"), ("int_4x4_no_tools", "no tools"), ("xfer_5x5_subset_tool_invalid", "tool-invalid\n(25 items)"),
         ("int_4x4_subset_wrong_value", "corrupted\nsubset return"), ("xfer_4x4_tight64", "tight64\n(unseen)")]
groups = [("greedy-first learned\n(single-r0/r1, multistart-r1, E11 light.)", ["single-r0", "single-r1", "multistart-r1", "arch-lightweight-rl"], C["single"]),
          ("tool-first learned\n(PBT×3, PBT+curr.×3, ms-r0/r2, single-r2)", ["pbt-r0", "pbt-r1", "pbt-r2", "curriculum-pbt-r0", "curriculum-pbt-r1", "curriculum-pbt-r2", "multistart-r0", "multistart-r2", "single-r2"], C["pbt"]),
          ("public teacher", ["reference-cheap_first"], MUTED)]
fig, ax = plt.subplots(figsize=(9, 3.2))
w = 0.26
for gi, (label, arms, color) in enumerate(groups):
    for ci, (c, _) in enumerate(conds):
        vals = [t[c][x]["success"] for x in arms]
        x = ci + (gi-1)*w
        ax.bar(x, sum(vals)/len(vals), w*0.92, color=color, label=label if ci == 0 else None)
        ax.plot([x]*len(vals), vals, "o", ms=3, color=INK, alpha=0.55)
ax.set_xticks(range(len(conds)), [c[1] for c in conds]); ax.set_ylabel("sealed verified success")
ax.set_ylim(0, 1.05); ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
fig.tight_layout(); fig.savefig(OUT/"phase2-interventions.png", dpi=160); plt.close(fig)
print("ok")
