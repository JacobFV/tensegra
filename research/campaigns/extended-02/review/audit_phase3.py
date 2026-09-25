"""Phase-3 independent audit: local analysis (python3 stdlib only).

Input: raw JSON printed by audit_phase3_remote.py (read-only extraction on gb10-direct).
Also reads repo configs (configs/campaign02/*.json) and budget.json.
Usage: python3 audit_phase3.py RAW_JSON REPO_ROOT WORLDS_EQ_JSON > phase3-independent-audit.json
(WORLDS_EQ_JSON from audit_phase3_worlds_remote.py)
Written independently; does not import campaign analysis tools or earlier audit scripts.
"""
import glob, json, math, os, resource, sys, time

T0 = time.time()
raw = json.load(open(sys.argv[1]))
REPO = sys.argv[2]
S = raw["sealed"]
W = raw["worlds"]
R = raw["runs"]
res = {}

IID = ["iid_3x3", "iid_4x4", "iid_4x4_tight128", "iid_4x4_expensive_work", "iid_3x3_obstacle",
       "iid_4x4_obstacle", "iid_4x5", "iid_4x4_steps36"]
XFER = sorted(c for c in S["e09v2-sealed-r0"] if c.startswith("xfer_"))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def rows(root, cond, arm):
    return S[root][cond][arm]


def cond_mean(root, cond, arm, idx):  # idx 2 success, 3 utility
    return mean(r[idx] for r in rows(root, cond, arm))


def eqw(root, conds, arm, idx):
    return mean(cond_mean(root, c, arm, idx) for c in conds)


def paired(pairs):
    """pairs: list of (rootA, armA, rootB, armB, conds). Per-world diffs pooled; normal CI."""
    d = []
    for ra, aa, rb, ab, conds in pairs:
        for c in conds:
            A = {r[0]: (r[1], r[3]) for r in rows(ra, c, aa)}
            B = {r[0]: (r[1], r[3]) for r in rows(rb, c, ab)}
            assert set(A) == set(B), (ra, rb, c)
            for s in A:
                assert A[s][0] == B[s][0], ("spec mismatch", c, s)
                d.append(A[s][1] - B[s][1])
    m = mean(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    se = sd / math.sqrt(len(d))
    return {"mean": m, "ci95_normal": [m - 1.96 * se, m + 1.96 * se], "n_worlds": len(d)}


# ---------------- Claim 1: E15 ----------------
c1 = {"iid_utility": {}, "transfer_utility": {}, "diffs": {}, "finalists": {}, "mechanics": {}}
arms_old = {"pbt": "pbt", "multistart": "multistart", "single": "single"}
for rep in range(3):
    old = "e09v2-sealed-r%d" % rep
    for a in ("halving-plain", "halving-niche"):
        c1["iid_utility"]["%s-r%d" % (a, rep)] = eqw("e09v2-sealed-e15", IID, "%s-r%d" % (a, rep), 3)
        c1["transfer_utility"]["%s-r%d" % (a, rep)] = eqw("e09v2-sealed-e15", XFER, "%s-r%d" % (a, rep), 3)
    for a in ("pbt", "multistart", "single"):
        c1["iid_utility"]["%s-r%d" % (a, rep)] = eqw(old, IID, "%s-r%d" % (a, rep), 3)
        c1["transfer_utility"]["%s-r%d" % (a, rep)] = eqw(old, XFER, "%s-r%d" % (a, rep), 3)
for a in ("pbt", "multistart", "single"):
    per = [c1["iid_utility"]["halving-plain-r%d" % r] - c1["iid_utility"]["%s-r%d" % (a, r)] for r in range(3)]
    c1["diffs"]["plain_minus_" + a] = {
        "iid_per_replicate": per,
        "iid_pooled": paired([("e09v2-sealed-e15", "halving-plain-r%d" % r, "e09v2-sealed-r%d" % r, "%s-r%d" % (a, r), IID) for r in range(3)]),
        "transfer_pooled": paired([("e09v2-sealed-e15", "halving-plain-r%d" % r, "e09v2-sealed-r%d" % r, "%s-r%d" % (a, r), XFER) for r in range(3)]),
        "iid_per_replicate_paired": [paired([("e09v2-sealed-e15", "halving-plain-r%d" % r, "e09v2-sealed-r%d" % r, "%s-r%d" % (a, r), IID)]) for r in range(3)],
    }
c1["diffs"]["niche_minus_plain"] = {
    "iid_per_replicate": [c1["iid_utility"]["halving-niche-r%d" % r] - c1["iid_utility"]["halving-plain-r%d" % r] for r in range(3)],
    "iid_pooled": paired([("e09v2-sealed-e15", "halving-niche-r%d" % r, "e09v2-sealed-e15", "halving-plain-r%d" % r, IID) for r in range(3)]),
    "transfer_pooled": paired([("e09v2-sealed-e15", "halving-niche-r%d" % r, "e09v2-sealed-e15", "halving-plain-r%d" % r, XFER) for r in range(3)]),
}
# finalist greedy-first/no-tools (control / no-tools conditions) for plain
for r in range(3):
    arm = "halving-plain-r%d" % r
    c1["finalists"][arm + "_no_tools_success"] = cond_mean("e09v2-sealed-e15", "int_4x4_no_tools", arm, 2)

mech_issues = []
for arm in ("plain", "niche"):
    for rep in range(3):
        run = "e15-halving-%s-r%d" % (arm, rep)
        st = R[run]
        al = st["allocations"]
        info = {"survivors_protocol": st["halving_survivors"], "finalist_member": st["finalist"]["member"],
                "finalist_cumulative": st["finalist"]["cumulative_slot_updates"]}
        # per round: members that played, and their slot sequence
        by_round = {}
        for a in al:
            by_round.setdefault(a["round"], []).append(a)
        played = []
        for rd in sorted(by_round):
            A = sorted(by_round[rd], key=lambda a: a["slot"])
            mems = []
            for a in A:
                if not mems or mems[-1] != a["member"]:
                    mems.append(a["member"])
            if len(mems) != len(set(mems)):
                mech_issues.append("%s round %d: non-consecutive slots %s" % (run, rd, [a["member"] for a in A]))
            k = len(set(mems))
            played.append(k)
            per = {m: [a for a in A if a["member"] == m] for m in set(mems)}
            if any(len(v) != 6 // k for v in per.values()):
                mech_issues.append("%s round %d: uneven slots" % (run, rd))
            for m, v in per.items():
                cum = [x["cumulative_slot_updates"] for x in v]
                if any(cum[i + 1] - cum[i] != 60 for i in range(len(cum) - 1)):
                    mech_issues.append("%s round %d member %d cumulative not +60" % (run, rd, m))
            # training seed continuity per member across all rounds
        for m in set(a["member"] for a in al):
            iv = [a["training_seed_interval"] for a in sorted(al, key=lambda a: (a["round"], a["slot"])) if a["member"] == m]
            if any(iv[i][1] != iv[i + 1][0] for i in range(len(iv) - 1)):
                mech_issues.append("%s member %d training stream not contiguous" % (run, m))
        info["members_per_round"] = played
        # cut legality from allocations (last slot utility of each member in round)
        cuts = []
        for rd in sorted(by_round)[:-1]:
            A = sorted(by_round[rd], key=lambda a: a["slot"])
            last = {}
            for a in A:
                last[a["member"]] = a
            nxt = sorted(set(a["member"] for a in by_round[rd + 1]))
            k = len(nxt)
            ranked = sorted(last.values(), key=lambda a: (-a["utility"], a["member"]))
            if arm == "plain" or k < 2:
                expect = sorted(a["member"] for a in ranked[:k])
            else:
                keep = []
                hi = [a for a in ranked if a["behavior_greedy_first"] >= 0.5]
                lo = [a for a in ranked if a["behavior_greedy_first"] < 0.5]
                if hi:
                    keep.append(hi[0]["member"])
                if lo:
                    keep.append(lo[0]["member"])
                for a in ranked:
                    if len(keep) >= k:
                        break
                    if a["member"] not in keep:
                        keep.append(a["member"])
                expect = sorted(keep[:k])
            ok = expect == nxt
            cuts.append({"round": rd, "kept": nxt, "expected_by_rule": expect, "ok": ok})
            if not ok:
                mech_issues.append("%s round %d cut %s != rule %s" % (run, rd, nxt, expect))
        info["cuts"] = cuts
        # final pick: best final-round last-slot utility
        A = sorted(by_round[max(by_round)], key=lambda a: a["slot"])
        last = {}
        for a in A:
            last[a["member"]] = a
        best = sorted(last.values(), key=lambda a: (-a["utility"], a["slot"]))[0]
        info["finalist_rule_ok"] = best["member"] == st["finalist"]["member"]
        info["finalist_rl_updates"] = st["finalist"]["cumulative_slot_updates"] - 600
        info["lineage_halving_events"] = [{"round": e.get("round"), "kept": e.get("kept"), "dropped": e.get("dropped")}
                                          for e in st["lineage_events"] if e.get("kind") == "halving"]
        c1["mechanics"][run] = info
c1["mechanics_issues"] = mech_issues
c1["finalist_member0_count"] = sum(1 for v in c1["mechanics"].values() if v["finalist_member"] == 0)
res["claim1_E15"] = c1

# ---------------- Modular helpers ----------------
IIDP = ["iid_S-R", "iid_S-A", "iid_A-R"]
HELD = ["heldpair_R-S", "heldpair_A-S", "heldpair_R-A"]
TRI = ["triple_A-R-S", "triple_A-S-R", "triple_R-A-S", "triple_R-S-A", "triple_S-A-R", "triple_S-R-A"]
TRI_S_AFTER_A = [t for t in TRI if t.split("_")[1].split("-").index("S") > t.split("_")[1].split("-").index("A")]


def succ(root, cond, arm):
    return cond_mean(root, cond, arm, 2)


# ---------------- Claim 2: E16 ----------------
c2 = {"success": {}, "rule": {}}
for arm in ["rl-r0", "rl-r1", "rl-r2", "boot-r0", "boot-r1", "boot-r2", "reference-modular_cheap_first", "reference-modular_cheap"]:
    c2["success"][arm] = {c: succ("e16-sealed", c, arm) for c in S["e16-sealed"]}
for arm in ["rl-r0", "rl-r1", "rl-r2"]:
    iidp = mean(succ("e16-sealed", c, arm) for c in IIDP)
    held = mean(succ("e16-sealed", c, arm) for c in HELD)
    tri = mean(succ("e16-sealed", c, arm) for c in TRI)
    c2["rule"][arm] = {"iid_pair_mean": iidp, "heldpair_mean": held, "triple_mean": tri,
                       "pass_mean_reading": held >= 0.9 * iidp and tri >= 0.8 * iidp,
                       "pass_each_condition_reading": all(succ("e16-sealed", c, arm) >= 0.9 * iidp for c in HELD) and all(succ("e16-sealed", c, arm) >= 0.8 * iidp for c in TRI),
                       "failure_criterion": held < 0.5 * iidp}
c2["lineages_pass_mean_reading"] = sum(v["pass_mean_reading"] for v in c2["rule"].values())
c2["lineages_pass_each_condition_reading"] = sum(v["pass_each_condition_reading"] for v in c2["rule"].values())
loc = {}
for arm in ["rl-r0", "rl-r1", "rl-r2"]:
    rr = rows("e16-sealed", "heldpair_A-S", arm)
    fail = [r for r in rr if not r[2]]
    loc[arm] = {"failed": len(fail),
                "failed_with_assign_completed": sum(1 for r in fail if r[6] and "assign" in r[6]),
                "failed_with_select_completed": sum(1 for r in fail if r[6] and "select" in r[6]),
                "use_return_mismatch_any": sum(r[7] for r in fail),
                "use_return_mismatch_as_select": sum(r[8] for r in fail),
                "use_return_mismatch_as_select_on_csp": sum(r[9] for r in fail),
                "failed_steps_eq_64": sum(1 for r in fail if r[4] == 64),
                "mismatch_all_rows_as_select_csp": sum(r[9] for r in rr)}
    rs = rows("e16-sealed", "heldpair_R-S", arm)
    loc[arm]["R-S_failed"] = sum(1 for r in rs if not r[2])
    loc[arm]["R-S_mismatch_as_select"] = sum(r[8] for r in rs)
c2["localization_heldpair_A-S"] = loc
res["claim2_E16"] = c2

# ---------------- Claim 3: E16 distractor diagnostic ----------------
c3 = {}
for cond in S["e16-sealed-distractors"]:
    wd = {w[0]: w[2] for w in W["e16-sealed-distractors/" + cond]}
    zero_frac = mean(1 if v == 0 else 0 for v in wd.values())
    entry = {"zero_distractor_fraction": zero_frac, "distractor_count_hist": {}}
    for v in wd.values():
        entry["distractor_count_hist"][str(v)] = entry["distractor_count_hist"].get(str(v), 0) + 1
    for arm in S["e16-sealed-distractors"][cond]:
        rr = rows("e16-sealed-distractors", cond, arm)
        s0 = [r for r in rr if wd[r[0]] == 0]
        s1 = [r for r in rr if wd[r[0]] > 0]
        prior_mismatch = sum(1 for r in rr if r[5] is not None and r[5] != wd[r[0]])
        entry[arm] = {"success": mean(r[2] for r in rr), "success_zero_distr": mean(r[2] for r in s0) if s0 else None,
                      "success_with_distr": mean(r[2] for r in s1) if s1 else None,
                      "successes_with_distr": sum(r[2] for r in s1),
                      "failures_zero_distr": sum(1 - r[2] for r in s0),
                      "first_obs_prior_count_mismatch_vs_spec": prior_mismatch,
                      "first_obs_priors_available": sum(1 for r in rr if r[5] is not None)}
    c3[cond] = entry
res["claim3_E16_distractors"] = c3

# ---------------- Claim 4: E17 ----------------
c4 = {"success": {}, "rule": {}}
for arm in ["e17rl-r0", "e17rl-r1", "e17rl-r2"]:
    c4["success"][arm] = {c: succ("e17-sealed", c, arm) for c in S["e17-sealed"]}
e16AS = [succ("e16-sealed", "heldpair_A-S", "rl-r%d" % i) for i in range(3)]
e17AS = [succ("e17-sealed", "heldpair_A-S", "e17rl-r%d" % i) for i in range(3)]
e16ASx = [mean([succ("e16-sealed", c, "rl-r%d" % i) for c in ["heldpair_A-S"] + TRI_S_AFTER_A]) for i in range(3)]
e17ASx = [mean([succ("e17-sealed", c, "e17rl-r%d" % i) for c in ["heldpair_A-S"] + TRI_S_AFTER_A]) for i in range(3)]
iidp17 = [mean(succ("e17-sealed", c, "e17rl-r%d" % i) for c in IIDP) for i in range(3)]
c4["rule"] = {
    "pair_only": {"e16": e16AS, "e17": e17AS, "mean_gain": mean(e17AS) - mean(e16AS), "min_e17": min(e17AS)},
    "pair_plus_S_after_A_triples": {"conds": ["heldpair_A-S"] + TRI_S_AFTER_A, "e16": e16ASx, "e17": e17ASx,
                                   "mean_gain": mean(e17ASx) - mean(e16ASx), "min_e17": min(e17ASx)},
    "iid_pairs_per_lineage": iidp17, "iid_pairs_mean": mean(iidp17),
    "iid_pairs_min_single_condition": min(succ("e17-sealed", c, "e17rl-r%d" % i) for c in IIDP for i in range(3)),
}
rp = c4["rule"]
rp["pass_mean_reading_pair"] = rp["pair_only"]["mean_gain"] >= 0.2 and rp["pair_only"]["min_e17"] >= 0.8 and rp["iid_pairs_mean"] >= 0.95
rp["pass_mean_reading_pair_plus_triples"] = rp["pair_plus_S_after_A_triples"]["mean_gain"] >= 0.2 and rp["pair_plus_S_after_A_triples"]["min_e17"] >= 0.8 and rp["iid_pairs_mean"] >= 0.95
rp["pass_per_lineage_iid"] = all(x >= 0.95 for x in iidp17)
DISTR = sorted(c for c in S["e17-sealed"] if c.startswith("distr_"))
c4["distractor_mean"] = {a: mean(succ("e17-sealed", c, a) for c in DISTR) for a in ["e17rl-r0", "e17rl-r1", "e17rl-r2"]}
c4["distractor_min"] = {a: min(succ("e17-sealed", c, a) for c in DISTR) for a in ["e17rl-r0", "e17rl-r1", "e17rl-r2"]}
c4["r2_failed_nonsuccess_feedback"] = raw.get("e17_r2_failed_nonsuccess_feedback")
# r2 select-containing failure rate
sel = [c for c in S["e17-sealed"] if not c.startswith("distr_") and not c.startswith("hard") and "S" in c.split("_")[1].split("-")]
c4["r2_select_containing_failure_rate"] = mean(1 - r[2] for c in sel for r in rows("e17-sealed", c, "e17rl-r2"))
c4["mismatch_as_select_total"] = {a: sum(r[8] for c in S["e17-sealed"] for r in rows("e17-sealed", c, a)) for a in ["e17rl-r0", "e17rl-r1", "e17rl-r2"]}
res["claim4_E17"] = c4

# ---------------- Claim 5: E18 ----------------
c5 = {"means": {}, "retention": {}}
ASSIGN = [c for c in S["e18-sealed"] if not c.startswith("distr_") and not c.startswith("hard") and "A" in c.split("_")[1].split("-")]
SRONLY = [c for c in S["e18-sealed"] if not c.startswith("distr_") and not c.startswith("hard") and "A" not in c.split("_")[1].split("-")]
c5["assign_conditions"] = ASSIGN
c5["retention_conditions"] = SRONLY
for arm in ["adapt120", "scratch120", "zeroshot", "adapt20", "scratch20"]:
    c5["means"][arm] = [mean(succ("e18-sealed", c, "%s-r%d" % (arm, i)) for c in ASSIGN) for i in range(3)]
    c5["retention"][arm] = [mean(succ("e18-sealed", c, "%s-r%d" % (arm, i)) for c in SRONLY) for i in range(3)]
d = [c5["means"]["adapt120"][i] - c5["means"]["scratch120"][i] for i in range(3)]
c5["adapt_minus_scratch_120"] = d
npos = sum(1 for x in d if x >= 0.10)
nneg = sum(1 for x in d if x <= 0)
c5["rule"] = "SUPPORTED" if npos >= 2 else ("NEGATIVE" if nneg >= 2 else "PARTIAL")
res["claim5_E18"] = c5

# ---------------- Claim 6: paired-world integrity ----------------
c6 = {"within_root_mismatches": [], "cross_root_mismatches": [], "worlds_file_mismatches": [], "checked": 0}
sig = {}
for root, conds in S.items():
    for cond, arms in conds.items():
        base = None
        wkey = "%s/%s" % (root, cond)
        if wkey in W:
            base = [(w[0], w[1]) for w in W[wkey]]
        for arm, rr in arms.items():
            s = sorted((r[0], r[1]) for r in rr)
            c6["checked"] += 1
            if base is not None and s != sorted(base):
                c6["worlds_file_mismatches"].append(wkey + ":" + arm)
            if len(s) != 256 or len(set(x[0] for x in s)) != 256:
                c6["within_root_mismatches"].append("%s:%s n=%d" % (wkey, arm, len(s)))
            prev = sig.get((root, cond))
            if prev is None:
                sig[(root, cond)] = s
            elif prev != s:
                c6["within_root_mismatches"].append(wkey + ":" + arm)
groups = {}
for (root, cond), s in sig.items():
    groups.setdefault(cond, []).append((root, s))
cross = {}
for cond, lst in groups.items():
    if len(lst) > 1:
        same = all(x[1] == lst[0][1] for x in lst)
        cross[cond] = {"roots": [x[0] for x in lst], "identical": same, "seed_min": lst[0][1][0][0], "seed_max": lst[0][1][-1][0]}
        if not same:
            c6["cross_root_mismatches"].append(cond)
c6["cross_root"] = cross
# E16 distractor root vs E17/E18 distr_ conditions
res["claim6_paired_integrity"] = c6

# ---------------- Claim 7: seed disjointness ----------------
sealed_ranges = []
for f in sorted(glob.glob(os.path.join(REPO, "configs/campaign02/*.json"))):
    c = json.load(open(f))
    if "sealed" in os.path.basename(f):
        for cd in c.get("conditions", []):
            sealed_ranges.append((cd["seed_start"], cd["seed_start"] + cd["examples"], os.path.basename(f), cd["name"]))
obs_sealed = []
for root, conds in S.items():
    for cond, arms in conds.items():
        seeds = [r[0] for rr in arms.values() for r in rr]
        obs_sealed.append((min(seeds), max(seeds) + 1, root, cond))
used = []
for run, st in R.items():
    for a in st["allocations"]:
        iv = a.get("training_seed_interval")
        if iv:
            used.append((iv[0], iv[1], run, "train r%s s%s" % (a["round"], a["slot"])))
    if st.get("dev_seed_min") is not None:
        used.append((st["dev_seed_min"], st["dev_seed_max"] + 1, run, "development(observed %d)" % st["dev_seed_count"]))
    if st.get("development_seed_start") is not None:
        used.append((st["development_seed_start"], st["development_seed_start"] + (st.get("development_examples") or 0), run, "development(protocol)"))
# non-sealed configs with seed_start/examples or episodes (profiles, leverage, transfer, phase-1)
for f in sorted(glob.glob(os.path.join(REPO, "configs/campaign02/*.json"))):
    if "sealed" in os.path.basename(f):
        continue
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("seed_start"), int):
                n = x.get("examples") or x.get("episodes") or 1024
                used.append((x["seed_start"], x["seed_start"] + n, os.path.basename(f), "config seed_start"))
            for k in ("validation_seed", "first_seed", "training_seed_start"):
                if isinstance(x.get(k), int) and not isinstance(x.get(k), bool):
                    used.append((x[k], x[k] + 10 ** 6, os.path.basename(f), k + " (+1e6 conservative window)"))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(json.load(open(f)))
overlaps = []
for s0, s1, sf, sn in sealed_ranges + [(a, b, r, c) for a, b, r, c in obs_sealed]:
    for u0, u1, uf, un in used:
        if u0 < s1 and s0 < u1:
            overlaps.append({"sealed": [s0, s1, sf, sn], "used": [u0, u1, uf, un]})
res["claim7_seed_disjointness"] = {
    "sealed_config_span": [min(x[0] for x in sealed_ranges), max(x[1] for x in sealed_ranges)],
    "sealed_observed_spans": {root: [min(x[0] for x in obs_sealed if x[2] == root), max(x[1] for x in obs_sealed if x[2] == root)] for root in S},
    "used_ranges_checked": len(used),
    "min_used_start_ge_sealed": sorted(set(u[0] for u in used if u[0] >= 60_000_000))[:5],
    "used_starts_below_100M": sorted(set((u[0], u[2]) for u in used if u[0] < 100_000_000)),
    "overlaps": overlaps,
}

# ---------------- Claim 8: held-out integrity ----------------
HELD_ORDERS = {("route", "select"), ("assign", "select"), ("route", "assign")}
c8 = {"runs": {}, "violations": []}
for run, st in R.items():
    if not run.startswith(("e16-", "e17-", "e18-")):
        continue
    mix = st["world_mix"] or []
    orders = sorted(set(tuple(w.get("stages") or []) for w in mix))
    c8["runs"][run] = {"orders": orders, "distractors": sorted(set(w.get("distractors", 0) for w in mix)),
                       "max_steps": st["train_max_steps"], "step_limits": sorted(set(w.get("step_limit") for w in mix)),
                       "dev_world_mix_orders": sorted(set(tuple(w.get("stages") or []) for w in (st.get("development_world_mix") or [])))}
    for o in orders:
        if len(o) >= 3 or o in HELD_ORDERS:
            c8["violations"].append("%s trains on %s" % (run, o))
        if run.startswith("e18-base") and "assign" in o:
            c8["violations"].append("%s (E18 base) contains assign %s" % (run, o))
# cross-check repo configs
cfg = {}
for f in sorted(glob.glob(os.path.join(REPO, "configs/campaign02/e1[678]-*.json"))):
    c = json.load(open(f))
    wm = c.get("world_mix")
    if wm is None:
        continue
    orders = sorted(set(tuple(w.get("stages") or []) for w in wm))
    cfg[os.path.basename(f)] = orders
    for o in orders:
        if len(o) >= 3 or o in HELD_ORDERS:
            c8["violations"].append("config %s has %s" % (os.path.basename(f), o))
        if "e18-base" in f and "assign" in o:
            c8["violations"].append("config %s (E18 base) has assign" % os.path.basename(f))
c8["configs"] = cfg
res["claim8_heldout"] = c8

# ---------------- Step caps ----------------
caps = {}
for root in ["e16-sealed", "e16-sealed-distractors", "e17-sealed", "e18-sealed"]:
    for cond, arms in S[root].items():
        for arm, rr in arms.items():
            mx = max(r[4] for r in rr)
            caps.setdefault(root, {}).setdefault(arm, []).append(mx)
res["step_caps"] = {
    "learned_train_max_steps_modular": {r: v["train_max_steps"] for r, v in R.items() if r.startswith(("e16-", "e17-", "e18-"))},
    "learned_train_max_steps_e15_e08": {r: v["train_max_steps"] for r, v in R.items() if r.startswith(("e15-", "e08-main"))},
    "world_step_limits_modular_sealed": sorted(set(w[3] for k, ws in W.items() if k.split("/")[0] in ("e16-sealed", "e16-sealed-distractors", "e17-sealed", "e18-sealed") for w in ws)),
    "max_observed_steps_per_root_arm": {root: {arm: max(v) for arm, v in d.items()} for root, d in caps.items()},
}

# ---------------- Claim 9: ledger ----------------
b = json.load(open(os.path.join(REPO, "research/campaigns/extended-02/budget.json")))
jobs = b["phase2_jobs"]
tot = sum(j["cpu_core_seconds"] for j in jobs)
occ = raw["occupancy"]
spot = []
for jid in ["e15-halving-plain-r0", "e16-sealed", "e17-sealed", "e18-sealed", "e18-adapt-r2", "e16-sealed-distractors", "e09v2-sealed-e15"]:
    j = next((x for x in jobs if x["id"] == jid), None)
    o = occ.get(jid + "-process")
    spot.append({"id": jid, "budget_cpu": j and j["cpu_core_seconds"], "receipt_cpu": o and o.get("cpu_core_seconds"),
                 "budget_wall": j and j["wall_seconds"], "receipt_wall": o and o.get("wall_seconds"),
                 "match": bool(j and o and abs(j["cpu_core_seconds"] - o["cpu_core_seconds"]) < 1e-6 and abs(j["wall_seconds"] - o["wall_seconds"]) < 1e-6)})
# all phase-3 jobs with receipts
allcmp = []
for j in jobs:
    o = occ.get(j["id"] + "-process")
    if o is not None:
        allcmp.append((j["id"], abs(j["cpu_core_seconds"] - o["cpu_core_seconds"]) < 1e-6))
res["claim9_ledger"] = {"phase2_charged_cpu_core_seconds": b["phase2_charged_cpu_core_seconds"], "sum_phase2_jobs": tot,
                        "equal": abs(tot - b["phase2_charged_cpu_core_seconds"]) < 1e-6,
                        "total_charged_cpu_core_seconds": b["total_charged_cpu_core_seconds"],
                        "phase1_plus_phase2": b["charged_cpu_core_seconds"] + b["phase2_charged_cpu_core_seconds"],
                        "spot_checks": spot,
                        "all_jobs_with_receipts_matching": sum(1 for _, ok in allcmp if ok), "all_jobs_with_receipts": len(allcmp),
                        "mismatching": [i for i, ok in allcmp if not ok],
                        "jobs_without_remote_receipt": [j["id"] for j in jobs if (j["id"] + "-process") not in occ]}
ids = set(j["id"] for j in jobs)
res["claim9_ledger"]["receipts_not_in_budget"] = sorted(k[: -len("-process")] for k in occ if k[: -len("-process")] not in ids and k.startswith(("e15", "e16", "e17", "e18")))
res["claim7_seed_disjointness"]["sealed_config_ranges_by_file"] = {}
for s0, s1, f, n in sealed_ranges:
    cur = res["claim7_seed_disjointness"]["sealed_config_ranges_by_file"].setdefault(f, [s0, s1])
    cur[0] = min(cur[0], s0); cur[1] = max(cur[1], s1)
wq = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
if wq:
    res["claim6_paired_integrity"]["e16_vs_e17_e18_world_content"] = {
        "conditions": len(wq["worlds"]),
        "all_same_seed_order": all(v[o]["same_seed_order"] for v in wq["worlds"].values() for o in v),
        "all_spec_equal_ignoring_empty_distractors": all(v[o]["spec_equal_ignoring_empty_distractors"] == v[o]["n"] for v in wq["worlds"].values() for o in v),
        "any_nonempty_distractors": any(v[o]["nonempty_distractors"] for v in wq["worlds"].values() for o in v),
        "all_address_seed_equal": all(v[o]["address_seed_equal"] == v[o]["n"] for v in wq["worlds"].values() for o in v)}
    res["claim4_E17"]["r2_rejected_commit_pending_per_failed_episode"] = wq["e17_r2_rejected_commit_per_failed_episode"]
ru = resource.getrusage(resource.RUSAGE_SELF)
res["audit_cpu"] = {"remote_extractor_cpu_seconds": raw["audit_cpu_seconds"], "remote_extractor_wall_seconds": raw["audit_wall_seconds"],
                    "local_analysis_cpu_seconds": ru.ru_utime + ru.ru_stime, "local_analysis_wall_seconds": time.time() - T0}
json.dump(res, sys.stdout, indent=1, sort_keys=True, default=str)
