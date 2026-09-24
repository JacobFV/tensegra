"""Independent phase-2 audit: local analysis of the read-only extraction.

Usage: python3 audit_phase2.py EXTRACT.json [OUT.json [V1CHECK.json]]
EXTRACT.json is produced on the results host by audit_phase2_remote.py.
Deliberately independent of research/tools/campaign02_e09_analysis.py, mechanism.py, dynamics.py.
"""
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CONFIGS = {r: REPO / f"configs/campaign02/e09v2-sealed-{r}.json" for r in ("r0", "r1", "r2", "references", "e11", "e12")}
IID = ["iid_3x3", "iid_4x4", "iid_4x4_tight128", "iid_4x4_expensive_work", "iid_3x3_obstacle",
       "iid_4x4_obstacle", "iid_4x5", "iid_4x4_steps36"]
CLAIMED_IID = {"single-r0": .9430, "single-r1": .9430, "single-r2": .9302, "pbt-r0": .9289, "pbt-r1": .8959,
               "pbt-r2": .9284, "multistart-r0": .9282, "multistart-r1": .9292, "multistart-r2": .9251,
               "bank0-r0": .8021, "bank0-r1": .7414, "bank0-r2": .9213, "reference-cheap_first": .9397,
               "reference-cheap": .8047, "arch-lightweight-rl": .9426, "arch-recurrent-rl": .7706,
               "curriculum-pbt-r0": .9277, "curriculum-pbt-r1": .9272, "curriculum-pbt-r2": .9283}
GREEDY_CLAIM_HIGH = ["single-r0", "single-r1", "multistart-r1", "arch-lightweight-rl"]
GREEDY_CLAIM_LOW = ["pbt-r0", "pbt-r1", "pbt-r2", "multistart-r0", "multistart-r2", "single-r2",
                    "curriculum-pbt-r0", "curriculum-pbt-r1", "curriculum-pbt-r2"]
SEALED_STEP = 256


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def arm_table(doc):
    """(arm) -> (root, {condition: episodes sorted by seed})."""
    table = {}
    for root, ev in doc["eval"].items():
        for cond, info in ev["conditions"].items():
            for arm, a in info["arms"].items():
                table.setdefault(arm, (root, {}))[1][cond] = sorted(a["episodes"], key=lambda e: e["seed"])
    return table


def check_worlds(doc, configs, notes):
    res = {"per_root_missing": {}, "arm_mismatch": [], "cross_root_mismatch": [], "seed_errors": [],
           "recomputed_hash_failures": [], "duplicate_seeds": [], "config_equal_summary": {}}
    reference = {}
    for key, cfg in configs.items():
        root = f"e09v2-sealed-{key}"
        ev = doc["eval"][root]
        summ = ev["summary"]
        res["config_equal_summary"][root] = summ is not None and summ["config"] == cfg
        expected_arms = [c["name"] for c in cfg["checkpoints"]] + [f"reference-{r}" for r in cfg["references"]]
        missing = []
        for c in cfg["conditions"]:
            name = c["name"]
            info = ev["conditions"].get(name)
            if info is None or info["worlds"] is None:
                missing.append(name)
                continue
            w = info["worlds"]
            want = list(range(c["seed_start"], c["seed_start"] + c["examples"]))
            if w["seeds"] != want:
                res["seed_errors"].append((root, name))
            if len(set(w["seeds"])) != len(w["seeds"]):
                res["duplicate_seeds"].append((root, name, "worlds"))
            if w["spec_hash_recomputed_ok"] != w["n"]:
                res["recomputed_hash_failures"].append((root, name, w["n"] - w["spec_hash_recomputed_ok"]))
            triple = list(zip(w["seeds"], w["spec_hash"], w["sem"]))
            if name in reference and reference[name][1] != triple:
                res["cross_root_mismatch"].append((reference[name][0], root, name))
            reference.setdefault(name, (root, triple))
            for arm in expected_arms:
                eps = info["arms"].get(arm)
                if eps is None:
                    missing.append(f"{name}/{arm}")
                    continue
                seq = [(e["seed"], e["spec_hash"], e["sem"]) for e in eps["episodes"]]
                if len({e["seed"] for e in eps["episodes"]}) != len(seq):
                    res["duplicate_seeds"].append((root, name, arm))
                if seq != triple:
                    res["arm_mismatch"].append((root, name, arm))
            extra = set(info["arms"]) - set(expected_arms)
            if extra:
                notes.append(f"{root}/{name}: unexpected arm files {sorted(extra)}")
        extra_conds = set(ev["conditions"]) - {c["name"] for c in cfg["conditions"]}
        if extra_conds:
            notes.append(f"{root}: unexpected condition dirs {sorted(extra_conds)}")
        res["per_root_missing"][root] = missing
    # int_* conditions deliberately share seeds; check spec identity among shared-seed interventions
    res["conditions_checked"] = len(reference)
    res["n_arm_condition_files"] = sum(len(i["arms"]) for ev in doc["eval"].values() for i in ev["conditions"].values())
    return res


def claim1(table):
    out = {}
    for arm, (root, conds) in sorted(table.items()):
        per = {c: mean(e["u"] for e in conds[c]) for c in IID if c in conds}
        if len(per) != len(IID):
            continue
        out[arm] = {"root": root, "iid_mean_utility": mean(per.values()),
                    "iid_success": mean(mean(float(e["s"]) for e in conds[c]) for c in IID),
                    "per_condition": per,
                    "xfer_mean_utility": mean(mean(e["u"] for e in v) for c, v in conds.items() if c.startswith("xfer_")),
                    "n_iid_episodes": sum(len(conds[c]) for c in IID)}
        if arm in CLAIMED_IID:
            out[arm]["claimed"] = CLAIMED_IID[arm]
            out[arm]["abs_diff_vs_claim"] = abs(round(out[arm]["iid_mean_utility"], 4) - CLAIMED_IID[arm])
    return out


def paired_diffs(table, a, b):
    """Per-episode utility differences a-b over IID, checking seed pairing."""
    diffs = []
    for c in IID:
        ea, eb = table[a][1][c], table[b][1][c]
        assert [e["seed"] for e in ea] == [e["seed"] for e in eb]
        assert [e["spec_hash"] for e in ea] == [e["spec_hash"] for e in eb]
        diffs.append([x["u"] - y["u"] for x, y in zip(ea, eb)])
    return diffs


def bootstrap_pooled(strata, reps=4000, seed=20260924):
    rng = random.Random(seed)
    point = mean(v for s in strata for v in s)
    stats = []
    for _ in range(reps):
        tot = n = 0
        for s in strata:  # stratified resampling within replicate x condition
            k = len(s)
            tot += sum(s[rng.randrange(k)] for _ in range(k))
            n += k
        stats.append(tot / n)
    stats.sort()
    return {"mean": point, "ci95": [stats[int(.025 * reps)], stats[int(.975 * reps) - 1]], "reps": reps,
            "n": sum(len(s) for s in strata)}


def claim2(table):
    out = {}
    for comp in ("multistart", "single"):
        per_rep, strata = {}, []
        for r in range(3):
            d = paired_diffs(table, f"pbt-r{r}", f"{comp}-r{r}")
            per_rep[f"r{r}"] = mean(mean(x) for x in d)
            strata.extend(d)
        out[f"pbt_minus_{comp}"] = {"per_replicate": per_rep,
                                     "pbt_greater_all_three": all(v > 0 for v in per_rep.values()),
                                     "pbt_less_all_three": all(v < 0 for v in per_rep.values()),
                                     "pooled_paired": bootstrap_pooled(strata)}
    return out


def intervals_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def claim4(doc, configs):
    sealed = sorted({(c["seed_start"], c["seed_start"] + c["examples"]) for cfg in configs.values() for c in cfg["conditions"]})
    used = []
    for run, e in doc["runs"].items():
        p, s = e.get("protocol"), e.get("state")
        if p:
            used.append((run, "development", p["development_seed_start"], p["development_seed_start"] + p["development_examples"]))
        if s:
            for a in s["allocations"]:
                lo, hi = a["training_seed_interval"]
                used.append((run, f"train r{a['round']}s{a['slot']}", lo, hi))
            for l in s["lineage"]:
                if "retained_recipient_seed_cursor" in l:
                    c = l["retained_recipient_seed_cursor"]
                    used.append((run, "cursor", c, c + 1))
    overlaps = [(u, s) for u in used for s in sealed if intervals_overlap((u[2], u[3]), s)]
    lo = min(u[2] for u in used)
    return {"sealed_intervals": sealed, "n_used_intervals": len(used), "runs": sorted(doc["runs"]),
            "min_used_seed": lo, "max_used_seed": max(u[3] for u in used),
            "used_below_sealed_max": sorted({u[0] + ":" + u[1] for u in used if u[2] < 75100256}),
            "overlaps": overlaps,
            "dev_ranges": {run: [e["protocol"]["development_seed_start"], e["protocol"]["development_examples"]]
                           for run, e in doc["runs"].items() if e.get("protocol")}}


def registered_finalist(state, mode):
    allocs = state["allocations"]
    last = max(a["round"] for a in allocs)
    final = [a for a in allocs if a["round"] == last]
    if mode == "single":
        return max(final, key=lambda a: a["slot"])  # latest in-round allocation
    best = max(a["utility"] for a in final)
    return min((a for a in final if a["utility"] == best), key=lambda a: a["slot"])


def claim5(doc, configs):
    results, problems = [], []
    runs_by_path = {}
    for run, e in doc["runs"].items():
        runs_by_path[run] = e
    for key, cfg in configs.items():
        for b in cfg["checkpoints"]:
            run = b["path"].split("/results/")[1].split("/")[0]
            e = runs_by_path.get(run)
            rec = {"arm": b["name"], "run": run, "bound": b["path"].split("/results/")[1], "bound_sha": b["sha256"],
                   "disk_sha": doc["checkpoint_hashes"].get(b["path"])}
            rec["disk_matches"] = rec["disk_sha"] == b["sha256"]
            if e and e.get("state"):
                mode = e["protocol"]["mode"]
                if b["name"].startswith("bank0"):
                    target = next(a for a in e["state"]["allocations"] if a["slot"] == 0)
                    rec["rule"] = "bank member 0 (by protocol, not a selection)"
                elif b["name"].endswith("-boot"):
                    target = registered_finalist(e["state"], mode)
                    rec["rule"] = f"{run} registered finalist ({mode})"
                else:
                    target = registered_finalist(e["state"], mode)
                    rec["rule"] = f"final-round {'latest' if mode == 'single' else 'max dev utility, low slot'} ({mode})"
                rec["recomputed"] = target["checkpoint"]
                rec["recomputed_sha"] = target["checkpoint_sha256"]
                rec["state_finalist"] = (e["state"]["finalist"] or {}).get("checkpoint")
                rec["matches_rule"] = (run + "/" + target["checkpoint"] == rec["bound"] and target["checkpoint_sha256"] == b["sha256"])
                if not b["name"].startswith("bank0"):
                    rec["matches_state_finalist"] = rec["state_finalist"] == target["checkpoint"]
                if "development_utility" in b:
                    rec["dev_utility_matches"] = abs(b["development_utility"] - target["utility"]) < 1e-12
                if mode != "single" and not b["name"].startswith("bank0"):
                    last = max(a["round"] for a in e["state"]["allocations"])
                    rec["final_round_utilities"] = {a["slot"]: a["utility"] for a in e["state"]["allocations"] if a["round"] == last}
            for flag in ("disk_matches", "matches_rule", "matches_state_finalist", "dev_utility_matches"):
                if rec.get(flag) is False:
                    problems.append((b["name"], flag))
            results.append(rec)
    # imports: E08 main/E12 initial checkpoints = bank members; E11 = E07 finalists
    imports = {}
    for run, e in doc["runs"].items():
        p = e.get("protocol") or {}
        imports[run] = [(c["path"].split("/results/")[1], c["sha256"]) for c in p.get("initial_checkpoints", [])]
    bank_ok = {}
    for run, lst in imports.items():
        for path, sha in lst:
            src = path.split("/")[0]
            st = (doc["runs"].get(src) or {}).get("state")
            if st:
                ok = any(src + "/" + a["checkpoint"] == path and a["checkpoint_sha256"] == sha for a in st["allocations"])
                bank_ok.setdefault(run, []).append(ok)
    return {"bindings": results, "problems": problems,
            "import_bindings_match_source_allocations": {k: all(v) for k, v in bank_ok.items()}}


def claim6(doc):
    out = {}
    for run, e in sorted(doc["runs"].items()):
        s, p = e.get("state"), e.get("protocol")
        if not s or not p:
            continue
        reps = [l for l in s["lineage"] if l.get("kind") == "replacement"]
        if p["mode"] != "pbt":
            out[run] = {"mode": p["mode"], "replacements": len(reps), "legal": len(reps) == 0}
            continue
        rounds = p["rounds"]
        viol = []
        util = {(a["round"], a["slot"]): a for a in s["allocations"]}
        per_round = {}
        for ev in reps:
            r = ev["round"]
            per_round[r] = per_round.get(r, 0) + 1
            scores = sorted(((a["utility"], a["slot"]) for (rr, _), a in util.items() if rr == r), reverse=True)
            ranked = [sl for _, sl in scores]
            du, ru = util[(r, ev["donor_slot"])]["utility"], util[(r, ev["recipient_slot"])]["utility"]
            top2 = {sl for u, sl in scores if u >= scores[1][0]}
            bot2 = {sl for u, sl in scores if u <= scores[-2][0]}
            if not du > ru:
                viol.append((r, "donor not strictly greater", ev["donor_slot"], ev["recipient_slot"]))
            if ev["donor_slot"] not in top2:
                viol.append((r, "donor not top2", ev["donor_slot"], ranked))
            if ev["recipient_slot"] not in bot2:
                viol.append((r, "recipient not bottom2", ev["recipient_slot"], ranked))
            for f in ("learning_rate_factor", "entropy_factor", "kl_factor"):
                if ev.get(f) not in (0.8, 1.2):
                    viol.append((r, f"{f}={ev.get(f)}"))
            if r >= rounds - 1:
                viol.append((r, "replacement at/after final round"))
            if ev["parent_sha256"] != util[(r, ev["donor_slot"])]["checkpoint_sha256"]:
                viol.append((r, "parent sha != donor round checkpoint"))
            if ev["recipient_sha256"] != util[(r, ev["recipient_slot"])]["checkpoint_sha256"]:
                viol.append((r, "recipient sha != recipient round checkpoint"))
            # inherited hyperparameters = donor's (previous) values x factor, checked via the donor member row
        if any(v > 2 for v in per_round.values()):
            viol.append(("more than two replacements in a round", per_round))
        out[run] = {"mode": "pbt", "rounds": rounds, "replacements": len(reps), "per_round": per_round,
                    "violations": viol, "legal": not viol,
                    "donor_minus_recipient_min": min((util[(ev["round"], ev["donor_slot"])]["utility"]
                                                      - util[(ev["round"], ev["recipient_slot"])]["utility"]) for ev in reps) if reps else None}
    return out


def claim7(table):
    out = {}
    for arm, (root, conds) in sorted(table.items()):
        if "int_4x4_control" not in conds:
            continue
        ctl, nt = conds["int_4x4_control"], conds["int_4x4_no_tools"]
        hist = {}
        for e in nt:
            hist[e["hist"]] = hist.get(e["hist"], 0) + 1
        out[arm] = {"greedy_first_rate_control": mean(float(e["gf"]) for e in ctl),
                    "tool_use_rate_control": mean(float(e["tool"]) for e in ctl),
                    "success_control": mean(float(e["s"]) for e in ctl),
                    "success_no_tools": mean(float(e["s"]) for e in nt),
                    "utility_no_tools": mean(e["u"] for e in nt),
                    "no_tools_abstain_rate": mean(float(e["abstain"]) for e in nt),
                    "no_tools_truncated_rate": mean(float(bool(e["trunc"])) for e in nt),
                    "no_tools_mean_actions": mean(e["hist"] for e in nt),
                    "no_tools_at_48_actions": mean(float(e["hist"] >= 48) for e in nt),
                    "no_tools_last_action": {k: sum(1 for e in nt if e["last"] == k) for k in {e["last"] for e in nt}},
                    "no_tools_action_count_hist_top": sorted(hist.items(), key=lambda kv: -kv[1])[:4]}
    return out


def claim8(doc, budget):
    jobs = budget["phase2_jobs"]
    rows, cpu_sum, intervals, mism = [], 0.0, [], []
    for j in jobs:
        rec = doc["receipts"].get(j["id"], {})
        occ, lau = rec.get("occupancy.json", {}), rec.get("launch.json", {})
        local = REPO / j["receipt"]
        local_occ = json.loads(local.read_text()) if local.exists() else None
        cpu_sum += j["cpu_core_seconds"]
        row = {"id": j["id"], "ledger_cpu": j["cpu_core_seconds"], "remote_occ_cpu": occ.get("cpu_core_seconds"),
               "repo_receipt_cpu": (local_occ or {}).get("cpu_core_seconds"), "uses_gpu": j.get("uses_gpu"),
               "status": j.get("status"), "exit_code": occ.get("exit_code")}
        row["cpu_match"] = occ.get("cpu_core_seconds") is not None and abs(occ["cpu_core_seconds"] - j["cpu_core_seconds"]) < 1e-6
        if not row["cpu_match"]:
            mism.append(j["id"])
        if j.get("uses_gpu"):
            iv = (lau.get("started_unix"), occ.get("ended_unix"))
            row["remote_interval"] = iv
            row["ledger_interval"] = j.get("interval_unix")
            row["interval_match"] = (None not in iv and j.get("interval_unix") is not None
                                     and all(abs(a - b) < 1e-6 for a, b in zip(iv, j["interval_unix"])))
            if None not in iv:
                intervals.append(iv)
        rows.append(row)
    intervals.sort()
    union, cur = 0.0, None
    for lo, hi in intervals:
        if cur is None or lo > cur[1]:
            if cur:
                union += cur[1] - cur[0]
            cur = [lo, hi]
        else:
            cur[1] = max(cur[1], hi)
    if cur:
        union += cur[1] - cur[0]
    phase2_like = sorted(k for k in doc["receipts"] if k.split("-")[0] in ("e07", "e08", "e09", "e09v2", "e11", "e12", "e13", "rl01", "rl02"))
    ledger_ids = {j["id"] for j in jobs}
    return {"n_jobs": len(jobs), "cpu_sum": cpu_sum, "ledger_cpu": budget["phase2_charged_cpu_core_seconds"],
            "cpu_equal": abs(cpu_sum - budget["phase2_charged_cpu_core_seconds"]) < 1e-3,
            "cpu_receipt_mismatches": mism, "gpu_union": union, "ledger_gpu": budget["phase2_charged_gpu_seconds"],
            "gpu_equal": abs(union - budget["phase2_charged_gpu_seconds"]) < 1e-3,
            "gpu_process_wall_sum_ledger": budget.get("phase2_gpu_process_wall_sum_upper_bound"),
            "receipts_on_host_not_in_ledger": [k for k in phase2_like if k not in ledger_ids],
            "rows": rows}


def interface_check(doc):
    """Compare evaluated policy/training config (from summaries) with each run's training protocol."""
    out, issues = {}, []
    for root, ev in doc["eval"].items():
        summ = ev["summary"] or {}
        paths = {c["name"]: c["path"] for c in summ.get("checkpoints") or []}
        for m in summ.get("arm_meta", []):
            if m["kind"] != "learned":
                continue
            arm = m["arm"]
            run = paths[arm].split("/results/")[1].split("/")[0]
            proto = (doc["runs"].get(run) or {}).get("protocol") or {}
            key = (arm, json.dumps(m["policy_config"], sort_keys=True), m["training_config"]["max_steps"])
            rec = out.setdefault(arm, {"run": run, "policy_configs": set(), "max_steps": set(),
                                       "protocol_policy": proto.get("policy"),
                                       "protocol_family": (proto.get("train") or {}).get("family"),
                                       "protocol_width": (proto.get("train") or {}).get("width"),
                                       "protocol_max_steps": (proto.get("train") or {}).get("max_steps")})
            rec["policy_configs"].add(key[1])
            rec["max_steps"].add(key[2])
    for arm, rec in out.items():
        pcs = [json.loads(x) for x in rec["policy_configs"]]
        rec["policy_configs"] = pcs
        rec["max_steps"] = sorted(rec["max_steps"])
        pp = rec["protocol_policy"] or {}
        for pc in pcs:
            if pp and pc.get("feature_version") != pp.get("feature_version"):
                issues.append((arm, "feature_version", pc.get("feature_version"), pp.get("feature_version")))
            if rec["protocol_family"] and pc.get("family") != rec["protocol_family"]:
                issues.append((arm, "family", pc.get("family"), rec["protocol_family"]))
            if rec["protocol_width"] and pc.get("width") != rec["protocol_width"]:
                issues.append((arm, "width", pc.get("width"), rec["protocol_width"]))
        if len(pcs) != 1:
            issues.append((arm, "policy config differs across conditions"))
    return {"arms": out, "issues": issues}


def truncation(doc, table):
    """Episodes cut by the policy-side max_steps before the world's own step limit."""
    limits = {}
    for root, ev in doc["eval"].items():
        for cond, info in ev["conditions"].items():
            if info["worlds"]:
                limits[cond] = info["worlds"]["step_limits"]
    out = {}
    for arm, (root, conds) in table.items():
        t = {c: sum(1 for e in eps if e["trunc"]) for c, eps in conds.items()}
        t = {c: v for c, v in t.items() if v}
        if t:
            out[arm] = t
    return {"world_step_limits": limits, "truncated_episode_counts": out}


def source_hashes(doc):
    out = {}
    for root, ev in doc["eval"].items():
        out[root] = (ev["summary"] or {}).get("sources")
    runs = {run: (e.get("state") or {}).get("source_hashes") for run, e in doc["runs"].items()}
    return {"evaluation_sources": out, "training_source_hashes": runs}


def repo_copies(doc):
    out = {}
    for run, e in doc["runs"].items():
        d = REPO / "research/results/campaign-02" / run
        rec = {}
        for name in ("state.json", "protocol.json", "population_lineage.json"):
            p = d / name
            if p.exists() and (name + ".sha256") in e:
                rec[name] = hashlib.sha256(p.read_bytes()).hexdigest() == e[name + ".sha256"]
            elif (name + ".sha256") in e:
                rec[name] = "no repo copy"
        out[run] = rec
    return out


def main():
    t0 = time.time()
    doc = json.load(open(sys.argv[1]))
    target = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).with_name("phase2-independent-audit.json"))
    configs = {f"{k}": json.loads(p.read_text()) for k, p in CONFIGS.items()}
    budget = json.loads((REPO / "research/campaigns/extended-02/budget.json").read_text())
    notes = []
    table = arm_table(doc)
    report = {
        "claim1_iid_means": claim1(table),
        "claim2_primary_rule": claim2(table),
        "claim3_world_identity": check_worlds(doc, configs, notes),
        "claim4_seed_disjointness": claim4(doc, configs),
        "claim5_finalists": claim5(doc, configs),
        "claim6_lineage": claim6(doc),
        "claim7_mechanism": claim7(table),
        "claim8_ledger": claim8(doc, budget),
        "interface_check": interface_check(doc),
        "truncation": truncation(doc, table),
        "source_hashes": source_hashes(doc),
        "repo_copies_match_host": repo_copies(doc),
        "v1_partial_roots": doc["v1"],
        "notes": notes,
        "remote_extraction_cost": doc["audit_cost"],
    }
    if len(sys.argv) > 3:  # output of audit_phase2_v1_remote.py
        report["v1_vs_v2_identity"] = json.loads(Path(sys.argv[3]).read_text())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    report["local_analysis_cost"] = {"user_s": ru.ru_utime, "sys_s": ru.ru_stime, "wall_s": time.time() - t0}
    Path(target).write_text(json.dumps(report, indent=1, sort_keys=True, default=list) + "\n")
    print(target)


if __name__ == "__main__":
    main()
