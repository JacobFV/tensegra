"""Phase-3 independent audit: READ-ONLY remote extractor (python3 stdlib).

Run as: ssh gb10-direct 'python3 - ~/topoformer-campaign02/results' < audit_phase3_remote.py > raw.json
Reads sealed episode rows, worlds files, run states/protocols, development
prediction seeds and occupancy receipts. Writes nothing; prints one JSON document.
Written independently of research/tools/campaign02_e09_analysis.py and earlier audits.
"""
import glob, gzip, json, os, sys, time, resource

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/topoformer-campaign02/results")
SEALED = ["e09v2-sealed-r0", "e09v2-sealed-r1", "e09v2-sealed-r2", "e09v2-sealed-e15",
          "e16-sealed", "e16-sealed-distractors", "e17-sealed", "e18-sealed"]
t0 = time.time()
out = {"root": ROOT, "sealed": {}, "worlds": {}, "runs": {}}


def handle_primitives(trace):
    m = {}
    for step in trace or []:
        obs = step.get("observation") or {}
        for rec in obs.get("records") or []:
            if rec.get("kind") == "computation":
                m[rec["handle"]] = rec.get("primitive")
    return m


for root in SEALED:
    rdir = os.path.join(ROOT, root)
    out["sealed"][root] = {}
    for cdir in sorted(glob.glob(os.path.join(rdir, "*/"))):
        cond = os.path.basename(cdir.rstrip("/"))
        wpath = os.path.join(cdir, "worlds.jsonl.gz")
        if os.path.exists(wpath):
            ws = []
            for line in gzip.open(wpath, "rt"):
                w = json.loads(line)
                spec = w.get("spec") or {}
                d = spec.get("distractors")
                ws.append([w["seed"], w["spec_hash"], len(d) if isinstance(d, list) else None,
                           spec.get("step_limit"), spec.get("stages")])
            out["worlds"]["%s/%s" % (root, cond)] = ws
        arms = {}
        for f in sorted(glob.glob(os.path.join(cdir, "*.jsonl.gz"))):
            arm = os.path.basename(f)[: -len(".jsonl.gz")]
            if arm == "worlds":
                continue
            rows = []
            for line in gzip.open(f, "rt"):
                r = json.loads(line)
                o = r["outcome"]
                trace = r.get("trace")
                priors = None
                if trace:
                    recs = (trace[0].get("observation") or {}).get("records") or []
                    priors = sum(1 for x in recs if x.get("prior"))
                hist = o.get("history") or []
                prim = handle_primitives(trace) if trace else {}
                mism_any = mism_sel = mism_sel_csp = 0
                for h in hist:
                    a = h.get("action") or {}
                    fb = h.get("feedback") or {}
                    if a.get("kind") == "use_return" and fb.get("reason") == "return type mismatch":
                        mism_any += 1
                        args = a.get("arguments") or {}
                        if args.get("as") == "select":
                            mism_sel += 1
                            if prim.get(args.get("handle")) == "csp":
                                mism_sel_csp += 1
                rows.append([r["seed"], r["spec_hash"], 1 if o.get("verified_success") else 0,
                             o.get("utility"), o.get("steps"), priors, o.get("stages_completed"),
                             mism_any, mism_sel, mism_sel_csp, len(hist), r.get("truncated")])
            arms[arm] = rows
        out["sealed"][root][cond] = arms

# Non-success feedback histogram for failed e17rl-r2 episodes (localization context)
kinds = {}
for cond in ("iid_S-R", "iid_S", "iid_S-A"):
    p = os.path.join(ROOT, "e17-sealed", cond, "e17rl-r2.jsonl.gz")
    if not os.path.exists(p):
        continue
    for line in gzip.open(p, "rt"):
        r = json.loads(line)
        if r["outcome"].get("verified_success"):
            continue
        for h in r["outcome"].get("history") or []:
            fb = h.get("feedback") or {}
            if fb.get("status") != "success":
                k = "%s|%s|%s|%s" % (cond, h["action"].get("kind"), fb.get("status"), fb.get("reason"))
                kinds[k] = kinds.get(k, 0) + 1
out["e17_r2_failed_nonsuccess_feedback"] = kinds

for sp in sorted(glob.glob(os.path.join(ROOT, "*", "state.json"))):
    run = os.path.basename(os.path.dirname(sp))
    s = json.load(open(sp))
    pp = os.path.join(os.path.dirname(sp), "protocol.json")
    p = json.load(open(pp)) if os.path.exists(pp) else {}
    wm = p.get("world_mix")
    if isinstance(wm, str):
        wm = json.loads(wm)
    dev_seeds = []
    for dp in glob.glob(os.path.join(os.path.dirname(sp), "development", "*.jsonl.gz")):
        for line in gzip.open(dp, "rt"):
            r = json.loads(line)
            if "seed" in r:
                dev_seeds.append(r["seed"])
    lin = [l for l in s.get("lineage", []) if l.get("kind") != "initialization"]
    fin = s.get("finalist") or {}
    out["runs"][run] = {
        "mode": p.get("mode"), "halving_survivors": p.get("halving_survivors"),
        "niche_protection": p.get("niche_protection"), "train_max_steps": (p.get("train") or {}).get("max_steps"),
        "training_seed_start": p.get("training_seed_start"), "training_seed_stride": p.get("training_seed_stride"),
        "development_seed_start": p.get("development_seed_start"), "development_examples": p.get("development_examples"),
        "development_world_mix": p.get("development_world_mix"),
        "world_mix": wm, "teacher": p.get("teacher"), "methods": p.get("methods"),
        "dev_seed_min": min(dev_seeds) if dev_seeds else None, "dev_seed_max": max(dev_seeds) if dev_seeds else None,
        "dev_seed_count": len(dev_seeds),
        "allocations": [{k: a.get(k) for k in ("round", "slot", "member", "utility", "success",
                         "behavior_greedy_first", "cumulative_slot_updates", "updates", "training_seed_interval")}
                        for a in s.get("allocations", [])],
        "lineage_events": lin,
        "finalist": {k: fin.get(k) for k in ("member", "round", "slot", "utility", "cumulative_slot_updates", "checkpoint")},
        "survivors": s.get("survivors"), "status": s.get("status"),
    }

occ = {}
for op in glob.glob(os.path.join(ROOT, "*-process", "occupancy.json")):
    occ[os.path.basename(os.path.dirname(op))] = json.load(open(op))
out["occupancy"] = occ
ru = resource.getrusage(resource.RUSAGE_SELF)
out["audit_cpu_seconds"] = ru.ru_utime + ru.ru_stime
out["audit_wall_seconds"] = time.time() - t0
json.dump(out, sys.stdout)
