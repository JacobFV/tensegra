"""Phase-3 audit, READ-ONLY remote check: are E16/E17/E18 sealed worlds identical per
condition apart from an empty 'distractors' field? Also per-episode rejected-commit counts
for failed e17rl-r2 select-containing episodes. Prints JSON. Run via ssh stdin."""
import glob, gzip, json, os, sys
ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/topoformer-campaign02/results")
out = {"worlds": {}, "e17_r2_rejected_commit_per_failed_episode": {}}
for cdir in sorted(glob.glob(os.path.join(ROOT, "e16-sealed", "*/"))):
    c = os.path.basename(cdir.rstrip("/"))
    A = [json.loads(l) for l in gzip.open(os.path.join(cdir, "worlds.jsonl.gz"), "rt")]
    res = {}
    for other in ("e17-sealed", "e18-sealed"):
        B = [json.loads(l) for l in gzip.open(os.path.join(ROOT, other, c, "worlds.jsonl.gz"), "rt")]
        same_seed = [a["seed"] for a in A] == [b["seed"] for b in B]
        eq = nonempty = addr = 0
        for a, b in zip(A, B):
            sb = dict(b["spec"])
            d = sb.pop("distractors", None)
            if d:
                nonempty += 1
            if sb == a["spec"]:
                eq += 1
            if a.get("address_seed") == b.get("address_seed"):
                addr += 1
        res[other] = {"same_seed_order": same_seed, "spec_equal_ignoring_empty_distractors": eq,
                      "nonempty_distractors": nonempty, "address_seed_equal": addr, "n": len(A)}
    out["worlds"][c] = res
for cdir in sorted(glob.glob(os.path.join(ROOT, "e17-sealed", "*/"))):
    c = os.path.basename(cdir.rstrip("/"))
    counts = []
    for line in gzip.open(os.path.join(cdir, "e17rl-r2.jsonl.gz"), "rt"):
        r = json.loads(line)
        o = r["outcome"]
        if o.get("verified_success"):
            continue
        n = sum(1 for h in o.get("history") or [] if h["action"].get("kind") == "commit_pending"
                and (h.get("feedback") or {}).get("status") == "rejected")
        counts.append(n)
    if counts:
        s = sorted(counts)
        out["e17_r2_rejected_commit_per_failed_episode"][c] = {"failed": len(s), "median": s[len(s) // 2],
                                                              "min": s[0], "max": s[-1], "zero": s.count(0)}
json.dump(out, sys.stdout)
