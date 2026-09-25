#!/usr/bin/env python3
"""Phase-3b follow-up probes (stdlib, read-only; prints JSON). Written for this audit.

    ssh gb10-direct 'python3 -' < phase3b_detail_remote.py > phase3b-detail.json

1. E20-r2 failures: which completion kinds were rejected, and with which reasons.
2. E22 failures (6): action-kind histogram, to characterise the non-perseveration failures.
3. E17 r0 vs r1: per-episode identity of outcomes/actions on e19 and e17 sealed sets.
4. E20/E22 learned arms on same_* conditions: any use_return on a record the actor did not create.
5. E19: do E17 successes in distractor worlds coincide with an accidentally goal-valid same-type distractor?
"""
import collections
import gzip
import json
import os
import resource
import sys

R = os.path.expanduser("~/topoformer-campaign02/results")
COMPLETION = {"commit_pending", "commit_assignment", "use_return", "deliver", "move"}
BAD = {"rejected", "invalid_input"}
PRIM = {"select": "constrained_subset", "route": "shortest_path", "assign": "csp"}


def head(line):
    i = line.find(', "timing": ')
    return json.loads(line[:i] + "}") if i > 0 else json.loads(line)


def load(root, cond, arm):
    with gzip.open(os.path.join(R, root, cond, arm + ".jsonl.gz"), "rt") as fh:
        return [head(l) for l in fh]


def conds(root):
    return sorted(c for c in os.listdir(os.path.join(R, root)) if os.path.isdir(os.path.join(R, root, c)))


def own_handles(h):
    return {x["feedback"].get("return") for x in h if x["action"]["kind"] in ("call", "inspect")}


out = {}
# 1
kinds, reasons, per_ep_commit = collections.Counter(), collections.Counter(), []
for c in conds("e20-sealed"):
    if c.startswith("hard"):
        continue
    for d in load("e20-sealed", c, "e20rl-r2"):
        o = d["outcome"]
        if o["verified_success"]:
            continue
        n = 0
        for x in o["history"]:
            if x["action"]["kind"] in COMPLETION and x["feedback"].get("status") in BAD:
                kinds[x["action"]["kind"]] += 1
                reasons[(x["action"]["kind"], x["feedback"].get("reason"))] += 1
                n += x["action"]["kind"] == "commit_pending"
        per_ep_commit.append(n)
out["e20_r2_failure_rejected_kinds"] = dict(kinds)
out["e20_r2_failure_rejected_reasons"] = {f"{k}|{r}": v for (k, r), v in reasons.items()}
out["e20_r2_commit_pending_rejections_per_failure"] = [min(per_ep_commit), max(per_ep_commit), len(per_ep_commit)]

# 2
e22f = []
for c in conds("e22-sealed"):
    if c.startswith("hard"):
        continue
    for r in range(3):
        for d in load("e22-sealed", c, f"e22rl-r{r}"):
            o = d["outcome"]
            if not o["verified_success"]:
                h = o["history"]
                e22f.append({"cond": c, "arm": f"e22rl-r{r}", "seed": d["seed"],
                             "kinds": dict(collections.Counter(x["action"]["kind"] for x in h)),
                             "statuses": dict(collections.Counter(x["feedback"].get("status") for x in h)),
                             "last8": [x["action"]["kind"] for x in h[-8:]],
                             "stages_completed": o.get("stages_completed")})
out["e22_failures"] = e22f

# 3
ident = {}
for root in ("e19-sealed", "e17-sealed"):
    same_outcome = same_actions = n = 0
    for c in conds(root):
        a = load(root, c, "e17rl-r0")
        b = {d["seed"]: d for d in load(root, c, "e17rl-r1")}
        for d in a:
            e = b[d["seed"]]
            n += 1
            same_outcome += d["outcome"]["verified_success"] == e["outcome"]["verified_success"]
            same_actions += [x["action"] for x in d["outcome"]["history"]] == [x["action"] for x in e["outcome"]["history"]]
    ident[root] = {"episodes": n, "same_outcome": same_outcome, "identical_action_sequences": same_actions}
out["e17_r0_vs_r1"] = ident

# 4
foreign = {}
for root, arms in (("e20-sealed", ["e20rl-r0", "e20rl-r1", "e20rl-r2"]), ("e22-sealed", ["e22rl-r0", "e22rl-r1", "e22rl-r2"])):
    for a in arms:
        uses = retrieves = 0
        for c in conds(root):
            if not c.startswith("same_"):
                continue
            for d in load(root, c, a):
                h = d["outcome"]["history"]
                own = own_handles(h)
                for x in h:
                    k = x["action"]["kind"]
                    if k in ("use_return", "retrieve") and x["action"]["arguments"].get("handle") not in own:
                        uses += k == "use_return"
                        retrieves += k == "retrieve"
        foreign[a] = {"foreign_use_return": uses, "foreign_retrieve": retrieves}
out["same_type_foreign_record_actions"] = foreign

# 5
def valid_for_goal(spec, prim, payload, snap):
    if prim == "constrained_subset":
        hs = [snap["problem"]["handles"][i] for i in payload[0]]
        inv = {x["handle"]: x for x in spec["items"]}
        ch = [inv[h] for h in hs]
        return (sorted(x["category"] for x in ch) == sorted(spec["categories"])
                and sum(x["weight"] for x in ch) <= spec["capacity"] and sum(x["price"] for x in ch) <= spec["funds"]
                and not any(a in hs and b in hs for a, b in spec["incompatible"]) and len(set(hs)) == len(hs))
    if prim == "csp":
        v = list(payload)
        return (len(v) == len(spec["domains"]) and all(x in dm for x, dm in zip(v, spec["domains"]))
                and not any(v[i] == x and v[j] == y for i, x, j, y in spec["forbidden"]))
    return False


coincide = collections.Counter()
for c in conds("e19-sealed"):
    W = {}
    with gzip.open(os.path.join(R, "e19-sealed", c, "worlds.jsonl.gz"), "rt") as fh:
        for l in fh:
            w = json.loads(l)
            W[w["seed"]] = w["spec"]
    for d in load("e19-sealed", c, "e17rl-r0"):
        sp = W[d["seed"]]
        if not sp["distractors"]:
            continue
        any_valid = any(valid_for_goal(sp, p, pl, sn) for p, pl, sn in sp["distractors"])
        coincide[(bool(d["outcome"]["verified_success"]), any_valid)] += 1
out["e19_r0_success_vs_any_goal_valid_distractor"] = {f"success={a},valid_distractor={b}": v for (a, b), v in coincide.items()}

ru = resource.getrusage(resource.RUSAGE_SELF)
out["cpu_seconds_remote"] = ru.ru_utime + ru.ru_stime
json.dump(out, sys.stdout, indent=1)
