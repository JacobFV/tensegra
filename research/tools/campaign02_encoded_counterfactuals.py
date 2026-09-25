"""Encoded-input counterfactual audit (CPU, no torch, no solver process).

Builds minimal paired public states and compares the exact vectors the policy
consumes (m1, m2, m3). Question 1: are an own current-draft return and a
same-type foreign/stale return distinguishable? Question 2: are rejection
contexts (reason; rejection followed by an intervening action) distinguishable?
"""
import json, sys
from dataclasses import replace
from functools import partial
sys.path.insert(0, "src")
from topoformer.campaign02_modular import (ModularWorkshop, generate_modular, action_catalog, modular_executor,
    encode_action, encode_action_m2, encode_action_m3, encode_observation, encode_observation_m3)
from topoformer.campaign02_protocol import execute
from topoformer.campaign02_world import Action

EXEC = partial(modular_executor, execute_call=execute)
ENC = {"m1": (encode_observation, encode_action), "m2": (encode_observation, encode_action_m2),
       "m3": (encode_observation_m3, encode_action_m3)}
out = {}

# Q1a: own current-draft csp return vs same-type prior (foreign) csp return, same status/validity/retrieved flags.
spec = next(s for s in (generate_modular(i, stages=("assign",), same_type_distractors=2) for i in range(200))
            if any(p == "csp" for p, _, _ in s.distractors))
env = ModularWorkshop(spec, executor=EXEC, address_seed=1)
o = env.observe(); prior = next(r["handle"] for r in o.records if r["primitive"] == "csp")
for a in (Action("inspect", {"target": "roster"}), Action("start_assign", {"handle": "problem_0"}),
          Action("add_constraint", {"problem": "problem_0", "constraint": "forbidden"}),
          Action("call", {"problem": "problem_0", "budget": 1024})):
    o = env.step(a)
own = o.feedback["return"]
for h in (own, prior):
    o = env.step(Action("retrieve", {"handle": h}))
q1a = {}
for v, (_, ea) in ENC.items():
    x = ea(o, Action("use_return", {"handle": own, "as": "assign"}))
    y = ea(o, Action("use_return", {"handle": prior, "as": "assign"}))
    q1a[v] = {"identical": x == y, "differing_coordinates": [i for i, (p, q) in enumerate(zip(x, y)) if p != q]}
out["own_vs_foreign_same_type_return"] = q1a

# Q1b: own STALE return (called before constraints were added: same draft handle, different snapshot)
# vs own CURRENT return (after constraints).
env = ModularWorkshop(generate_modular(3, stages=("assign",)), executor=EXEC, address_seed=2)
for a in (Action("inspect", {"target": "roster"}), Action("start_assign", {"handle": "problem_0"})):
    env.step(a)
stale = env.step(Action("call", {"problem": "problem_0", "budget": 1024})).feedback["return"]
env.step(Action("add_constraint", {"problem": "problem_0", "constraint": "forbidden"}))
cur = env.step(Action("call", {"problem": "problem_0", "budget": 1024})).feedback["return"]
for h in (stale, cur):
    o = env.step(Action("retrieve", {"handle": h}))
q1b = {}
for v, (_, ea) in ENC.items():
    x = ea(o, Action("use_return", {"handle": cur, "as": "assign"}))
    y = ea(o, Action("use_return", {"handle": stale, "as": "assign"}))
    q1b[v] = {"identical": x == y, "differing_coordinates": [i for i, (p, q) in enumerate(zip(x, y)) if p != q]}
out["own_current_vs_own_stale_draft_return"] = q1b

# Q2: rejection contexts, matched in step count.
import itertools
def find_world():
    for seed in range(500):
        spec = generate_modular(seed, stages=("select",), categories=2, choices=3)
        c0 = [x for x in spec.items if x.category == 0]; c1 = [x for x in spec.items if x.category == 1]
        combos = {}
        for a, b in itertools.product(c0, c1):
            if frozenset((a.handle, b.handle)) in {frozenset(q) for q in spec.incompatible}:
                continue
            if a.weight + b.weight > spec.capacity and a.price + b.price <= spec.funds:
                combos.setdefault("capacity", (a.handle, b.handle))
            if a.weight + b.weight <= spec.capacity and a.price + b.price > spec.funds:
                combos.setdefault("funds", (a.handle, b.handle))
        if len(combos) == 2:
            return seed, combos
seed, combos = find_world()
def run_path(pair, tail):
    env = ModularWorkshop(generate_modular(seed, stages=("select",), categories=2, choices=3), executor=EXEC, address_seed=3)
    for r in env.observe().item_inventory:
        env.step(Action("inspect", {"target": r["handle"]}))
    for h in pair:
        env.step(Action("choose_item", {"item": h}))
    o = None
    for a in tail:
        o = env.step(a)
    return o
C, T = Action("commit_pending"), Action("think")
oc, of = run_path(combos["capacity"], (C,)), run_path(combos["funds"], (C,))
later, never = run_path(combos["capacity"], (C, T)), run_path(combos["capacity"], (T, T))
q2 = {"world_seed": seed, "reasons_seen": [oc.feedback.get("reason"), of.feedback.get("reason")],
      "steps_matched": [later.remaining_steps == never.remaining_steps]}
for v, (eo, ea) in ENC.items():
    q2[f"{v}_capacity_vs_funds_identical"] = (eo(oc) == eo(of) and ea(oc, C) == ea(of, C))
    q2[f"{v}_rejected_then_other_action_vs_never_committed_identical"] = (eo(later) == eo(never) and ea(later, C) == ea(never, C))
out["rejection_context"] = q2
print(json.dumps(out, indent=1))
json.dump(out, open("research/campaigns/extended-02/diagnostics/encoded-counterfactuals.json", "w"), indent=1)
