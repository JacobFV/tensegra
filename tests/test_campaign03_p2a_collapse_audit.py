"""extended-03 P2a Part D collapse audit (research/tools/campaign03_p2a_collapse_audit.py), CPU."""
from dataclasses import asdict
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import (TrainConfig, actor_critic_objective, batched_on_policy,
                                          independent_address_seed, public_frame)
from tensegra.campaign02_world import Action
from tensegra.campaign03_depworld import DepReference, DepWorkshop, depworld_executor, generate_depworld

ROOT = Path(__file__).resolve().parents[1]
EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_070_000_000  # development seeds only


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load("campaign03_p2a_collapse_audit")
C = _load("campaign03_p1_configs")


def _model(seed=0, width=16):
    torch.manual_seed(seed)
    _, obs, cand = public_frame(DepWorkshop(generate_depworld(DEV)).observe(), "d1")
    return CandidatePolicy(PolicyConfig(len(obs), len(cand[0]), width=width, feature_version="d1")).eval()


def _env(seed, **kwargs):
    return DepWorkshop(generate_depworld(seed, **kwargs), executor=EXEC,
                       address_seed=independent_address_seed(seed, "test-p2a"))


# --- probe worlds -----------------------------------------------------------------------------

def test_probe_worlds_are_fresh_interleaved_iid_group():
    worlds = D.probe_worlds()
    sealed = dict(C.SEALED)
    assert len(worlds) == 64 and len({s for _, s, _ in worlds}) == 64
    assert [c for c, _, _ in worlds[:4]] == ["iid_f0", "iid_f2", "iid_f0", "iid_f2"]
    assert all(kwargs == sealed[c] for c, _, kwargs in worlds)
    assert {s for c, s, _ in worlds if c == "iid_f0"} == set(range(1_990_000_000, 1_990_000_032))
    assert {s for c, s, _ in worlds if c == "iid_f2"} == set(range(1_990_100_000, 1_990_100_032))
    # Every 8-world batch holds 4 worlds of each condition.
    for b in range(0, 64, 8):
        assert sum(c == "iid_f0" for c, _, _ in worlds[b:b + 8]) == 4
    # Disjoint from every P1 seed family (sealed 110M+, boot 3.0e9+, RL 3.4e9+, dev 3.9e9+).
    for _, s, _ in worlds:
        assert not 110_000_000 <= s < 111_000_000 and s < 3_000_000_000


# --- loop definitions -------------------------------------------------------------------------

def _s(key, pub, dec, status="success", kind="add_constraint", call=False, new=False):
    return {"kind": kind, "key": key, "status": status, "accepted": status not in D.REJECTED,
            "success": status == "success", "pub_before": pub[0], "pub_after": pub[1],
            "dec_before": dec[0], "dec_after": dec[1], "call": call, "new_inspection": new}


def test_idempotent_repeat_needs_same_key_as_last_accepted_and_unchanged_state():
    steps = [_s("k", ("p0", "p1"), ("d0", "d1")),      # first: changes state
             _s("k", ("p1", "p1"), ("d1", "d1")),      # repeat, unchanged -> idempotent (+ cycle)
             _s("k", ("p1", "p2"), ("d1", "d2")),      # same key but state changed -> not idempotent
             _s("x", ("p2", "p2"), ("d2", "d2"), status="rejected"),
             _s("k", ("p2", "p2"), ("d2", "d2"))]      # last *accepted* key is k -> idempotent
    flags = D.loop_flags(steps)
    assert [f["idempotent"] for f in flags] == [False, True, False, False, True]
    assert flags[1]["cycle"] and not flags[3]["cycle"]  # rejected actions are never cycle members
    assert [f["action_class"] for f in flags] == ["other", "loop", "other", "other", "loop"]


def test_short_cycle_alternation_window_and_breaks():
    # finish_by 8 / 9 alternation: keys differ (never idempotent) but states recur.
    alt = [_s("b8" if t % 2 == 0 else "b9", (f"p{t}", f"p{t+1}"), ("A" if t % 2 == 0 else "B", "B" if t % 2 == 0 else "A"))
           for t in range(6)]
    flags = D.loop_flags(alt)
    assert not any(f["idempotent"] for f in flags)
    assert [f["cycle"] for f in flags] == [False, True, True, True, True, True]
    # A solver call or a new inspection between the visits breaks the cycle.
    broken = [alt[0], _s("c", ("p1", "p2"), ("B", "B2"), kind="call", call=True), _s("b8", ("p2", "p3"), ("B2", "A"))]
    assert not D.loop_flags(broken)[2]["cycle"]
    inspected = [alt[0], _s("i", ("p1", "p2"), ("B", "B"), kind="inspect", new=True), _s("b8", ("p2", "p3"), ("B", "A"))]
    assert not D.loop_flags(inspected)[2]["cycle"] and not D.loop_flags(inspected)[1]["cycle"]
    # Only the 6 preceding states count: a return to a state 7 steps back is not a short cycle.
    chain = [_s(f"k{t}", (f"p{t}", f"p{t+1}"), (f"d{t}", f"d{t+1}")) for t in range(6)]
    chain.append(_s("back", ("p6", "p7"), ("d6", "d0")))
    assert not D.loop_flags(chain)[6]["cycle"]
    chain[-1] = _s("back", ("p6", "p7"), ("d6", "d1"))
    assert D.loop_flags(chain)[6]["cycle"]


def test_productive_classes():
    steps = [_s("c", ("a", "b"), ("x", "y"), kind="call", call=True),
             _s("m", ("b", "c"), ("y", "z"), kind="commit_pending"),
             _s("n", ("c", "c"), ("z", "z"), kind="commit_pending", status="rejected"),
             _s("u", ("c", "d"), ("z", "w"), kind="use_return", status="rejected")]
    assert [f["action_class"] for f in D.loop_flags(steps)] == ["productive", "productive", "other", "productive"]


def test_recorded_real_steps_detect_retrieve_repeat_and_choice_cycle():
    env = D.Recording(_env(DEV + 1, foreign_records=2, p_event=0.0))
    o = env.observe()
    handle = o.records[0]["handle"]
    for _ in range(3):
        env.step(Action("retrieve", {"handle": handle}))
    flags = D.loop_flags(env.steps)
    assert [f["idempotent"] for f in flags] == [False, True, True]
    assert [f["cycle"] for f in flags] == [False, True, True]
    # choose_item A, B (same category), A: pending returns to an earlier state.
    env = D.Recording(_env(DEV + 2))
    o = env.observe()
    first = [r["handle"] for r in o.item_inventory if r["category"] == o.item_inventory[0]["category"]]
    for h in (first[0], first[1], first[0]):
        env.step(Action("choose_item", {"item": h}))
    flags = D.loop_flags(env.steps)
    assert [f["cycle"] for f in flags] == [False, False, True] and not any(f["idempotent"] for f in flags)
    # New inspections are progress; re-inspecting a known item is a no-op repeat.
    env = D.Recording(_env(DEV + 3))
    target = env.observe().item_inventory[0]["handle"]
    for _ in range(2):
        env.step(Action("inspect", {"target": target}))
    flags = D.loop_flags(env.steps)
    assert env.steps[0]["new_inspection"] and not env.steps[1]["new_inspection"]
    assert [f["loop"] for f in flags] == [False, True]


def test_reference_episode_has_few_loops_and_summary_fields():
    env = D.Recording(_env(DEV + 4, p_event=0.5, foreign_records=2, event_trigger="progress"))
    o, ref = env.observe(), DepReference("reuse")
    while not o.done:
        o = env.step(ref.choose(o))
    summary = D.episode_summary(env, env.evaluate())
    assert summary["success"] and not summary["cap"] and summary["steps"] == len(env.steps)
    agg = D.aggregate_episodes([summary])
    assert agg["success"] == 1 and abs(sum(agg["action_mix"].values()) - 1) < 1e-9
    assert agg["idempotent_repeat_rate"] <= .1


# --- training-identical computations ----------------------------------------------------------

def test_greedy_probe_kl_matches_training_masked_kl_and_greedy_actions():
    current, reference = _model(1), _model(2)
    seeds = [DEV + 10 + i for i in range(3)]
    envs_a = [D.Recording(_env(s, foreign_records=2)) for s in seeds]
    with torch.no_grad():
        results, _, kls = batched_on_policy(current, envs_a, max_steps=12, sample=False, reference=reference)
    envs_b = [D.Recording(_env(s, foreign_records=2)) for s in seeds]
    outcomes, probe_kls, disagree = D.greedy_probe(current, envs_b, {"ref": reference, "cur": current},
                                                   [("ref", "cur"), ("cur", "cur")], max_steps=12)
    assert [[x["key"] for x in e.steps] for e in envs_a] == [[x["key"] for x in e.steps] for e in envs_b]
    assert [r["outcome"]["utility"] for r in results] == [o["utility"] for o in outcomes]
    # Training appends per step then active episode; the probe appends per step batch in the same order.
    by_step = []
    for t in range(12):
        by_step += [float(k[t]) for k in kls if t < len(k)]
    assert probe_kls[("ref", "cur")] == pytest.approx(by_step, abs=1e-6)
    assert max(probe_kls[("cur", "cur")]) < 1e-6 and not any(disagree[("cur", "cur")])


def _terms(n_steps, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n_steps, 4, generator=g, requires_grad=True)
    values = torch.randn(n_steps, generator=g, requires_grad=True)
    rewards = (torch.randn(n_steps, generator=g) * .1).tolist()
    dist = torch.distributions.Categorical(logits=logits)
    actions = torch.zeros(n_steps, dtype=torch.long)
    logp, entropy = dist.log_prob(actions), dist.entropy()
    return logits, [(logp[t], values[t], entropy[t], rewards[t]) for t in range(n_steps)]


def test_training_advantages_match_actor_critic_objective_weights():
    episodes, leaves = [], []
    for e, n in enumerate((5, 3, 7)):
        logits, terms = _terms(n, e)
        leaves.append(logits)
        episodes.append(terms)
    config = TrainConfig(method="actor_critic", advantage_normalization=True, entropy_weight=0.0, value_weight=0.0)
    rows = D.training_advantages(episodes)
    _, parts = actor_critic_objective(episodes, config)
    logps = [t[0] for ep in episodes for t in ep]
    grads = torch.autograd.grad(parts["policy_loss"], logps, retain_graph=True)
    flat = [r for part in rows for r in part]
    # decision_mean policy loss = mean(-logp * weight): d/dlogp = -weight / N.
    assert [float(-g * len(flat)) for g in grads] == pytest.approx([r["normalized"] for r in flat], abs=1e-5)
    # Raw advantage = undiscounted return-to-go minus value; return-to-go telescopes to the episode sum.
    for ep, part in zip(episodes, rows):
        assert part[0]["return_to_go"] == pytest.approx(sum(t[3] for t in ep))
        assert all(r["raw"] == pytest.approx(r["return_to_go"] - r["value"]) for r in part)


def test_gradient_terms_are_additive_and_take_no_step():
    model = _model(3)
    reference = _model(4)
    for p in model.parameters():
        p.requires_grad_(True)
    envs = [D.Recording(_env(DEV + 20 + i, foreign_records=2)) for i in range(2)]
    torch.manual_seed(0)
    results, terms, kls = batched_on_policy(model, envs, max_steps=10, sample=True, reference=reference)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    config = TrainConfig(method="actor_critic", rollout_mode="batched", learning_rate=3e-5, entropy_weight=.003,
                         kl_weight=.3, advantage_normalization=True)
    out = D.gradient_terms(model, terms, kls, config)
    assert set(out["norms"]) == {"actor", "critic", "entropy", "kl", "total"}
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    # Critic gradient never reaches the actor head; actor/entropy/KL never reach the critic head.
    assert out["norms"]["critic"]["actor_head"] == 0
    assert all(out["norms"][t]["critic_head"] == 0 for t in ("actor", "entropy", "kl"))
    assert out["norms"]["kl"]["all"] > 0 and out["norms"]["critic"]["shared"] > 0
    assert all(v is None or -1 - 1e-6 <= v <= 1 + 1e-6 for v in out["shared_cosine"].values())
    assert {D.parameter_group(n) for n, _ in model.named_parameters()} == set(D.PARAMETER_GROUPS)


def test_critic_summary_classes():
    rows = [{"value": .1, "return_to_go": .0, "raw": -.1, "normalized": -1., "action_class": "loop"},
            {"value": .1, "return_to_go": .5, "raw": .4, "normalized": 1., "action_class": "productive"},
            {"value": .0, "return_to_go": .0, "raw": .0, "normalized": 0., "action_class": "other"}]
    out = D.critic_summary(rows)
    assert out["critic_bias"] == pytest.approx((.1 - .4) / 3)
    assert out["loop_minus_productive_normalized"] == pytest.approx(-2.)
    assert out["by_class"]["loop"]["n"] == 1


# --- tiny end-to-end audit ------------------------------------------------------------------

def test_tiny_audit_lineage_and_report(tmp_path):
    rl = D.rl_train_config(0)
    assert (rl.learning_rate, rl.entropy_weight, rl.kl_weight, rl.advantage_normalization, rl.batch_size) == (3e-5, .003, .3, True, 8)
    bindings = {}
    for i, name in enumerate(("boot", "a1", "a2", "a3")):
        model = _model(10 + i)
        path = tmp_path / f"{name}.pt"
        config = {**asdict(rl), "device": "cuda"} if name != "boot" else asdict(TrainConfig(method="supervised"))
        torch.save({"policy_config": asdict(model.config), "model": model.state_dict(), "config": config,
                    "updates": 60 * i}, path)
        bindings[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    worlds = D.probe_worlds()[:2]
    result = D.audit_lineage(0, attempts=(2, 3), worlds=worlds, samples=1, grad_batches=1,
                             make_executor=lambda: EXEC, bindings=bindings, log=lambda m: None)
    names = [r["subject"] for r in result["subjects"]]
    assert names == ["boot", "a2", "a3"]
    a3 = result["subjects"][2]
    assert a3["previous_tranche"] == "a2" and a3["gradients"]["kl_reference"] == "a2"
    assert a3["greedy"]["episodes"] == 2 and a3["sampled"]["episodes"] == 2
    assert a3["kl"]["bootstrap_states"]["boot_to_ckpt"]["states"] > 0
    assert a3["kl"]["own_greedy_states"]["prev_to_ckpt"]["mean"] >= 0
    assert result["subjects"][0]["kl"]["own_greedy_states"]["boot_to_ckpt"]["mean"] < 1e-9
    assert a3["critic"]["decisions"] == round(a3["sampled"]["mean_steps"] * a3["sampled"]["episodes"])
    assert a3["gradients"]["mean"]["norms"]["total"]["all"] > 0
    json.dumps(result)
    (tmp_path / "lineage-r0.json").write_text(json.dumps(result))
    report = D.write_report(tmp_path)
    assert "x1-r0" in report["locations"]
    text = (tmp_path / "partd.md").read_text()
    assert "## X1-r0" in text and "Summary" in text
    # A lineage stopped early (e.g. CPU cap) is reported from its per-subject partial file.
    (tmp_path / "lineage-r1.partial.json").write_text(json.dumps(
        {"lineage": 1, "partial": True, "subjects": result["subjects"][:2], "bindings": bindings}))
    report = D.write_report(tmp_path)
    assert report["partial_lineages"] == [1] and set(report["locations"]) == {"x1-r0", "x1-r1"}
    assert "Partial lineages" in (tmp_path / "partd.md").read_text()
    # A hash mismatch is refused.
    bindings["a3"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        D.audit_lineage(0, attempts=(3,), worlds=worlds[:1], samples=1, grad_batches=0,
                        make_executor=lambda: EXEC, bindings={k: bindings[k] for k in ("boot", "a2", "a3")},
                        log=lambda m: None)
