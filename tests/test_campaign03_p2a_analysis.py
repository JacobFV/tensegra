"""extended-03 P2a analysis (protocol-P2a.md): no-progress metrics on synthetic histories, the
decision-state reconstruction against real depworld observations, configs, the deployment rule,
sampled evaluation and a tiny end-to-end screening on CPU."""
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEV = 2_095_000_000  # development seeds only (never screening/sealed)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load("campaign03_p2a_analysis")
C = _load("campaign03_p2a_configs")


def ev(kind, status="success", **args):
    feedback = {"status": status}
    return {"action": {"kind": kind, "arguments": args}, "feedback": feedback}


def fb(kind, feedback, **args):
    return {"action": {"kind": kind, "arguments": args}, "feedback": feedback}


# ---------------------------------------------------------------------------
# Synthetic histories
# ---------------------------------------------------------------------------

def test_alternating_finish_by_is_a_short_cycle_not_an_idempotent_repeat():
    """X1-rl-r0's loop: add_constraint finish_by 8/9 alternating, never calling the csp solver."""
    history = [fb("start_assign", {"status": "success", "problem": "p"}, handle="p")]
    for i in range(10):
        history.append(ev("add_constraint", problem="p", constraint="finish_by", bound=8 + i % 2))
    flags = A.no_progress_flags(history)
    idem = [a for a, _ in flags]
    cycle = [b for _, b in flags]
    assert not any(idem)
    # step 1 new draft, steps 2 (bound 8) and 3 (bound 9) are new states; from step 4 each returns
    # to the state two steps back.
    assert cycle == [False, False, False] + [True] * 8
    m = A.episode_no_progress({"outcome": {"history": history, "verified_success": False}}, {"step_limit": 11})
    assert m["short_cycles"] == 8 and m["idempotent_repeats"] == 0 and m["no_progress"] == 8
    assert m["steps_to_cap"] == 1 and m["no_progress_episode"] == 1


def test_repeated_retrieve_of_same_record_is_idempotent():
    """X1-rl-r2's loop: retrieving the same route record again and again after committing."""
    history = [fb("commit_assignment", {"status": "success", "assignment_id": "a1", "finish_time": 3}),
               ev("retrieve", handle="rX")] + [ev("retrieve", handle="rX") for _ in range(5)]
    flags = A.no_progress_flags(history, start=0)
    assert flags[0] == (False, False) and flags[1] == (False, False)  # first retrieve changes the retrieved set
    assert all(f == (True, True) for f in flags[2:])


def test_calls_new_inspections_rejections_and_window():
    base = [fb("start_assign", {"status": "success", "problem": "p"}, handle="p")]
    b8 = ev("add_constraint", problem="p", constraint="finish_by", bound=8)
    b9 = ev("add_constraint", problem="p", constraint="finish_by", bound=9)
    call = fb("call", {"status": "success", "return": "r1"}, problem="p", budget=64)
    # an intervening solver call breaks the cycle; the call itself is never a no-progress step
    flags = A.no_progress_flags(base + [b8, b9, call, b8])
    assert [b for _, b in flags] == [False, False, False, False, False]
    # a new inspection breaks it; a repeated inspection does not
    new_insp = ev("inspect", target="requirements")
    flags = A.no_progress_flags(base + [new_insp, b8, b9, new_insp, b8])
    assert flags[4] == (False, True)  # re-inspection: accepted, no state change -> cycle of length 1
    assert flags[5] == (False, True)  # re-inspection is not a new inspection: the cycle closes
    flags = A.no_progress_flags(base + [b8, b9, ev("inspect", target="map"), b8])
    assert flags[3] == (False, False) and flags[4] == (False, False)  # first inspection of the map is new
    # rejected actions are never flagged and do not break a cycle
    rejected = ev("commit_pending", status="rejected")
    flags = A.no_progress_flags(base + [b8, b9, rejected, rejected, b8])
    assert flags[2] == (False, False) and flags[3] == (False, False) and flags[4] == (False, False)
    assert flags[5] == (False, True)
    # a return to a state 7+ steps back is outside the 6-step window
    steps = base + [b8] + [ev("add_constraint", problem="p", constraint="finish_by", bound=b) for b in range(1, 7)] + [b8]
    flags = A.no_progress_flags(steps)
    assert flags[-1] == (False, False)
    # idempotence requires the previous *accepted* action to share the key
    flags = A.no_progress_flags(base + [ev("think"), rejected, ev("think")])
    assert flags[3] == (True, True)
    flags = A.no_progress_flags(base + [ev("think"), ev("retrieve", handle="r"), ev("think")])
    assert flags[3] == (False, True)


def test_commitments_pending_and_event_revocation_change_state():
    history = [fb("choose_item", {"status": "success", "pending": ["A1"]}, item="A1"),
               fb("choose_item", {"status": "success", "pending": ["A2"]}, item="A2"),
               fb("choose_item", {"status": "success", "pending": ["A1"]}, item="A1"),  # back to a seen draft
               fb("commit_pending", {"status": "success", "selection_id": "s1"}),
               fb("uncommit", {"status": "success", "uncommitted": ["selection", "assignment"]}, target="select"),
               fb("commit_pending", {"status": "success", "selection_id": "s1",
                                     "event": {"revoked": ["selection"], "affected": "capacity"}}),
               fb("think", {"status": "success"})]
    flags = A.no_progress_flags(history)
    # the event step is new public information (breaks cycles, like a new inspection) although its
    # commit was revoked; the next no-effect action closes a cycle of length 1 again
    assert [b for _, b in flags] == [False, False, True, False, True, False, True]
    m = A.no_progress_flags([fb("verify", {"status": "success", "verified": True})])
    assert m == [(False, False)]  # verification changes the public state


def test_steps_to_cap_uses_history_length_not_truncated_field():
    row = {"outcome": {"history": [ev("think")] * 5, "verified_success": False}, "truncated": False}
    assert A.episode_no_progress(row, {"step_limit": 5})["steps_to_cap"] == 1
    assert A.episode_no_progress(row, {"step_limit": 96})["steps_to_cap"] == 0
    assert A.episode_no_progress(row, {"step_limit": 96}, decision_cap=5)["steps_to_cap"] == 1
    row["outcome"]["verified_success"] = True
    assert A.episode_no_progress(row, {"step_limit": 5})["steps_to_cap"] == 0


def test_mirrored_constants_match_environment():
    from tensegra import campaign03_depworld as D
    assert A.REQUIREMENTS == D.REQUIREMENTS and A.READS == D.READS


def test_action_key_matches_environment():
    from tensegra.campaign02_world import Action
    from tensegra.campaign03_depworld import action_key
    for action in (Action("think"), Action("add_constraint", {"problem": "p", "constraint": "finish_by", "bound": 8}),
                   Action("retrieve", {"handle": "r0123"})):
        assert A.action_key({"kind": action.kind, "arguments": dict(action.arguments)}) == action_key(action)


# ---------------------------------------------------------------------------
# Reconstruction against real observations; reference calibration
# ---------------------------------------------------------------------------

def _observed_key(o):
    """The same decision-state key, read directly from the environment's public observation."""
    drafts = {name: (p["primitive"], tuple(sorted(p["problem"].get("constraints", ()))), p["problem"].get("finish_by"),
                     tuple(sorted(p["problem"].get("excluded", ()))), json.dumps(p["depends_on"], sort_keys=True))
              for name, p in o.problems.items()}
    return (tuple(sorted(drafts.items())), tuple(sorted(o.pending)), tuple(sorted(o.pending_assignment.items())),
            o.selection_id, o.assignment_id, o.verified, tuple(sorted(o.retrieved)), o.position)


@pytest.mark.parametrize("policy", ["random", "dep_reuse", "dep_naive_reuse", "dep_greedy"])
def test_reconstruction_matches_environment_observations(policy):
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign03_depworld import DepReference, DepWorkshop, action_catalog, depworld_executor, generate_depworld
    rng = random.Random(5)
    for k in range(6):
        seed = DEV + 10 * k + len(policy)
        spec = generate_depworld(seed, p_event=1.0, foreign_records=2 + k % 3)
        env = DepWorkshop(spec, executor=partial(depworld_executor, execute_call=execute), address_seed=seed)
        ref = None if policy == "random" else DepReference(policy.removeprefix("dep_"))
        o = env.observe()
        observed = [_observed_key(o)]
        while not o.done:
            a = rng.choice(action_catalog(o)) if ref is None else ref.choose(o)
            o = env.step(a)
            observed.append(_observed_key(o))
        spec_dict = json.loads(json.dumps(asdict(spec)))
        state = A.DecisionState(spec_dict["start"], A.initial_drafts(spec_dict))
        rebuilt = [state.key()]
        for event in env.evaluate()["history"]:
            state.apply(event)
            rebuilt.append(state.key())
        assert rebuilt == observed, (policy, seed)


def test_reference_calibration_is_near_zero():
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign03_depworld import DepReference, DepWorkshop, depworld_executor, generate_depworld
    totals = {}
    for mode in ("reuse", "recompute"):
        n = steps = flagged = 0
        for k in range(12):
            seed = DEV + 500 + k
            spec = generate_depworld(seed, p_event=0.5, foreign_records=2)
            env = DepWorkshop(spec, executor=partial(depworld_executor, execute_call=execute), address_seed=seed)
            ref, o = DepReference(mode), env.observe()
            while not o.done:
                o = env.step(ref.choose(o))
            m = A.episode_no_progress({"outcome": env.evaluate()}, json.loads(json.dumps(asdict(spec))))
            n, steps, flagged = n + 1, steps + m["decisions"], flagged + m["no_progress"]
        totals[mode] = flagged / steps
    assert all(v <= 0.02 for v in totals.values()), totals


# ---------------------------------------------------------------------------
# Configs and the deployment rule
# ---------------------------------------------------------------------------

def test_arms_c0_is_byte_identical_and_c1_adds_only_the_anchor(tmp_path):
    c0, c1 = C.write_arms(ROOT / "configs/campaign03", tmp_path)
    assert (tmp_path / "p2a-c0-x1-r2.json").read_bytes() == (ROOT / "configs/campaign03/p1-rl-x1-r2.json").read_bytes()
    banks = json.loads((ROOT / "configs/campaign03/p1-banks.json").read_text())
    assert set(C.arm_difference(c0, c1)) == {"train.anchor_kl_weight", "train.anchor_checkpoint"}
    assert c1["train"]["anchor_checkpoint"] == banks["x1-r2"] and c1["train"]["anchor_kl_weight"] == 0.3
    assert c1["member_hyperparameters"][0]["kl_weight"] == 0.3  # the tranche KL is kept
    assert c1["rounds"] * c1["updates_per_slot"] * 6 == 1800
    if (ROOT / "configs/campaign03/p2a-c1-x1-r2.json").exists():
        assert (ROOT / "configs/campaign03/p2a-c1-x1-r2.json").read_text() == (tmp_path / "p2a-c1-x1-r2.json").read_text()
        assert (ROOT / "configs/campaign03/p2a-c0-x1-r2.json").read_bytes() == (tmp_path / "p2a-c0-x1-r2.json").read_bytes()
    from tensegra.campaign02_population import PopulationConfig
    from tensegra.campaign02_training import TrainConfig
    cfg = PopulationConfig.from_json(c1)
    assert TrainConfig(**cfg.train).anchor_checkpoint["sha256"] == banks["x1-r2"]["sha256"]


def _run(tmp_path, name, rows, status="completed", seeds_hash="S"):
    run = tmp_path / name
    run.mkdir()
    allocations = [{"round": i // 6, "slot": i % 6, "cumulative_slot_updates": 60 * (i + 1),
                    "checkpoint": f"checkpoints/round-{i // 6}-slot-{i % 6}-attempt-{i}.pt",
                    "checkpoint_sha256": f"h{i}", "success": s, "utility": u, "seeds_hash": seeds_hash,
                    "examples": 128} for i, (s, u) in enumerate(rows)]
    state = {"status": status, "allocations": allocations, "finalist": allocations[-1]}
    (run / "state.json").write_text(json.dumps(state))
    return run


def test_deployment_rule_latest_qualifying_else_rollback(tmp_path):
    boot = C.bootstrap_development(_run(tmp_path, "boot", [(0.95, 0.80)]))
    assert boot["success"] == 0.95 and boot["checkpoint"].endswith("attempt-0.pt")
    rows = [(0.95, 0.80), (0.94, 0.79), (0.93, 0.785), (0.96, 0.77), (0.60, 0.40), (0.0, -0.1)]
    d = C.deployment(_run(tmp_path, "c0", rows), boot)
    assert [c["qualifies"] for c in d["candidates"]] == [True, True, True, False, False, False]
    assert d["deployed"]["label"] == "round-0-slot-2-attempt-2" and d["deployed"]["kind"] == "tranche"
    assert d["final"]["success"] == 0.0 and not d["deployed_is_final"]
    assert d["thresholds"]["success"] == pytest.approx(0.93)
    d = C.deployment(_run(tmp_path, "c1", [(0.5, 0.3), (0.9, 0.9)]), boot)
    assert d["deployed"]["kind"] == "rollback" and d["deployed"]["sha256"] == boot["sha256"]
    d = C.deployment(_run(tmp_path, "c2", [(0.5, 0.3), (0.97, 0.9)]), boot)
    assert d["deployed_is_final"]
    with pytest.raises(ValueError, match="development worlds differ"):
        C.deployment(_run(tmp_path, "c3", [(0.9, 0.9)], seeds_hash="T"), boot)
    with pytest.raises(ValueError, match="not completed"):
        C.deployment(_run(tmp_path, "c4", [(0.9, 0.9)], status="running"), boot)


def test_screening_config_structure():
    policies = [{"name": f"p-{r}", "path": "x", "sha256": "y", "role": r} for r in C.POLICY_ROLES]
    cfg = C.screening(policies)
    names = [c["name"] for c in cfg["conditions"]]
    assert names == ["iid_f0", "iid_f0-sampled", "iid_f2", "iid_f2-sampled", "events_train_kinds_p1",
                     "events_train_kinds_p1-sampled", "foreign4", "foreign4-sampled"]
    p1 = _load("campaign03_p1_configs")
    sealed = dict(p1.SEALED)
    for c in cfg["conditions"]:
        base = c["name"].removesuffix("-sampled")
        assert c["seed_start"] == 120_000_000 + 100_000 * C.SCREENING_CONDITIONS.index(base)
        assert c["examples"] == 256 and c["world"] == sealed[base]
        if c["name"].endswith("-sampled"):
            assert c["policy_mode"] == "sampled" and c["sampling_seed"] == C.SAMPLING_SEED and c["references"] == []
        else:
            assert "policy_mode" not in c and "references" not in c
    assert cfg["references"] == ["dep_reuse"]
    # Fresh: disjoint from P1 sealed (110M+), development and training streams.
    assert all(not (110_000_000 <= c["seed_start"] < 111_000_000) for c in cfg["conditions"])
    assert len(C.screening(policies, sampled=False)["conditions"]) == 4
    with pytest.raises(ValueError):
        C.screening(policies + [dict(policies[0])])


# ---------------------------------------------------------------------------
# Sampled evaluation mode and a tiny end-to-end screening
# ---------------------------------------------------------------------------

def test_sampling_is_per_world_deterministic_and_greedy_default_unchanged():
    import torch
    from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign02_training import batched_episodes, inverse_cdf_choice, sampling_rng_seed
    from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld
    assert inverse_cdf_choice([0.0, 0.5, 0.0, 0.5], 0.0) == 1
    assert inverse_cdf_choice([0.0, 0.5, 0.0, 0.5], 0.75) == 3
    assert inverse_cdf_choice([0.0, 0.5, 0.0, 0.5 - 1e-12], 0.9999999999999) == 3
    torch.manual_seed(0)
    model = CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1")).eval()
    seeds = [DEV + 700 + i for i in range(4)]
    make = lambda s: DepWorkshop(generate_depworld(s, foreign_records=2),
                                 executor=partial(depworld_executor, execute_call=execute), address_seed=s)
    actions = lambda rows: [[t["action"] for t in r["trace"]] for r in rows]
    with torch.no_grad():
        greedy = batched_episodes(model, [make(s) for s in seeds], max_steps=24)
        explicit_none = batched_episodes(model, [make(s) for s in seeds], max_steps=24, samplers=None)
        full = batched_episodes(model, [make(s) for s in seeds], max_steps=24,
                                samplers=[random.Random(sampling_rng_seed(s, 7)) for s in seeds])
        split = []
        for s in seeds:  # batch layout must not matter
            split += batched_episodes(model, [make(s)], max_steps=24, samplers=[random.Random(sampling_rng_seed(s, 7))])
        other = batched_episodes(model, [make(s) for s in seeds], max_steps=24,
                                 samplers=[random.Random(sampling_rng_seed(s, 8)) for s in seeds])
    assert actions(greedy) == actions(explicit_none)
    assert actions(full) == actions(split)
    assert actions(full) != actions(greedy) and actions(full) != actions(other)


def test_tiny_screening_end_to_end(tmp_path):
    """Random-init d1 checkpoints through campaign02_evaluate.py (greedy + sampled) and the P2a analysis."""
    import torch
    from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
    from tensegra.campaign02_training import TrainConfig
    policies = []
    for i, role in enumerate(C.POLICY_ROLES):
        torch.manual_seed(i)
        model = CandidatePolicy(PolicyConfig(82, 153, width=8, feature_version="d1"))
        ck = tmp_path / f"{role}.pt"
        torch.save({"config": asdict(TrainConfig(width=8, max_steps=24)), "policy_config": asdict(model.config),
                    "model": model.state_dict(), "updates": 0, "presentations": 0}, ck)
        policies.append({"name": f"p2a-{role}", "role": role, "path": str(ck),
                         "sha256": hashlib.sha256(ck.read_bytes()).hexdigest()})
    (tmp_path / "policies.json").write_text(json.dumps(policies))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(ROOT / "research/tools/campaign03_p2a_configs.py"), "screening",
                    "--policies", str(tmp_path / "policies.json"), "--examples", "2", "--seed-start", str(DEV + 900),
                    "--output", str(tmp_path / "cfg")], check=True, env=env, timeout=120)
    cfg_path = tmp_path / "cfg" / f"p2a-screening-nonprotocol-e2-s{DEV + 900}.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["conditions"] = [c for c in cfg["conditions"] if c["name"].startswith("iid_f")]
    cfg_path.write_text(json.dumps(cfg))
    outs = []
    for batch in (32, 1):
        cfg["evaluation_batch"] = batch
        path = tmp_path / f"cfg-{batch}.json"
        path.write_text(json.dumps(cfg))
        out = tmp_path / f"eval-{batch}"
        subprocess.run([sys.executable, str(ROOT / "research/tools/campaign02_evaluate.py"), str(path),
                        "--output", str(out), "--device", "cpu"], check=True, env=env, timeout=900)
        outs.append(out)
    summary = json.loads((outs[0] / "summary.json").read_text())
    sampled = [r for r in summary["results"] if r["condition"].endswith("-sampled")]
    greedy = [r for r in summary["results"] if not r["condition"].endswith("-sampled")]
    assert sampled and all(r["policy_mode"] == "sampled" and r["kind"] == "learned" for r in sampled)
    assert all("policy_mode" not in r for r in greedy)
    assert {r["arm"] for r in greedy if r["kind"] == "supplied_schedule"} == {"reference-dep_reuse"}
    read = lambda p: [json.loads(l) for l in gzip.open(p, "rt")]
    for result in summary["results"]:
        a, b = read(outs[0] / result["artifact"]), read(outs[1] / result["artifact"])
        assert [x["outcome"]["history"] for x in a] == [x["outcome"]["history"] for x in b]
        if result["condition"].endswith("-sampled"):
            assert all(x["policy_mode"] == "sampled" and x["sampling_seed"] == C.SAMPLING_SEED for x in a)
        elif result["kind"] == "learned":
            assert all("policy_mode" not in x for x in a)
    deployment = {"bootstrap_development": {"success": 0.95, "utility": 0.8}, "final": {"success": 0.1},
                  "deployed": {"label": "bootstrap"}}
    result = A.analyze(outs[0], deployment, deployment)
    assert set(result["readings"]) == {"S0", "S1", "S2", "S3", "replay_on_screening_worlds"}
    assert result["readings"]["S0"]["collapse_reproduced"] is True
    assert "bootstrap/greedy/iid_f0" in result["table"] and "bootstrap/sampled/iid_f2" in result["table"]
    assert "dep_reuse/greedy/iid_f0" in result["table"]
    row = result["table"]["bootstrap/greedy/iid_f0"]
    for key in ("success", "utility", "work_per_success", "correct_reuse_rate", "invalid_reuse_rate",
                "idempotent_repeat_rate", "short_cycle_rate", "no_progress_rate", "no_progress_per_episode",
                "steps_to_cap_rate"):
        assert key in row
    assert len(row["no_progress_counts"]) == 2
    assert A.markdown(result).startswith("# P2a screening analysis")
