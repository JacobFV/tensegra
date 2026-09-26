"""extended-03 Stage B preflight gate (CPU): the encoded d1 inputs preserve every required distinction and the
P1 ablations d1-noapp / d1-noattempt erase exactly the intended ones. A failing preflight fails this suite.

The audit itself lives in research/tools/campaign03_preflight.py (also writes the review JSON).
"""
from functools import partial
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensegra.campaign02_protocol import execute
from tensegra.campaign02_world import Action
from tensegra.campaign03_depworld import (ATTEMPT_BLOCK, CANDIDATE_NAMES_D1, FEATURE_MASKS, FEATURE_VERSIONS,
    NOAPP_CANDIDATE, NOATTEMPT_CANDIDATE, OBSERVATION_NAMES_D1, REASONS, DepReference, DepWorkshop, action_catalog,
    depworld_executor, encode_action_d1, encode_observation_d1, encode_public, generate_depworld, relations)

ROOT = Path(__file__).resolve().parents[1]
EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_000_000_000  # development seeds only


def _load_tool():
    spec = importlib.util.spec_from_file_location("campaign03_preflight", ROOT / "research/tools/campaign03_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def result(tool):
    torch.set_num_threads(1)
    return tool.audit(functional_seeds=4)


def _fmt(pairs):
    return json.dumps([{k: p[k] for k in ("id", "description", "expected", "version_pass", "preconditions")}
                       | {"d1": {k: p["versions"]["d1"].get(k) for k in ("row_diff", "observation_diff")}}
                       for p in pairs], indent=1, default=str)


# --- the gate --------------------------------------------------------------------------------

@pytest.mark.parametrize("distinction", range(1, 11))
def test_preflight_distinction(result, distinction):
    pairs = [p for p in result["pairs"] if p["distinction"] == distinction]
    assert pairs, f"no pairs for distinction {distinction}"
    failed = [p for p in pairs if not p["pass"]]
    assert not failed, "preflight pairs failed:\n" + _fmt(failed)


def test_preflight_d1_preserves_required_distinctions(result):
    """Every pair with a required d1 verdict (True/False) meets it; required distinctions 1-8 differ in d1."""
    for p in result["pairs"]:
        if p["expected"].get("d1") is not None:
            assert p["version_pass"]["d1"], _fmt([p])
    for d in range(1, 9):
        assert any(p["distinction"] == d and p["expected"]["d1"] is True and p["versions"]["d1"]["distinguishable"]
                   for p in result["pairs"]), d


def test_preflight_ablations_erase_exactly_the_intended_distinctions(result):
    by = {}
    for p in result["pairs"]:
        by.setdefault(p["distinction"], []).append(p)
    # d1-noapp erases the relation evidence of 1-4 (metadata-only pairs become identical) ...
    for d in (1, 2, 3, 4):
        erased = [p for p in by[d] if p["expected"]["d1-noapp"] is False]
        assert erased and all(not p["versions"]["d1-noapp"]["distinguishable"] for p in erased), d
    # ... and keeps 5-8; d1-noattempt erases the attempt-memory pairs of 6-7 and keeps 1-5 and 8.
    for d in (5, 6, 7, 8):
        assert all(p["version_pass"]["d1-noapp"] for p in by[d]), d
    for d in (6, 7):
        erased = [p for p in by[d] if p["expected"]["d1-noattempt"] is False]
        assert erased and all(not p["versions"]["d1-noattempt"]["distinguishable"] for p in erased), d
    for d in (1, 2, 3, 4, 5, 8):
        assert all(p["version_pass"]["d1-noattempt"] for p in by[d]), d
    # the retrieved-payload variants keep only declared feasibility facts under d1-noapp
    kept = set(result["noapp_kept_payload_facts"])
    for p in result["pairs"]:
        if p["expected"].get("d1-noapp") == "residual":
            assert set(p["versions"]["d1-noapp"]["row_diff"]) <= kept, _fmt([p])


def test_preflight_functional_ablation_checks(result):
    f = result["functional"]
    assert f["pass"], f
    assert f["states"] > 100 and f["d1_sensitive_to_relations_states"] > 0 and f["d1_sensitive_to_attempts_states"] > 0


def test_preflight_catalogue_and_dimensions(result):
    assert all(c["pass"] for c in result["catalog_checks"].values()), result["catalog_checks"]
    assert result["dimensions"] == {v: [len(OBSERVATION_NAMES_D1), len(CANDIDATE_NAMES_D1)] for v in FEATURE_VERSIONS}
    assert result["gate"]["pass"], result["gate"]


# --- the ablation feature versions -----------------------------------------------------------

def _states(n=3, mode="reuse"):
    for i in range(n):
        env = DepWorkshop(generate_depworld(DEV + 8_000 + i, p_event=1.0, foreign_records=4), executor=EXEC,
                          address_seed=i)
        ref, o = DepReference(mode), env.observe()
        while not o.done:
            yield o
            o = env.step(ref.choose(o))


def test_feature_versions_are_registered_everywhere():
    from tensegra.campaign02_policy import PolicyConfig
    from tensegra.campaign02_training import DEPWORLD_FEATURE_VERSIONS, public_frame
    assert FEATURE_VERSIONS == DEPWORLD_FEATURE_VERSIONS == ("d1", "d1-noapp", "d1-noattempt")
    for v in FEATURE_VERSIONS:
        PolicyConfig(82, 153, width=8, feature_version=v)
    with pytest.raises(ValueError):
        PolicyConfig(82, 153, width=8, feature_version="d1-bogus")
    o = DepWorkshop(generate_depworld(DEV + 1)).observe()
    for v in FEATURE_VERSIONS:
        acts, x, m = public_frame(o, v)
        assert acts == action_catalog(o) and len(x) == 82 and all(len(r) == 153 for r in m)
    with pytest.raises(ValueError):
        public_frame(o, "v1")
    with pytest.raises(ValueError):
        encode_public(o, action_catalog(o), "d2")
    from tensegra.campaign02_world import generate_world, Workshop
    with pytest.raises(ValueError):
        public_frame(Workshop(generate_world(0)).observe(), "d1-noapp")


def test_mask_definitions():
    assert len(OBSERVATION_NAMES_D1) == len(set(OBSERVATION_NAMES_D1)) == 82
    assert len(CANDIDATE_NAMES_D1) == len(set(CANDIDATE_NAMES_D1)) == 153
    assert FEATURE_MASKS["d1"] == ((), ())
    assert NOATTEMPT_CANDIDATE == CANDIDATE_NAMES_D1[-ATTEMPT_BLOCK:]
    assert {CANDIDATE_NAMES_D1[i] for i in FEATURE_MASKS["d1-noapp"][1]} == set(NOAPP_CANDIDATE)
    assert {OBSERVATION_NAMES_D1[i] for i in FEATURE_MASKS["d1-noattempt"][0]} == {
        "attempts.count", "attempts.rejected", "attempts.uncommits"}
    # the two ablations are disjoint and neither touches type match, usable status or last-step feedback
    assert not set(FEATURE_MASKS["d1-noapp"][1]) & set(FEATURE_MASKS["d1-noattempt"][1])
    kept = ("record.type_match", "record.usable", "record.certificate_valid", "record.retrieved")
    assert not {CANDIDATE_NAMES_D1.index(n) for n in kept} & (set(FEATURE_MASKS["d1-noapp"][1]) |
                                                               set(FEATURE_MASKS["d1-noattempt"][1]))
    assert not any(OBSERVATION_NAMES_D1[i].startswith("feedback.") for v in FEATURE_VERSIONS
                   for i in FEATURE_MASKS[v][0])


def test_coordinate_names_match_the_encoder():
    """The semantic names that define the masks point at the coordinates the encoder fills."""
    O, C = OBSERVATION_NAMES_D1.index, CANDIDATE_NAMES_D1.index
    seen = {"relations": 0, "attempt_reason": 0}
    for o in _states(2, "naive_reuse"):
        x = encode_observation_d1(o)
        assert x[O("attempts.count")] == len(o.attempts) / 32
        assert x[O("req.known")] == float(o.requirements is not None)
        assert x[O("plan.selection_committed")] == float(o.selection_id is not None)
        if o.feedback.get("reason") in REASONS:
            assert x[O("feedback.reason." + o.feedback["reason"])] == 1.0
        for a in action_catalog(o):
            if a.kind != "use_return" and a.kind not in ("commit_pending", "commit_assignment", "move"):
                continue
            row = encode_action_d1(o, a)
            if a.kind == "use_return":
                rec = next(r for r in o.records if r["handle"] == a.arguments["handle"])
                want = {"select": "constrained_subset", "assign": "csp", "route": "shortest_path"}[a.arguments["as"]]
                rel = relations(o, rec, want)
                for k in ("type_match", "request_match", "canonical_match", "dependency_match",
                          "requirements_match", "selection_match", "usable"):
                    assert row[C("record." + k)] == float(rel[k]), k
                seen["relations"] += 1
            last = next((at for at in reversed(o.attempts) if at["action_kind"] == a.kind
                         and at["arguments"] == dict(a.arguments)), None)
            assert row[C("attempt.attempted")] == float(last is not None)
            if last is not None and last["reason"] is not None:
                assert row[C("attempt.last_reason." + last["reason"])] == 1.0
                seen["attempt_reason"] += 1
    assert seen["relations"] > 100 and seen["attempt_reason"] > 0


def test_ablations_mask_only_their_coordinates_and_keep_d1_elsewhere():
    for o in _states(2):
        acts = action_catalog(o)
        x1, m1 = encode_public(o, acts, "d1")
        for v in ("d1-noapp", "d1-noattempt"):
            xv, mv = encode_public(o, acts, v)
            om, cm = FEATURE_MASKS[v]
            assert all(xv[i] == 0.0 for i in om) and all(r[i] == 0.0 for r in mv for i in cm)
            assert len(xv) == len(x1) and all(len(r) == len(s) for r, s in zip(mv, m1))
        xa, ma = encode_public(o, acts, "d1-noattempt")
        skip = set(FEATURE_MASKS["d1-noattempt"][1])
        assert [v for i, v in enumerate(xa) if i not in FEATURE_MASKS["d1-noattempt"][0]] == \
            [v for i, v in enumerate(x1) if i not in FEATURE_MASKS["d1-noattempt"][0]]
        assert all(r[i] == s[i] for r, s in zip(ma, m1) for i in range(len(s)) if i not in skip)


def test_p1_ablation_configs_train_on_masked_inputs():
    """The frozen X3/X4 boot configs load, build a policy of their feature version, and the teacher-collection
    path (prepare_frame via model.config.feature_version) feeds the masked tensors."""
    from tensegra.campaign02_population import PopulationConfig, new_policy
    from tensegra.campaign02_training import collect_teacher
    torch.set_num_threads(1)
    for arm, version in (("x3", "d1-noapp"), ("x4", "d1-noattempt")):
        cfg = PopulationConfig.from_json(json.loads((ROOT / f"configs/campaign03/p1-boot-{arm}-r0.json").read_text()))
        assert cfg.policy["feature_version"] == version
        model = new_policy(cfg.policy, 82, 153, 8, "lightweight")
        assert model.config.feature_version == version
        spec = generate_depworld(DEV + 90, p_event=1.0, foreign_records=2)
        frames, out = collect_teacher(DepWorkshop(spec, executor=EXEC), DepReference("reuse"), 96, model)
        assert frames and out["outcome"]["verified_success"]
        om, cm = FEATURE_MASKS[version]
        assert all(f.observation[i] == 0.0 for f in frames for i in om)
        assert all(r[i] == 0.0 for f in frames for r in f.candidates for i in cm)


def test_d1_noattempt_hides_a_rejection_after_one_intervening_action(tool):
    """Direct, minimal form of distinction 6/7 under the ablation (independent of the audit bookkeeping)."""
    r = tool.Run(tool.make_spec())
    r.inspect_all()
    r.commit("A1", "B3")                                # capacity rejection
    r.step("think")
    s = tool.Run(tool.make_spec())
    s.inspect_all()
    s.steps([("choose_item", {"item": "A1"}), ("choose_item", {"item": "B3"}), ("think", {}), ("think", {})])
    from tensegra.campaign02_training import public_frame
    assert public_frame(r.o, "d1") != public_frame(s.o, "d1")
    assert public_frame(r.o, "d1-noapp") != public_frame(s.o, "d1-noapp")
    assert public_frame(r.o, "d1-noattempt")[1:] == public_frame(s.o, "d1-noattempt")[1:]
    commit = Action("commit_pending")
    acts = public_frame(r.o, "d1")[0]
    assert public_frame(r.o, "d1")[2][acts.index(commit)] != public_frame(s.o, "d1")[2][acts.index(commit)]
