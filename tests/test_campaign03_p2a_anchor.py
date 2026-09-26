"""extended-03 P2a fixed bootstrap anchor (protocol-P2a.md, arm C1) and phase profiling (CPU, tiny widths).

Mechanical fixtures only: width-8 policies on small depworld worlds, never experimental substitutes.
"""
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import (Learner, PhaseClock, TrainConfig, actor_critic_objective, batched_on_policy,
                                          load_anchor)
from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld

EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_070_000_000  # development seeds only
WORLD = {"categories": 2, "choices": 2, "locations": 5, "slots": 5, "foreign_records": 2, "p_event": 0.5,
         "compute_price": 0.0001, "event_trigger": "progress"}


def factory(seed):
    return DepWorkshop(generate_depworld(seed, **WORLD), executor=EXEC, address_seed=seed + 3)


def dims():
    from tensegra.campaign02_training import public_frame
    _, obs, candidates = public_frame(factory(DEV).observe(), "d1")
    return len(obs), len(candidates[0])


def new_model(seed):
    torch.manual_seed(seed)
    o, c = dims()
    return CandidatePolicy(PolicyConfig(o, c, width=8, feature_version="d1"))


def config(**over):
    base = dict(width=8, method="actor_critic", rollout_mode="batched", batch_size=2, max_steps=10,
                learning_rate=1e-3, entropy_weight=0.003, kl_weight=0.3, advantage_normalization=True,
                training_seed_start=DEV + 100)
    base.update(over)
    return TrainConfig(**base)


def save_checkpoint(tmp_path, seed, name="anchor.pt"):
    model = new_model(seed)
    learner = Learner(model, config())
    path = tmp_path / name
    learner.save(path, {"role": "anchor fixture"})
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def train(cfg, updates=3, model_seed=11, torch_seed=5):
    model = new_model(model_seed)
    learner = Learner(model, cfg)
    torch.manual_seed(torch_seed)
    timing = learner.train_tranche(updates, factory)
    return learner, timing


def test_config_validation():
    with pytest.raises(ValueError, match="anchor_checkpoint"):
        config(anchor_kl_weight=0.3)
    with pytest.raises(ValueError):
        config(anchor_kl_weight=-1.0, anchor_checkpoint={"path": "x", "sha256": "y"})
    with pytest.raises(ValueError, match="path, sha256"):
        config(anchor_checkpoint={"path": "x"})
    assert config().anchor_kl_weight == 0.0 and config().anchor_checkpoint is None


def test_absent_and_zero_weight_anchor_are_bit_identical(tmp_path):
    """C0 contract: the option absent vs present with weight 0 gives identical parameters,
    optimizer state, data hash and curves (the anchor file is never even opened at weight 0)."""
    anchor = save_checkpoint(tmp_path, 99)
    a, _ = train(config())
    b, _ = train(config(anchor_kl_weight=0.0, anchor_checkpoint=anchor))
    c, _ = train(config(anchor_kl_weight=0.0, anchor_checkpoint={"path": str(tmp_path / "missing.pt"),
                                                                 "sha256": "0" * 64}))
    for other in (b, c):
        assert a.model.state_dict().keys() == other.model.state_dict().keys()
        assert all(torch.equal(v, other.model.state_dict()[k]) for k, v in a.model.state_dict().items())
        # data_hash is not compared: rollout traces carry wall-clock timing fields.
        assert [x["loss"] for x in a.curves] == [x["loss"] for x in other.curves]
        assert [x["objective_parts"] for x in a.curves] == [x["objective_parts"] for x in other.curves]
        assert all("kl_to_anchor" not in x["objective_parts"] for x in other.curves)
        sa, sb = a.optimizer.state_dict()["state"], other.optimizer.state_dict()["state"]
        assert all(torch.equal(sa[k][n], sb[k][n]) for k in sa for n in sa[k])
    assert other.anchor is None


def test_positive_anchor_changes_training_and_logs_kl(tmp_path):
    anchor = save_checkpoint(tmp_path, 99)
    base, _ = train(config())
    anchored, timing = train(config(anchor_kl_weight=0.3, anchor_checkpoint=anchor))
    parts = [x["objective_parts"] for x in anchored.curves]
    assert all(p["kl_to_anchor"] > 0 for p in parts)  # a different frozen policy: KL strictly positive
    assert all("kl_to_round_start" in p for p in parts)  # the tranche KL is kept
    assert any(not torch.equal(v, anchored.model.state_dict()[k]) for k, v in base.model.state_dict().items())
    assert timing["anchor"]["sha256"] == anchor["sha256"] and timing["anchor"]["verified"]
    assert timing["anchor"]["anchor_kl_weight"] == 0.3
    # The anchor is frozen and never replaced: bit-identical to the file after training.
    saved = torch.load(anchor["path"], map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(v, anchored.anchor.state_dict()[k]) for k, v in saved.items())
    assert not any(p.requires_grad for p in anchored.anchor.parameters())


def test_anchor_equal_to_start_has_zero_initial_kl(tmp_path):
    anchor = save_checkpoint(tmp_path, 11)  # same weights as the learner's start
    learner, _ = train(config(anchor_kl_weight=0.3, anchor_checkpoint=anchor), updates=2)
    first = learner.curves[0]["objective_parts"]
    assert abs(first["kl_to_anchor"]) < 1e-6 and abs(first["kl_to_round_start"]) < 1e-6
    assert learner.curves[1]["objective_parts"]["kl_to_anchor"] > 0  # drift after one update is measured


def test_anchor_kl_matches_tranche_kl_definition_and_gradients_flow(tmp_path):
    """Same masking/valid-action handling and direction: with the same frozen policy as
    reference and anchor, the two per-decision KLs are identical tensors; the anchor term
    backpropagates into the current policy only."""
    anchor_info = save_checkpoint(tmp_path, 99)
    model = new_model(11)
    frozen, provenance = load_anchor(anchor_info, model)
    assert provenance["sha256"] == anchor_info["sha256"]
    torch.manual_seed(1)
    envs = [factory(DEV + i) for i in range(3)]
    results, terms, kls, anchor_kls = batched_on_policy(model, envs, max_steps=6, reference=frozen, anchor=frozen)
    assert [len(t) for t in terms] == [len(k) for k in anchor_kls]
    for x, y in zip(kls, anchor_kls):
        assert all(torch.equal(a, b) for a, b in zip(x, y))
    cfg = config(anchor_kl_weight=0.3, anchor_checkpoint=anchor_info, kl_weight=0.0)
    loss, parts = actor_critic_objective(terms, cfg, None, anchor_kls)
    assert float(parts["kl_to_anchor"].detach()) > 0
    model.zero_grad()
    parts["kl_to_anchor"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in frozen.parameters())
    # Objective arithmetic: the anchor adds exactly weight * mean KL.
    torch.manual_seed(1)
    envs = [factory(DEV + i) for i in range(3)]
    results2, terms2, anchor_kls2 = (lambda r: (r[0], r[1], r[3]))(
        batched_on_policy(model, envs, max_steps=6, anchor=frozen))
    plain, _ = actor_critic_objective(terms2, replace(cfg, anchor_kl_weight=0.0, anchor_checkpoint=None))
    anchored, parts2 = actor_critic_objective(terms2, cfg, None, anchor_kls2)
    assert float(anchored - plain) == pytest.approx(0.3 * float(parts2["kl_to_anchor"]), rel=1e-5, abs=1e-7)


def test_anchor_hash_and_architecture_are_verified(tmp_path):
    anchor = save_checkpoint(tmp_path, 99)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_anchor({**anchor, "sha256": "f" * 64}, new_model(11))
    torch.manual_seed(0)
    o, c = dims()
    wide = CandidatePolicy(PolicyConfig(o, c, width=16, feature_version="d1"))
    with pytest.raises(ValueError, match="architecture"):
        load_anchor(anchor, wide)
    learner = Learner(new_model(11), config(anchor_kl_weight=0.3, anchor_checkpoint={**anchor, "sha256": "e" * 64}))
    with pytest.raises(ValueError, match="hash mismatch"):
        learner.train_tranche(1, factory)


def test_old_checkpoints_without_anchor_fields_import_unchanged(tmp_path):
    """A checkpoint written before the anchor fields existed records no anchor config change at
    import/resume when the anchor is off (the historical resume record is unchanged)."""
    model = new_model(11)
    learner = Learner(model, config())
    path = tmp_path / "old.pt"
    learner.save(path)
    saved = torch.load(path, weights_only=False)
    for key in ("anchor_kl_weight", "anchor_checkpoint"):
        saved["config"].pop(key)
    torch.save(saved, path)
    fresh = Learner(new_model(11), config())
    fresh.load(path)  # no authorization needed: nothing changed
    assert fresh.resume_history[-1]["config_changes"] == {}
    anchor = save_checkpoint(tmp_path, 99)
    with_anchor = Learner(new_model(11), config(anchor_kl_weight=0.3, anchor_checkpoint=anchor))
    with pytest.raises(ValueError, match="Explicit resume"):
        with_anchor.load(path)
    with_anchor.load(path, allow_config_changes=True)
    assert set(with_anchor.resume_history[-1]["config_changes"]) == {"anchor_kl_weight", "anchor_checkpoint"}


def test_phase_clock_is_observational():
    model = new_model(11)
    runs = []
    for clock in (None, PhaseClock()):
        torch.manual_seed(3)
        results, terms = batched_on_policy(model, [factory(DEV + i) for i in range(2)], max_steps=8, clock=clock)
        runs.append(([[s["action"] for s in r["trace"]] for r in results],
                     [[float(t[0]) for t in e] for e in terms], [[t[3] for t in e] for e in terms]))
    assert runs[0] == runs[1]
    assert {"encode", "collate", "forward_policy", "environment_step", "trace_export"} <= set(clock.as_dict())
    _, timing = train(config(), updates=1)
    phases = timing["phase_timing"]
    assert {"world_construction", "rollout_total", "objective", "backward", "optimizer_step",
            "trace_hash_export", "environment_step", "forward_frozen"} <= set(phases)
    assert all(v["wall_seconds"] >= 0 and v["calls"] >= 1 for v in phases.values())


def _population_config(anchor=None, rounds=2):
    world = dict(WORLD)
    train_cfg = {"width": 8, "family": "lightweight", "device": "cpu", "batch_size": 2, "learning_rate": 3e-4,
                 "rollout_mode": "batched", "max_steps": 10, "evaluation_batch": 4,
                 "policy_loss_reduction": "decision_mean", "advantage_normalization": True}
    if anchor is not None:
        train_cfg.update(anchor_kl_weight=0.3, anchor_checkpoint=anchor)
    return {"mode": "single", "population_seed": 7, "initialization_seeds": [70, 71, 72, 73, 74, 75],
            "optimizer_policy": "inherit", "teacher": "dep_reuse", "threads": 1, "world_family": "depworld",
            "policy": {"interface": "legacy", "feature_version": "d1"}, "world_mix": [world], "train": train_cfg,
            "development_seed_start": DEV + 900_000, "development_examples": 2, "training_seed_stride": 1000,
            "member_hyperparameters": [{"learning_rate": 3e-5, "entropy_weight": 0.003, "kl_weight": 0.3}] * 6,
            "rounds": rounds, "updates_per_slot": 1, "methods": ["actor_critic"] * rounds,
            "training_seed_start": DEV + 500_000, "address_namespace": "p2a-test"}


def test_population_flows_anchor_and_records_provenance(tmp_path):
    from tensegra.campaign02_population import PopulationConfig, PopulationRun
    from tensegra.campaign03_depworld import DepReference
    torch.set_num_threads(1)
    bank = save_checkpoint(tmp_path, 99, "bank.pt")
    runs = {}
    for label, anchor in (("c0", None), ("c1", bank)):
        raw = {**_population_config(anchor), "initial_checkpoints": [bank] * 6}
        cfg = PopulationConfig.from_json(json.loads(json.dumps(raw)))
        run = PopulationRun(cfg, tmp_path / label, factory, lambda: DepReference("reuse"))
        run.run()
        runs[label] = run
    c0, c1 = runs["c0"], runs["c1"]
    assert "anchor" not in c0.state
    assert c1.state["anchor"]["checkpoint"] == bank and c1.state["anchor"]["sha256_verified_at_initialization"]
    protocol = json.loads((tmp_path / "c1" / "protocol.json").read_text())
    assert protocol["train"]["anchor_checkpoint"]["sha256"] == bank["sha256"]
    for row in c1.state["allocations"]:
        assert row["training_timing"]["anchor"]["sha256"] == bank["sha256"]
        assert row["training_timing"]["last"]["objective_parts"]["kl_to_anchor"] >= 0
        assert "development_evaluation" in row["training_timing"]["phase_timing"]
    for row in c0.state["allocations"]:
        assert "anchor" not in row["training_timing"]
        assert "kl_to_anchor" not in row["training_timing"]["last"]["objective_parts"]
    final = c1._checkpoint(c1.state["members"][0])
    assert final["config"]["anchor_kl_weight"] == 0.3 and final["config"]["anchor_checkpoint"] == bank
    # The C0 import records exactly the historical config changes (no anchor keys).
    history = c0._checkpoint(c0.state["members"][0])["resume_history"]
    assert all(not ({"anchor_kl_weight", "anchor_checkpoint"} & set(h.get("config_changes", {}))) for h in history)
    # A changed anchor file is refused at initialization.
    bad = {**bank, "sha256": "0" * 64}
    raw = {**_population_config(bad), "initial_checkpoints": [bank] * 6}
    with pytest.raises(ValueError, match="Anchor checkpoint hash mismatch"):
        PopulationRun(PopulationConfig.from_json(raw), tmp_path / "bad", factory, lambda: DepReference("reuse"))
