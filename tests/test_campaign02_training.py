"""Small-width mathematical/mechanical tests, not experimental substitutes."""
from dataclasses import asdict

import pytest
import torch

from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
from topoformer.campaign02_training import Frame, TrainConfig, Learner, collate, supervised_loss, batched_episodes, live_episode
from topoformer.campaign02_world import Workshop, generate_world, encode_observation, encode_action, action_catalog


def test_padding_and_targets():
    obs, candidates, mask, targets = collate([Frame([1.], [[1., 2.]], 0), Frame([2.], [[3., 4.], [5., 6.]], 1)])
    assert candidates.shape == (2, 2, 2)
    assert mask.tolist() == [[True, False], [True, True]]
    assert targets.tolist() == [0, 1]
    with pytest.raises(ValueError):
        collate([Frame([1.], [[1.]], 1)])


@pytest.mark.parametrize("family", ["lightweight", "recurrent"])
def test_sequence_gradients(family):
    torch.manual_seed(7)
    model = CandidatePolicy(PolicyConfig(2, 2, width=8, heads=2, family=family))
    frame = Frame([.1, .3], [[.2, .1], [.5, .2]], 1)
    loss, count = supervised_loss(model, [[frame, frame], [frame]], bptt_steps=1)
    loss.backward()
    assert count == 3 and torch.isfinite(loss)
    assert model.candidate.weight.grad.abs().sum() > 0


def test_checkpoint_restores_rng_and_optimizer(tmp_path):
    torch.manual_seed(8)
    cfg = TrainConfig(width=8)
    model = CandidatePolicy(PolicyConfig(2, 2, width=8, heads=2))
    learner = Learner(model, cfg)
    loss, _ = supervised_loss(model, [[Frame([.1, .2], [[.2, .3], [.4, .5]], 1)]])
    loss.backward()
    learner.optimizer.step()
    learner.save(tmp_path/"model.pt", {"scope": "mechanical"})
    expected_rng = torch.rand(3)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert learner.load(tmp_path/"model.pt") == {"scope": "mechanical"}
    assert torch.equal(torch.rand(3), expected_rng)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    assert learner.optimizer.state


@pytest.mark.parametrize("family", ["lightweight", "recurrent"])
def test_batched_matches_single_closed_loop(family):
    torch.manual_seed(9)
    factory = lambda seed: Workshop(generate_world(seed, choices=1, step_limit=3))
    observation = factory(10).observe()
    model = CandidatePolicy(PolicyConfig(len(encode_observation(observation)),
        len(encode_action(observation, action_catalog(observation)[0])), width=8, heads=2, family=family))
    model.eval()
    with torch.no_grad():
        batch = batched_episodes(model, [factory(10), factory(11)], max_steps=3)
        single = [live_episode(model, factory(seed), max_steps=3)[0] for seed in (10, 11)]
    assert [[r["action"] for r in e["trace"]] for e in batch] == [[r["action"] for r in e["trace"]] for e in single]
    assert [r["outcome"]["utility"] for r in batch] == [r["outcome"]["utility"] for r in single]
    assert batch[1]["timing"] is None


def test_resume_requires_explicit_changes(tmp_path):
    from dataclasses import replace
    model = CandidatePolicy(PolicyConfig(2, 2, width=8, heads=2))
    learner = Learner(model, TrainConfig(width=8))
    learner.save(tmp_path/"model.pt")
    changed = Learner(CandidatePolicy(model.config), replace(learner.config, learning_rate=.001))
    with pytest.raises(ValueError, match="Explicit resume"):
        changed.load(tmp_path/"model.pt")
    changed.load(tmp_path/"model.pt", allow_config_changes=True, reset_learning_rate=True)
    assert changed.optimizer.param_groups[0]["lr"] == .001
    assert changed.resume_history[-1]["learning_rate_policy"] == "requested"


def test_address_stream_independent_and_repeatable():
    from topoformer.campaign02_training import independent_address_seed
    assert independent_address_seed(1, "a") == independent_address_seed(1, "a")
    assert len({independent_address_seed(1, "a"), independent_address_seed(2, "a"), independent_address_seed(1, "b")}) == 3


def test_batched_on_policy_hidden_identity_gradients_and_rewards():
    from topoformer.campaign02_training import batched_on_policy, actor_critic_terms
    torch.manual_seed(12)
    def factory(seed):
        return Workshop(generate_world(seed, choices=1+seed%2, step_limit=2+seed%2))
    obs = factory(10).observe()
    model = CandidatePolicy(PolicyConfig(len(encode_observation(obs)),
        len(encode_action(obs, action_catalog(obs)[0])), width=8, heads=2, family="recurrent"))
    original = model.score
    def prefer_verify(*args, **kwargs):
        logits, value, hidden = original(*args, **kwargs)
        offset = torch.zeros_like(logits)
        offset[:, 0] = 100  # Fixture: incomplete verification keeps worlds alive.
        return logits+offset, value, hidden
    model.score = prefer_verify
    batch, terms = batched_on_policy(model, [factory(10), factory(11)], max_steps=4,
                                    bptt_steps=1, sample=False)
    assert [len(t) for t in terms] == [2, 3]
    for i, seed in enumerate((10, 11)):
        serial, serial_terms = live_episode(model, factory(seed), max_steps=4,
            sample=False, gradients=True, bptt_steps=1)
        assert [r["action"] for r in batch[i]["trace"]] == [r["action"] for r in serial["trace"]]
        assert torch.allclose(torch.stack([r[1] for r in terms[i]]),
                              torch.stack([r[1] for r in serial_terms]), atol=1e-5)
        assert sum(r[3] for r in terms[i]) == pytest.approx(batch[i]["outcome"]["utility"])
    loss = torch.stack([term for episode in terms for term in actor_critic_terms(episode, TrainConfig(width=8))]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.value[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_actor_critic_reward_to_go_not_repeated_total():
    from topoformer.campaign02_training import actor_critic_terms
    cfg = TrainConfig(width=8, entropy_weight=0, value_weight=1)
    terms = [(torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), -.1),
             (torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), .9)]
    assert [float(v) for v in actor_critic_terms(terms, cfg)] == pytest.approx([.81, .64])
