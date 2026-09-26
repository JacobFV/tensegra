"""extended-03 P2a reward accounting (protocol-P2a.md, "Reward accounting check").

On explicit depworld trajectories driven through the actor-critic rollout
(``batched_on_policy``, the code P1's RL used), check that the per-step rewards
telescope to the episode's task utility (verified success - costs) and document
the terminal handling on each path:

- success: the episode ends at a successful ``verify`` (environment done). The last
  reward carries the +1; return-to-go at t=0 equals utility = 1 - cost.
- step-cap failure: the environment itself ends the episode at ``step_limit``
  (done=True, so the row's ``truncated`` is False). No terminal reward; the return
  is -cost; the return-to-go treats the end as terminal (value 0), which is exact.
- truncation (decision cap ``max_steps`` below ``step_limit``): the rollout stops
  with the environment not done (``truncated`` True). The rewards still telescope to
  the utility so far (-cost so far), but the return-to-go treats the cut as terminal:
  no bootstrap V(s_T), no forfeited terminal reward. In P1 (and P2a) max_steps ==
  step_limit == 96, so this path coincides with the step cap and never occurs.
- abstain: environment done; return = -cost.

Initial utility is exactly 0 (no step, observation, work, travel or compute yet),
which is what the rollout's ``previous_utilities = 0.0`` assumes.
"""
from functools import partial

import pytest
import torch
from torch import nn

from tensegra.campaign02_policy import PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import TrainConfig, actor_critic_objective, batched_on_policy, public_frame
from tensegra.campaign02_world import Action
from tensegra.campaign03_depworld import DepReference, DepWorkshop, action_catalog, depworld_executor, generate_depworld

EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_090_000_000  # development seeds only
WORLD = {"categories": 3, "choices": 3, "locations": 7, "slots": 6, "compute_price": 0.0001,
         "event_trigger": "progress", "p_event": 0.5, "foreign_records": 2,
         "event_kinds": ["edge_closed", "capacity_reduced", "slot_closed"]}


def make_env(seed, **over):
    return DepWorkshop(generate_depworld(seed, **{**WORLD, **over}), executor=EXEC, address_seed=seed + 1)


class ScriptedPolicy(nn.Module):
    """Plays a fixed list of catalog indices (argmax), with a live value parameter so the
    actor-critic objective is differentiable. Mechanical fixture, not a learned policy."""

    def __init__(self, obs_dim, cand_dim, indices):
        super().__init__()
        self.config = PolicyConfig(obs_dim, cand_dim, width=8, feature_version="d1")
        self.v = nn.Parameter(torch.zeros(()))
        self.indices, self.calls = list(indices), 0

    def score(self, obs, candidates, hidden=None, mask=None):
        logits = torch.full(candidates.shape[:2], -50.0) + 0 * self.v
        logits[0, self.indices[self.calls]] = 50.0
        self.calls += 1
        return logits.masked_fill(~mask, -torch.inf), self.v.expand(obs.shape[0]), None


def script(seed, choose, limit=200, **over):
    """Catalog indices of `choose(observation)` on a twin environment."""
    env, indices = make_env(seed, **over), []
    o = env.observe()
    while not o.done and len(indices) < limit:
        action = choose(o)
        indices.append(action_catalog(o).index(action))
        o = env.step(action)
    return indices


def rollout(seed, indices, max_steps=96, **over):
    env = make_env(seed, **over)
    assert env.evaluate()["utility"] == 0.0 and env.evaluate()["cost"] == 0.0  # initial utility is exactly 0
    _, obs, cand = public_frame(env.observe(), "d1")
    model = ScriptedPolicy(len(obs), len(cand), indices)
    results, terms = batched_on_policy(model, [env], max_steps=max_steps, sample=False)
    return env, results[0], terms[0], model


def step_costs(env, result):
    """Explicit per-step cost from the recorded history and the world prices."""
    s, out = env._spec, []
    travel_before = 0
    for event in result["outcome"]["history"]:
        kind = event["action"]["kind"]
        cost = s.action_price + s.compute_price * 1.0  # one action + one neural forward per decision
        if kind == "inspect" and event["feedback"].get("status") == "success":
            cost += s.observation_price
        out.append(cost)
    return out


def returns_to_go(terms):
    g, out = 0.0, []
    for *_, r in reversed(terms):
        g += r
        out.append(g)
    return list(reversed(out))


def check_telescoping(result, terms):
    rewards = [t[3] for t in terms]
    utility = result["outcome"]["utility"]
    assert len(rewards) == len(result["trace"]) == result["outcome"]["steps"]
    assert sum(rewards) == pytest.approx(utility, abs=1e-9)
    assert returns_to_go(terms)[0] == pytest.approx(utility, abs=1e-9)
    assert utility == pytest.approx(float(result["outcome"]["verified_success"]) - result["outcome"]["cost"], abs=1e-12)
    # The objective's return-to-go is the same quantity: with zero value, advantage(t=0) = G_0.
    cfg = TrainConfig(width=8, method="actor_critic", value_weight=1.0, entropy_weight=0.0)
    _, parts = actor_critic_objective([terms], cfg)
    expected = sum(g * g for g in returns_to_go(terms)) / len(terms)
    assert float(parts["critic_decision_mean_loss"].detach()) == pytest.approx(expected, rel=1e-5, abs=1e-9)
    return rewards


@pytest.mark.parametrize("seed", [DEV + 1, DEV + 2, DEV + 3])
def test_success_path_telescopes_to_one_minus_cost(seed):
    ref = DepReference("reuse")
    indices = script(seed, ref.choose)
    env, result, terms, _ = rollout(seed, indices)
    assert result["outcome"]["verified_success"] and not result["truncated"]
    rewards = check_telescoping(result, terms)
    history = result["outcome"]["history"]
    assert history[-1]["action"]["kind"] == "verify" and history[-1]["feedback"]["verified"]
    # Every non-terminal reward is minus that step's costs; the final one adds +1.
    work_price, travel_price = env._spec.work_price, env._spec.travel_price
    for t, (r, event) in enumerate(zip(rewards, history)):
        extra = 1.0 if t == len(history) - 1 else 0.0
        assert r <= extra + 1e-12  # costs are nonnegative
    explicit = sum(step_costs(env, result)) + result["outcome"]["work_units"] * work_price \
        + result["outcome"]["travel_distance"] * travel_price
    assert result["outcome"]["cost"] == pytest.approx(explicit, abs=1e-12)
    assert rewards[-1] == pytest.approx(1.0 - step_costs(env, result)[-1], abs=1e-12)


def test_step_cap_failure_is_environment_terminal_with_negative_cost_return():
    seed = DEV + 4
    think = Action("think")
    indices = script(seed, lambda o: think)
    env, result, terms, _ = rollout(seed, indices, max_steps=96)
    assert len(indices) == env._spec.step_limit == 96
    assert not result["outcome"]["verified_success"]
    assert result["truncated"] is False  # the environment ended the episode (done at step_limit)
    rewards = check_telescoping(result, terms)
    assert sum(rewards) == pytest.approx(-result["outcome"]["cost"], abs=1e-9)
    per_step = env._spec.action_price + env._spec.compute_price
    assert all(r == pytest.approx(-per_step, abs=1e-12) for r in rewards)  # no terminal reward or penalty


def test_decision_cap_truncation_telescopes_but_is_treated_as_terminal():
    seed = DEV + 5
    think = Action("think")
    indices = script(seed, lambda o: think, limit=10)
    env, result, terms, _ = rollout(seed, indices, max_steps=10)
    assert result["truncated"] is True and not env.observe().done
    rewards = check_telescoping(result, terms)
    assert sum(rewards) == pytest.approx(-10 * (env._spec.action_price + env._spec.compute_price), abs=1e-12)
    # Terminal handling: the last decision's return-to-go is its own reward (no bootstrap of the
    # remaining episode value). P1/P2a use max_steps == step_limit, so this path does not occur there.
    assert returns_to_go(terms)[-1] == rewards[-1]


def test_partial_progress_then_cap_forfeits_terminal_reward():
    """A P1-style collapse loop: productive steps, then an accepted no-effect loop to the cap.
    Rewards telescope to -cost; the loop steps each cost one action + one forward."""
    seed = DEV + 6
    ref = DepReference("reuse")
    env = make_env(seed)
    o, prefix = env.observe(), []
    while not o.done and o.assignment is None:
        a = ref.choose(o)
        prefix.append(action_catalog(o).index(a))
        o = env.step(a)
    record = o.records[0]["handle"]
    retrieve = Action("retrieve", {"handle": record})

    replay, count = DepReference("reuse"), [0]

    def choose(obs):
        count[0] += 1
        return replay.choose(obs) if count[0] <= len(prefix) else retrieve
    indices = script(seed, choose)
    env, result, terms, _ = rollout(seed, indices, max_steps=96)
    assert not result["outcome"]["verified_success"] and result["outcome"]["steps"] == 96
    rewards = check_telescoping(result, terms)
    loop = rewards[len(prefix) + 1:]
    assert loop and all(r == pytest.approx(-(env._spec.action_price + env._spec.compute_price), abs=1e-12)
                        for r in loop)


def test_abstain_is_terminal_with_negative_cost_return():
    seed = DEV + 7
    indices = script(seed, lambda o: Action("abstain"))
    env, result, terms, _ = rollout(seed, indices)
    assert len(terms) == 1 and not result["truncated"] and not result["outcome"]["verified_success"]
    check_telescoping(result, terms)
