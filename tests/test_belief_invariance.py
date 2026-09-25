"""Small widths below are explicitly mechanical fixtures, not study runs."""
import torch
from tensegra.belief_state import collate, loss, oracle
from tensegra.belief_contracts import contract_episodes
from tensegra.belief_invariance import InvariantBeliefModel, rename_observations, make_training_episodes, ARMS

def test_paired_initialization_inventory():
    states = []
    for arm in ARMS:
        torch.manual_seed(12)
        states.append(InvariantBeliefModel(arm, width=16, inner=32).state_dict())
    for key in states[0]:
        assert all(torch.equal(states[0][key], state[key]) for state in states[1:])

def test_ledger_only_bijection_and_gold_independence():
    model = InvariantBeliefModel(width=16, inner=32)
    for condition in ('clean', 'duplicate', 'long_duplicate', 'retract', 'full_retract'):
        episodes = contract_episodes(3, 81, condition=condition)
        a = collate(episodes)
        b = collate(rename_observations(episodes, 111))
        assert torch.equal(a['targets']['posterior'], oracle(b['public']))
        assert torch.equal(model(a['public'])['logits'], model(b['public'])['logits'])

def test_exact_prior_finite_loss_and_inactive_id_weights():
    model = InvariantBeliefModel(width=16, inner=32)
    batch = collate(contract_episodes(3, 62, condition='full_retract'))
    output = model(batch['public'])
    p = output['logits'].softmax(-1)
    assert torch.equal(p[:, 0], batch['targets']['posterior'][:, 0])
    assert torch.equal(p[:, -1], batch['targets']['posterior'][:, -1])
    total = sum(loss(output, batch).values())
    assert torch.isfinite(total)
    total.backward()
    assert torch.count_nonzero(model.encode.linear.weight.grad[:, -16:]) == 0

def test_training_pair_preserves_contents_and_targets():
    original = make_training_episodes(8, 900, 'raw_correlated', 'retract')
    randomized = make_training_episodes(8, 900, 'raw_randomized', 'retract')
    for a, b in zip(original, randomized):
        assert a['records'] == b['records'] and a['keys'] == b['keys']
        assert a['posterior'] == b['posterior']
        assert [e[1:] for e in a['events']] == [e[1:] for e in b['events']]
        assert b['events'][-1][0] == b['events'][-2][0]
    assert any(e[0] > 4 for b in randomized for e in b['events'])
