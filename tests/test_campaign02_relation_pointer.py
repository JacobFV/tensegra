"""Tiny CPU fixtures for the RL02 pointer extension; not width1024 experiments."""
import json
from pathlib import Path

import pytest
import torch

from tensegra.campaign02_relation import relation_loss
from tensegra.campaign02_relation_pointer import (
    ARMS, MASK, PointerRecordActor, build_optimizer, historical_state_hash,
    load_historical, source_bindings, verify_config,
)
from tensegra.campaign_semantics_s19_actor import TypedRecordActor
from tensegra.campaign_semantics_s19_codec import EDGE, EOS, KINDS, NODE, PAD
from tensegra.campaign_semantics_s22_controller import NodePrefix, decode
from tensegra.thinking_language import ActorInput, state_hash

DIMS = dict(value_count=4, width=16, heads=2, node_capacity=8, max_records=24, max_slot=4)
PRED = KINDS.index('pred')
PUBLICS = [ActorInput('alpha likes beta and gamma', ()), ActorInput('delta sees epsilon', ())]


def records():
    """Two rows: 3 and 2 leading NODEs, then EDGEs, EOS, PAD."""
    rows = [
        [[NODE, PRED, 0, -1, -1], [NODE, PRED, 1, -1, -1], [NODE, PRED, 2, -1, -1],
         [EDGE, 0, 2, 0, -1], [EDGE, 1, 0, 3, 0], [EDGE, 2, 1, 3, 1], [EOS, -1, -1, -1, -1]],
        [[NODE, PRED, 3, -1, -1], [NODE, PRED, 1, -1, -1], [EDGE, 1, 0, 0, -1],
         [EOS, -1, -1, -1, -1], [PAD, -1, -1, -1, -1], [PAD, -1, -1, -1, -1], [PAD, -1, -1, -1, -1]],
    ]
    return torch.tensor(rows, dtype=torch.long)


def pointer_model(mode='replace', seed=0):
    torch.manual_seed(seed)
    return PointerRecordActor(pointer_init_seed=seed + 7, pointer_mode=mode, **DIMS).eval()


def test_disabled_pointer_is_historical_actor():
    torch.manual_seed(3)
    historical = TypedRecordActor(**DIMS).eval()
    disabled = PointerRecordActor(pointer_fields=(), **DIMS).eval()
    disabled.load_state_dict(historical.state_dict())
    assert state_hash(disabled) == state_hash(historical) == historical_state_hash(historical)
    assert set(disabled.state_dict()) == set(historical.state_dict())
    r = records()
    a, b = historical.teacher_forced(PUBLICS, r), disabled.teacher_forced(PUBLICS, r)
    for k in a:
        assert torch.equal(a[k], b[k])
    prefixes = [NodePrefix(tuple(tuple(x) for x in r[0, :3].tolist())), NodePrefix(tuple(tuple(x) for x in r[1, :2].tolist()))]
    assert decode(historical, PUBLICS, 'oracle_node_prefix', prefixes) == decode(disabled, PUBLICS, 'oracle_node_prefix', prefixes)
    assert decode(historical, PUBLICS) == decode(disabled, PUBLICS)


def test_pointer_arm_loads_historical_state_and_keeps_its_hash():
    torch.manual_seed(3)
    historical = TypedRecordActor(**DIMS)
    model = pointer_model()
    load_historical(model, historical.state_dict())
    assert historical_state_hash(model) == state_hash(historical)
    assert state_hash(model) != state_hash(historical)
    bad = dict(historical.state_dict()); bad.pop('target_head.weight')
    with pytest.raises(ValueError):
        load_historical(pointer_model(), bad)


def test_pointer_init_is_seeded_and_does_not_consume_global_rng():
    torch.manual_seed(5); a = PointerRecordActor(pointer_init_seed=11, **DIMS); after_a = torch.rand(1)
    torch.manual_seed(5); b = PointerRecordActor(pointer_init_seed=11, **DIMS); after_b = torch.rand(1)
    torch.manual_seed(5); TypedRecordActor(**DIMS); after_h = torch.rand(1)
    assert torch.equal(after_a, after_b) and torch.equal(after_a, after_h)
    for p, q in zip(a.pointer_parameters(), b.pointer_parameters()):
        assert torch.equal(p, q)


@pytest.mark.parametrize('mode', ['replace', 'residual'])
def test_mask_admits_only_previous_node_records(mode):
    model = pointer_model(mode)
    r = records()
    logits = model.teacher_forced(PUBLICS, r)
    # Output t sees inputs BOS,r[0..t-1]; admissible classes = #NODEs among r[0..t-1].
    for b in range(2):
        nodes_before = 0
        for t in range(r.shape[1]):
            for field in ('source', 'target'):
                row = logits[field][b, t]
                assert torch.all(row[nodes_before:] == MASK)
                assert torch.all(row[:nodes_before] > MASK / 2)
            nodes_before += int(r[b, t, 0] == NODE)
    # Non-node inputs never add classes: row 0 has 3 NODEs and 3 EDGEs before EOS.
    assert torch.all(logits['target'][0, 6, 3:] == MASK)


def test_pointer_ignores_future_records_and_scores_node_keys():
    model = pointer_model()
    r = records()
    base = model.teacher_forced(PUBLICS, r)
    changed = r.clone()
    changed[0, 4:6] = torch.tensor([[EDGE, 2, 2, 5, 2], [EDGE, 0, 0, 4, -1]])
    later = model.teacher_forced(PUBLICS, changed)
    for field in ('source', 'target'):
        torch.testing.assert_close(base[field][:, :5], later[field][:, :5])  # outputs t<=4 read inputs < 5
        assert not torch.allclose(base[field][0, 5:7], later[field][0, 5:7])
    # Manual score for row 0, output position 3 (first EDGE), node class 1 = input position 2.
    captured = {}
    hook = model.decoder_norm.register_forward_hook(lambda m, i, o: captured.setdefault('h', o))
    model.teacher_forced(PUBLICS, r)
    hook.remove()
    h = captured['h']
    q = model.pointer_query['target'](h[0, 3]); k = model.pointer_key(h[0, 2])
    torch.testing.assert_close(base['target'][0, 3, 1], (q @ k) / model.pointer_dim ** .5)


def test_incremental_cache_matches_teacher_forcing():
    model = pointer_model()
    r = records()
    full = model.teacher_forced(PUBLICS, r)
    cache = model.begin(PUBLICS)
    previous = torch.full((2, 5), -1, dtype=torch.long); previous[:, 0] = 0
    for t in range(r.shape[1]):
        step, cache = model.step(previous, cache)
        for field in ('type', 'source', 'target', 'role'):
            torch.testing.assert_close(step[field], full[field][:, t], rtol=1e-4, atol=1e-4)
        previous = r[:, t]


def test_gradients_reach_pointer_not_replaced_heads():
    model = pointer_model().train()
    r = records()
    loss, sums, counts, _ = relation_loss(model.teacher_forced(PUBLICS, r), r, 'all_records')
    assert torch.isfinite(loss) and counts['target'] == 4
    loss.backward()
    assert model.pointer_key.weight.grad.abs().sum() > 0
    for field in ('source', 'target'):
        assert model.pointer_query[field].weight.grad.abs().sum() > 0
        assert getattr(model, field + '_head').weight.grad is None
    assert model.decoder[0].ff[0].weight.grad.abs().sum() > 0  # backbone still trained


def test_optimizer_groups():
    c = dict(learning_rate=3e-5, pointer=dict(learning_rate=3e-4))
    model = pointer_model()
    groups = build_optimizer(model, c).param_groups
    assert [g['lr'] for g in groups] == [3e-5, 3e-4]
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))
    assert len(build_optimizer(TypedRecordActor(**DIMS), c).param_groups) == 1


def test_prefix_decoding_only_emits_valid_node_references():
    r = records()
    prefixes = [NodePrefix(tuple(tuple(x) for x in r[0, :3].tolist())), NodePrefix(tuple(tuple(x) for x in r[1, :2].tolist()))]
    edges = 0
    for seed in range(6):
        model = pointer_model(seed=seed)
        with torch.no_grad():  # push the controller toward long EDGE runs
            model.type_head.bias.copy_(torch.tensor([0., 50., -50.]))
        for out, prefix in zip(decode(model, PUBLICS, 'oracle_node_prefix', prefixes), prefixes):
            n = len(prefix.records)
            assert out['status'] != 'invalid_generated_node_reference'
            for rec in out['records'][n:]:
                if rec[0] == EDGE:
                    edges += 1
                    assert 0 <= rec[1] < n and 0 <= rec[2] < n
    assert edges > 20


def test_committed_configs_bind_current_sources():
    root = Path(__file__).resolve().parents[1] / 'configs' / 'campaign02'
    bindings = source_bindings()
    for name in ('rl02-profile.json', 'rl02-main.json'):
        c = json.loads((root / name).read_text())
        assert c['source_sha256'] == bindings
        verify_config(c)
        assert c['arms'] == list(ARMS)
    main = json.loads((root / 'rl02-main.json').read_text())
    rl01 = json.loads((root / 'rl01-main.json').read_text())
    for key in ('inputs', 'parent', 'parent_state_sha256', 'seed', 'schedule_seed', 'learning_rate',
                'batch_size', 'eval_batch_size', 'updates', 'checkpoints', 'width'):
        assert main[key] == rl01[key]
    bad = dict(main, pointer=dict(main['pointer'], mode='residual'))
    with pytest.raises(ValueError):
        verify_config(bad)
