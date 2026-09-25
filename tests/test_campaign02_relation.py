"""Tiny CPU mathematical fixtures, not width1024 acquisition experiments."""
import copy

import pytest
import torch

from tensegra.campaign02_relation import (
    ARMS, KNOWN_COUNTS, TRAIN_COUNTS, node_prefix, relation_loss,
    teacher_relation_rows, verify_sources,
)
from tensegra.campaign_semantics_s19 import field_losses
from tensegra.campaign_semantics_s19_codec import EDGE, EOS, NODE, PAD, KINDS
from tensegra.campaign_semantics_s22_controller import NodePrefix


def fixture():
    records = torch.tensor([[
        [NODE, KINDS.index('ident'), -1, 1, -1],
        [NODE, KINDS.index('pred'), 0, -1, -1],
        [EDGE, 1, 0, 0, 0],
        [EOS, -1, -1, -1, -1],
        [PAD, -1, -1, -1, -1],
    ]])
    classes = dict(type=3, kind=len(KINDS), value=4, copy=4,
                   source=4, target=4, role=4, slot=4)
    logits = {k:torch.randn(1, 5, n, requires_grad=True) for k,n in classes.items()}
    return logits, records


def test_all_record_objective_is_historical():
    logits, records = fixture()
    actual = relation_loss(logits, records, 'all_records')
    expected = field_losses(logits, records)
    assert torch.equal(actual[0], expected[0])
    assert actual[2] == expected[2]


def test_relation_support_and_eight_field_denominator():
    logits, records = fixture()
    loss, sums, counts, _ = relation_loss(logits, records, 'relation_only')
    assert counts == dict(type=2, kind=0, value=0, copy=0, source=1, target=1, role=1, slot=1)
    expected = (sums['type']/2 + sum(sums[k] for k in ('source','target','role','slot'))) / 8
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for k in ('kind','value','copy'):
        assert torch.count_nonzero(logits[k].grad) == 0
    assert torch.count_nonzero(logits['type'].grad[:, :2]) == 0
    assert torch.count_nonzero(logits['type'].grad[:, 2:4]) > 0
    assert torch.count_nonzero(logits['type'].grad[:, 4:]) == 0


def test_ordered_tuple_rejects_reversed_arguments():
    logits, records = fixture()
    for name, label in dict(type=EDGE-1, source=1, target=0, role=0, slot=1).items():
        logits[name].data[0, 2].fill_(-10)
        logits[name].data[0, 2, label] = 10
    logits['type'].data[0, 3].fill_(-10)
    logits['type'].data[0, 3, EOS-1] = 10
    row = teacher_relation_rows(logits, records)[0]
    assert row[0]['correct'] and row[1]['correct']
    logits['source'].data[0, 2].fill_(-10)
    logits['source'].data[0, 2, 0] = 10
    logits['target'].data[0, 2].fill_(-10)
    logits['target'].data[0, 2, 1] = 10
    assert not teacher_relation_rows(logits, records)[0][0]['correct']


def test_prefix_has_no_edge_or_termination_information():
    _, records = fixture()
    rows = records[0].tolist()
    prefix = node_prefix(rows)
    assert type(prefix) is NodePrefix and len(prefix.records) == 2
    altered = copy.deepcopy(rows)
    altered[2:4] = [[EDGE, 0, 1, 2, 2], [EDGE, 1, 0, 3, 1], [EOS,-1,-1,-1,-1]]
    assert node_prefix(altered) == prefix


def test_prefix_rejects_noncanonical_nodes():
    _, records = fixture()
    rows = records[0].tolist()
    with pytest.raises(ValueError):
        node_prefix([rows[2], rows[0], rows[3]])


def test_manifest_contract_populations():
    assert sum(TRAIN_COUNTS.values()) == 1024
    assert sum(KNOWN_COUNTS.values()) == 512
    assert (3,4) not in TRAIN_COUNTS and set(KNOWN_COUNTS) == set(TRAIN_COUNTS)
    assert ARMS == ('all_records','relation_only')


def test_unknown_objective_and_unfrozen_source_rejected():
    logits, records = fixture()
    with pytest.raises(ValueError):
        relation_loss(logits, records, 'five_field_rescale')
    with pytest.raises(ValueError):
        verify_sources({'source_sha256':{}})
