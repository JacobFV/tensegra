import torch
from tensegra.campaign_semantics_identity_contract import derive_refers_to
from tensegra.thinking_language import KINDS


def test_rule_uses_only_predicted_presence_type_and_identity():
    p=dict(presence=torch.tensor([1,1,1,1,0],dtype=torch.bool),kind=torch.tensor([KINDS.index(k) for k in ['ident','entity','entity','ident','entity']]),copy=torch.tensor([7,7,8,9,7]))
    got=derive_refers_to(p)
    assert got.nonzero().tolist()==[[0,1]]
    p['copy'][2]=7
    assert derive_refers_to(p).nonzero().tolist()==[[0,1],[0,2]]
