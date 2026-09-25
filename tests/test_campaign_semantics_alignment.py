import torch
from tensegra.campaign_semantics_alignment import copy_statistics


def test_occurrence_and_identity_are_scored_separately():
    s=copy_statistics(torch.tensor([0.,5.,-2.]),0,1,torch.tensor([0,0,2]))
    assert s['occurrence_correct'] and s['canonical_correct'] and not s['first_correct']
    assert s['identity_mass']>s['occurrence_mass']>s['first_mass']
