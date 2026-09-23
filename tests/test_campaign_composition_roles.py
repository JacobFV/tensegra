import torch
from topoformer.campaign_composition_study import get_data
from topoformer.campaign_composition_roles import reverse_public, reversed_supervision, choose_views


def test_public_swap_and_actual_supervision_match():
    data, public, labels = get_data(dict(seed=908,count=128))
    changed = reverse_public(public)
    evaluated, supplied = reversed_supervision(data['public'])
    assert all(torch.equal(changed[k], evaluated[k]) for k in public)
    assert all(torch.equal(reverse_public(changed)[k], public[k]) for k in public)
    assert all(torch.equal(public[k], changed[k]) for k in public if k != 'instruction_arguments')
    for i,op in enumerate(labels['primitive'].tolist()):
        assert supplied['targets'][i,0] == labels['targets'][i,0]
        if op == 3:
            assert torch.equal(labels['targets'][i], supplied['targets'][i])
        else:
            assert torch.equal(labels['targets'][i,1:].flip(0), supplied['targets'][i,1:])
        if op in (0,2,3):
            assert labels['value'][i] == supplied['value'][i]
            assert labels['task'][i] == supplied['task'][i]
        if op == 1:
            assert supplied['value'][i] == 32-labels['value'][i]
    compare = labels['primitive']==4
    assert torch.equal(supplied['value'][compare], 34-labels['value'][compare])


def test_presentation_coin_preserves_indices_keys_and_unselected_rows():
    data, public, labels = get_data(dict(seed=909,count=16))
    changed, targets = reversed_supervision(data['public'])
    state = torch.get_rng_state().clone()
    coin = torch.rand(16,generator=torch.Generator().manual_seed(77)) < .5
    assert torch.equal(state,torch.get_rng_state())
    mixed = choose_views(public,changed,coin); truth=choose_views(labels,targets,coin)
    assert torch.equal(mixed['keys'],public['keys'])
    for k in public:
        assert torch.equal(mixed[k][~coin],public[k][~coin])
        assert torch.equal(mixed[k][coin],changed[k][coin])
    for k in labels:
        assert torch.equal(truth[k][~coin],labels[k][~coin])
        assert torch.equal(truth[k][coin],targets[k][coin])
