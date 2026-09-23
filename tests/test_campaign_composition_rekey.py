import torch
import pytest
from topoformer.campaign_composition import make_lowering_batch, model_inputs
from topoformer.campaign_composition_rekey import identity_links, rekey, REFERENCES


def test_rekey_preserves_public_identity_graph_and_nonkey_evidence():
    data = make_lowering_batch(814, 12)
    public = model_inputs(data['public'])
    before = {k: v.clone() for k,v in public.items()}
    labels = data['labels'].copy()
    torch.manual_seed(66)
    state = torch.get_rng_state().clone()
    altered = rekey(public, torch.Generator().manual_seed(917))
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k,v in public.items())
    assert labels == data['labels']
    for name in REFERENCES:
        assert torch.equal(identity_links(public)[name], identity_links(altered)[name])
    assert torch.equal((public['keys'] == 0).all(-1), (altered['keys'] == 0).all(-1))
    for name in set(public) - set(REFERENCES) - {'keys'}:
        assert torch.equal(public[name], altered[name])
    assert not torch.equal(public['keys'], altered['keys'])
    repeat = rekey(public, torch.Generator().manual_seed(917))
    assert all(torch.equal(altered[k], repeat[k]) for k in altered)
    assert not torch.equal(altered['keys'], rekey(public, torch.Generator().manual_seed(918))['keys'])


def test_rekey_refuses_broken_identity_graph():
    public = model_inputs(make_lowering_batch(815, 2)['public'])
    broken = dict(public, query_destination=torch.ones_like(public['query_destination']))
    with pytest.raises(ValueError, match='unique public candidate'):
        rekey(broken, torch.Generator().manual_seed(1))
    broken = dict(public, keys=public['keys'].clone())
    broken['keys'][:, 0] = broken['keys'][:, 1]
    with pytest.raises(ValueError, match='unique'):
        identity_links(broken)
