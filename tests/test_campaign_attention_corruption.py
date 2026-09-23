import torch
from topoformer.campaign_attention_selector import generate,oracle_successors,targets,SelectorModel
from topoformer.campaign_attention_corruption import replace_edges


def test_replacement_keeps_all_regular_contracts():
    b=generate(3,64,4,groups=4,seed=91,balanced=True)
    for fraction,actual in [(0,0),(.1,.125),(.25,.25),(.5,.5)]:
        c=replace_edges(b,fraction,9)
        assert torch.equal(c.adjacency.sum(-1),b.adjacency.sum(-1))
        assert torch.equal(c.adjacency.sum(-2),b.adjacency.sum(-2))
        removed=(b.adjacency*(1-c.adjacency)).sum()/b.adjacency.sum()
        assert float(removed)==actual
        # Every publicly repeated code still occurs once per outgoing row.
        for i in range(3):
            _,ids=torch.unique(b.attributes[i],dim=0,return_inverse=True)
            for code in range(4):assert (c.adjacency[i,...,ids==code].sum(-1)==1).all()
        assert torch.equal(targets(c),targets(replace_edges(b,fraction,9)))


def test_clean_replay_exact_and_labels_unused():
    # Mechanical width32 fixture, not primary experiment.
    b=generate(2,16,2,groups=4,seed=33,balanced=True)
    torch.manual_seed(1);m=SelectorModel(width=32,heads=4)
    assert replace_edges(b,0,7) is b
    for mode in ['soft','hard','context']:
        kw=dict(selector_scale_override=16.,context_scale_override=16. if mode=='context' else None)
        assert torch.equal(m(b,mode,**kw)['logits'],m(replace_edges(b,0,7),mode,**kw)['logits'])
