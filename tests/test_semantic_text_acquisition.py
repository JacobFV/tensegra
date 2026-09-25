import torch
from tensegra.semantic_text_acquisition import prepare,fixed_corpus,corrected_losses
from tensegra.semantic_curriculum import SemanticCurriculumActor
from tensegra.thinking_language import ActorInput


def test_fixed_public_targets_observable():
    examples=fixed_corpus(2,11000000)
    items,vocab,audit=prepare(examples)
    assert audit['examples']==2 and audit['maxima']['nodes']<128
    assert all(isinstance(public,ActorInput) and not public.options for public,_,_ in items)
    assert all(not gold['value'].eq(0).any() for _,gold,_ in items)


def test_privileged_pair_queries_change_only_loss_readout():
    # Width16 is a declared mechanical fixture, never the primary model.
    torch.manual_seed(1)
    model=SemanticCurriculumActor(value_count=3,width=16,capacity=4,workspace_rows=2,microsteps=1,edge_width=4)
    public=ActorInput('parent alice bob',())
    pairs=torch.tensor([[0,1],[2,3]])
    all_outputs=model(public);selected=model(public,pairs=pairs)
    for key in ('presence','kind','value','copy'):assert torch.equal(all_outputs[key],selected[key])
    assert torch.allclose(all_outputs['edges'][pairs[:,0],pairs[:,1]],selected['edges'],atol=1e-6)
    assert torch.allclose(all_outputs['slots'][pairs[:,0],pairs[:,1]],selected['slots'],atol=1e-6)
