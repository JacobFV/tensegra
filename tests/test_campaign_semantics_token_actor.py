"""Width16 CPU mechanical fixtures only; primary workspace remains1024."""
import torch
from tensegra.campaign_semantics_token_actor import ContextualTokenActor
from tensegra.semantic_curriculum import SemanticCurriculumActor
from tensegra.thinking_language import ActorInput


def test_padding_batch_independence_and_unchanged_common_initialization():
    torch.set_num_threads(2);kwargs=dict(value_count=4,width=16,capacity=8,workspace_rows=8,microsteps=2,edge_width=4)
    torch.manual_seed(21);model=ContextualTokenActor(**kwargs).eval()
    torch.manual_seed(21);original=SemanticCurriculumActor(**kwargs).eval()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in model.state_dict().items())
    assert not model.initial.requires_grad
    short=ActorInput('parent alice bob',());long=ActorInput('parent alice bob carol dan erin frank A B C D E F',())
    with torch.no_grad():single=model(short);batch=model.forward_batch([short,long])[0]
    for key in single:torch.testing.assert_close(single[key],batch[key],atol=3e-6,rtol=3e-5)
    pairs=torch.tensor([[0,1],[2,3],[5,7]])
    with torch.no_grad():sample=model(short,pairs=pairs)
    torch.testing.assert_close(sample['edges'],single['edges'][pairs[:,0],pairs[:,1]],atol=2e-6,rtol=2e-5)
