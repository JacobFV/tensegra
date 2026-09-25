"""Width16 mechanical parity fixture; all research actors remain width1024."""
import torch
from tensegra.semantic_curriculum import SemanticCurriculumActor
from tensegra.thinking_language import ActorInput
from tensegra.campaign_semantics_edge_probe import edge_scores


def test_frozen_node_readout_preserves_bfloat16_dense_and_sampled_scores():
    torch.manual_seed(17)
    model=SemanticCurriculumActor(value_count=4,width=16,capacity=128,workspace_rows=8,microsteps=2,edge_width=4,autocast_dtype='bfloat16').eval()
    seen=[]
    hook=model.decode_nodes.register_forward_hook(lambda m,args,out:seen.append((out[0]+model.queries).detach()))
    public=ActorInput('parent alice bob parent bob alice',())
    pairs=torch.tensor([[0,1],[2,3],[17,9],[127,127]])
    with torch.no_grad():
        original=model(public);nodes=seen.pop()
        assert torch.equal(original['edges'],edge_scores(model,nodes)[0])
        sampled=model(public,pairs=pairs);seen.pop()
        assert torch.equal(sampled['edges'],edge_scores(model,nodes,[pairs])[0])
    hook.remove()
