import pytest
import torch
from topoformer.semantic_contracts import SlotScorer,slot_objective,single_slot_labels
from topoformer.semantic_graph import SemanticGraph,SemanticEdge


def test_additive_xor_margin_identity():
    # Width 4 is an explicitly mathematical fixture, not a primary experiment.
    torch.manual_seed(2); model=SlotScorer(4,2); nodes=torch.eye(4)
    pairs=torch.tensor([[0,2],[0,3],[1,2],[1,3]])
    logits=model(nodes,pairs); margin=logits[:,1]-logits[:,0]
    assert torch.allclose(margin[0]+margin[3],margin[1]+margin[2],atol=1e-6)
    # Positive diagonal and negative off-diagonal strict margins contradict equality.


def test_interaction_constructs_xor():
    model=SlotScorer(4,2,interaction=True,interaction_width=1)
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.pair_source.weight[1]=torch.tensor([1.,-1.,0.,0.])
        model.pair_target.weight[0]=torch.tensor([0.,0.,1.,-1.])
    assert model(torch.eye(4),torch.tensor([[0,2],[0,3],[1,2],[1,3]])).argmax(-1).tolist()==[1,0,0,1]


def test_edge_conditional_unordered_real_edge_and_nonedge():
    logits=torch.randn(3,3,requires_grad=True)
    loss=slot_objective(logits,torch.tensor([2,0,0]),torch.tensor([True,True,False]),edge_conditional=True)
    loss.backward()
    assert logits.grad[:2].abs().sum()>0
    assert logits.grad[2].abs().sum()==0


def test_conflicting_multiedges_rejected_not_overwritten():
    graph=SemanticGraph((),(SemanticEdge('a','b','argument',0),SemanticEdge('a','b','contains',None)),())
    with pytest.raises(ValueError,match='multiple slot'):single_slot_labels(graph)


def test_training_relation_thresholds_ties_and_absent_class():
    from topoformer.semantic_contracts import train_relation_thresholds
    scores=torch.tensor([[0.,3.],[0.,2.],[2.,1.],[3.,0.]])
    labels=torch.tensor([[False,False],[True,False],[True,False],[True,False]])
    threshold,records=train_relation_thresholds(scores,labels)
    assert threshold.tolist()==[-1.,3.]  # tie at ambiguous zero: lowest threshold
    assert [r['train_errors'] for r in records]==[1,0]
    assert not scores[:,1].gt(threshold[1]).any()
