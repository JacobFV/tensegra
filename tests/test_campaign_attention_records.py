import torch
from dataclasses import replace
from topoformer.campaign_attention_records import RecordAttention,tokenize
from topoformer.campaign_attention_selector import generate,permute_nodes,targets
from topoformer.campaign_attention import restore_node_order


def fixture():
    # Explicitly mechanical: width128,2heads,key8; primary remains1024/8/64.
    torch.manual_seed(4)
    return generate(2,8,3,groups=2,seed=12,key_dim=8,balanced=True),RecordAttention(width=128,key_dim=8,heads=2)


def test_records_public_and_value_independent():
    b,m=fixture();r=tokenize(b)
    altered=replace(b,values=(b.values+1)%16,starts=(b.starts+1)%8,instructions=b.instructions.roll(1,1))
    assert torch.equal(r.features,tokenize(altered).features)
    assert torch.equal(r.destination_keys,tokenize(altered).destination_keys)
    assert r.features.shape==(2,48,19)


def test_independent_edge_and_node_permutation():
    b,m=fixture();r=tokenize(b);order=torch.rand(2,48).argsort(-1)
    a=m(b);z=m(b,records=tokenize(b,order))
    assert torch.allclose(a['logits'],z['logits'],atol=2e-5)
    p=torch.rand(2,8).argsort(-1);changed=permute_nodes(b,p)
    z=restore_node_order(m(changed),p)
    assert torch.allclose(a['logits'],z['logits'],atol=2e-5)


def test_dense_graph_record_routing_and_gradients():
    b,m=fixture();r=tokenize(b);out=m(b,records=r)
    loss=torch.nn.functional.cross_entropy(out['logits'].flatten(0,2),targets(b).flatten());loss.backward()
    assert m.record_q.weight.grad.abs().sum()>0
    assert m.record_k.weight.grad.abs().sum()>0
    # With pretokenized public records, adjacency cannot influence predictions.
    z=m(replace(b,adjacency=torch.zeros_like(b.adjacency)),records=r)
    assert torch.equal(z['logits'],out['logits'])
    assert (out['weights']>0).all()


def test_consistent_key_reassignment_preserves_semantic_targets():
    b,m=fixture();order=torch.rand(2,8).argsort(-1)
    changed=replace(b,keys=b.keys.gather(1,order[...,None].expand(-1,-1,8)))
    assert torch.equal(targets(b),targets(changed))
    # Records reference reassigned endpoint keys, never numeric node positions.
    edge=changed.adjacency.bool().nonzero().reshape(2,48,4)
    bi,_,_,dest=edge.unbind(-1)
    assert torch.equal(tokenize(changed).destination_keys,changed.keys[bi,dest])
