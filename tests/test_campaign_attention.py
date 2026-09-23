import torch
from topoformer.campaign_attention import RoutingModel, RoutingBatch, generate, targets, permute_nodes, corrupt, metrics


def test_exact_oracle_and_relation_order():
    b=generate(64,16,4,seed=9)
    bi=torch.arange(64)
    pointer=b.starts.clone()
    for r in b.relations.unbind(1): pointer=b.adjacency[bi,r].argmax(-1)[bi,pointer]
    assert torch.equal(targets(b)[:,-1][bi,b.starts],b.values[bi,pointer])
    rev=RoutingBatch(b.keys,b.values,b.adjacency,b.relations.flip(1),b.starts)
    assert (targets(b)[:,-1]!=targets(rev)[:,-1]).float().mean()>.4


def test_permutation_and_zero_strength():
    # Width32 is a mechanical tensor fixture, not an experimental configuration.
    torch.manual_seed(1)
    m=RoutingModel(width=32,heads=2)
    b=generate(3,7,3,seed=5)
    order=torch.rand(3,7).argsort(-1)
    bp=permute_nodes(b,order)
    bi=torch.arange(3)[:,None]
    for mode in ('soft','hard','message','context','none'):
        x=m(b,mode)['logits']; y=m(bp,mode)['logits']
        assert torch.allclose(x.gather(2,order[:,None,:,None].expand(-1,3,-1,16)),y,atol=2e-5)
    assert torch.equal(m(b,'none')['logits'],m(b,'soft',zero_strength=True)['logits'])


def test_gradients_and_no_target_forward():
    m=RoutingModel(width=32,heads=2)
    b=generate(2,8,2,seed=42)
    out=m(b,'soft')
    torch.nn.functional.cross_entropy(out['logits'].flatten(0,2),targets(b).flatten()).backward()
    assert m.strength.grad.abs().sum()>0
    assert m.q.weight.grad.abs().sum()>0
    assert set(b.__dict__)=={'keys','values','adjacency','relations','starts'}


def test_degree_and_corruption_contracts():
    b=generate(4,16,4,seed=123)
    w=corrupt(b,'wrong',32)
    assert torch.equal(b.adjacency.sum(-1),w.adjacency.sum(-1))
    assert torch.equal(b.adjacency.sum(-2),w.adjacency.sum(-2))
    for mode in ('missing','spurious','identity'):
        c=corrupt(b,mode,77)
        assert (c.adjacency.sum(-1)>0).all()
        assert torch.equal(c.values,b.values)


def test_hard_message_routing_path():
    m=RoutingModel(width=32,heads=2)
    b=generate(4,16,8,seed=42)
    hard=m(b,'hard'); message=m(b,'message')
    assert torch.allclose(hard['logits'],message['logits'],atol=1e-6)
    assert metrics(hard,targets(b),b)['exact_pointer_path'].all()


def test_corrupted_routing_scored_against_clean_graph():
    b=generate(32,16,4,seed=211)
    c=corrupt(b,'wrong',55)
    m=RoutingModel(width=32,heads=2)
    result=m(c,'message')
    clean=metrics(result,targets(b),b)
    supplied=metrics(result,targets(c),c)
    assert supplied['exact_pointer_path'].all()
    assert clean['exact_pointer_path'].float().mean()<.1
    assert clean['edge_mass'].mean()<.2


def test_size_strength_is_public_and_zero_still_exact():
    b=generate(2,32,2,seed=777)
    m=RoutingModel(width=32,heads=2)
    assert torch.equal(m(b,'soft',zero_strength=True,size_adjust=True)['logits'],m(b,'none')['logits'])
    assert m(b,'soft',strength_override=8.)['logits'].shape==(2,2,32,16)


def test_equal_content_mass_fixture():
    import math
    b=generate(2,16,1,seed=100)
    m=RoutingModel(width=32,heads=2,strength=4.)
    with torch.no_grad():
        m.q.weight.zero_();m.k.weight.zero_()
    out=m(b,'soft')
    expected=math.exp(4)/(math.exp(4)+15)
    assert torch.allclose(out['edge_mass'],torch.full_like(out['edge_mass'],expected),atol=1e-6)


def test_restored_permutation_matches_all_metrics():
    from topoformer.campaign_attention import restore_node_order
    b=generate(2,16,4,seed=1235)
    order=torch.rand(2,16).argsort(-1)
    model=RoutingModel(width=32,heads=2)
    for mode in ('soft','context','message','hard','none'):
        original=model(b,mode)
        restored=restore_node_order(model(permute_nodes(b,order),mode),order)
        for key in ('logits','weights','edge_mass'):
            assert torch.allclose(original[key],restored[key],atol=2e-5)
        # Exact argmax ties among duplicate payloads may select another row in
        # none/soft. Do not falsely demand equivariant tie-breaking.
        if mode in ('context','message','hard'):
            assert torch.equal(original['routes'],restored['routes'])


def test_runner_keeps_confirmation_out_of_curves(tmp_path):
    import json
    from topoformer.campaign_attention_study import run
    cfg={'mode':'soft','width':32,'seed':9,'train_seed':100,'eval_seed':200,
         'curve_seed':300,'curve_examples':2,'curve_conditions':[{'nodes':4,'depth':1}],
         'steps':1,'checkpoints':[],'batch':2,'nodes':4,'eval_batch':2,'eval_examples':4,
         'lr':.0003,'device':'cpu','conditions':[{'nodes':8,'depth':3,'node_permutation':True}]}
    out=tmp_path/'run'
    run(cfg,out)
    before=json.loads((out/'eval-00000.json').read_text())
    after=json.loads((out/'eval-00001.json').read_text())
    assert before['eval_seed']==300 and before['eval_examples']==2
    assert after['eval_seed']==200 and after['eval_examples']==4
    assert after['rows'][0]['forward_seconds']>=0
