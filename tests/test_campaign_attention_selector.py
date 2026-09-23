"""A04 mechanical fixtures use width32; primary studies remain1024."""
import torch
from torch.nn import functional as F
from topoformer.campaign_attention_selector import generate,targets,oracle_successors,SelectorModel,permute_nodes,swap_instruction,metrics
from topoformer.campaign_attention import restore_node_order

def test_unique_neighbor_and_globally_repeated_attributes():
    b=generate(4,16,3,seed=1)
    assert (b.adjacency.sum(-1)==4).all()
    for i in range(4):
        codes,count=torch.unique(b.attributes[i],dim=0,return_counts=True)
        assert len(codes)==4 and (count==4).all()
        group=(b.attributes[i,:,None,:]==codes[None,:,:]).all(-1).float()
        assert torch.equal(b.adjacency[i]@group,torch.ones(3,16,4))
    s=oracle_successors(b)
    for t in range(3):
        selected=b.attributes.gather(1,s[:,t,:,None].expand(-1,-1,64))
        assert torch.equal(selected,b.instructions[:,t,None].expand_as(selected))

def test_oracle_path_and_instruction_swap():
    b=generate(128,16,3,seed=2);gold=targets(b);s=oracle_successors(b)
    pointer=b.starts;bi=torch.arange(len(pointer))
    for t in range(3):pointer=s[bi,t,pointer]
    assert torch.equal(b.values[bi,pointer],gold[:,-1][bi,b.starts])
    changed=swap_instruction(b)
    assert torch.equal(b.values,changed.values) and torch.equal(b.adjacency,changed.adjacency)
    assert not torch.equal(b.instructions,changed.instructions)
    assert (targets(changed)[:,-1][bi,b.starts]!=gold[:,-1][bi,b.starts]).float().mean()>.5

def test_model_contract_gradients_and_permutation():
    torch.manual_seed(7);b=generate(2,16,2,seed=3);m=SelectorModel(width=32,heads=4)
    for mode in ['soft','hard','context','none']:
        out=m(b,mode);assert out['logits'].shape==(2,2,16,16)
        assert torch.allclose(out['weights'].sum(-1),torch.ones(2,2,16),atol=1e-6)
        order=torch.rand(2,16).argsort(-1)
        restored=restore_node_order(m(permute_nodes(b,order),mode),order)
        assert torch.allclose(out['logits'],restored['logits'],atol=2e-6)
    loss=F.cross_entropy(m(b,'hard')['logits'].flatten(0,2),targets(b).flatten());loss.backward()
    assert m.q.weight.grad.abs().sum()>0 and m.k.weight.grad.abs().sum()>0
    assert torch.allclose(m(b,'soft',zero_strength=True)['logits'],m(b,'none')['logits'])

def test_exact_selected_attention_reference():
    b=generate(3,16,3,seed=4);s=oracle_successors(b).flip(1);gold=targets(b)
    weight=F.one_hot(s,16).float()
    out=dict(logits=F.one_hot(gold,16).float(),weights=weight,routes=s,edge_mass=torch.ones(3,3,16))
    score=metrics(out,gold,b)
    assert all(v.all() for k,v in score.items())

def test_exact_gather_equals_neighbor_masking():
    torch.manual_seed(12);b=generate(2,16,2,seed=8);m=SelectorModel(width=32,heads=4)
    t=1;z=m.attribute_encoder(b.attributes)
    k=F.normalize(m.k(z).reshape(2,16,4,8).transpose(1,2),dim=-1)
    q=F.normalize(m.q(m.attribute_encoder(b.instructions[:,t])).reshape(2,4,1,8),dim=-1)
    score=(q@k.transpose(-1,-2))*m.selector_log_scale.exp()
    a=b.adjacency[torch.arange(2),b.relations[:,t]]
    expected=score.expand(-1,-1,16,-1).masked_fill(~a[:,None].bool(),-torch.inf).softmax(-1).mean(1)
    assert torch.allclose(m(b,'hard')['weights'][:,0],expected,atol=1e-7)

def test_compact_supplied_and_swap_targets(tmp_path):
    import json
    import numpy as np
    from topoformer.campaign_attention_selector_study import run
    cfg=dict(seed=1,width=32,steps=1,batch=2,nodes=16,groups=4,train_seed=1,eval_seed=200,
             eval_examples=8,eval_batch=4,checkpoints=[1],mode='hard',device='cpu',conditions=[
                 dict(nodes=16,depth=2,data_group=0),dict(nodes=16,depth=2,instruction_swap=True,data_group=0),
                 dict(nodes=16,depth=2,wrong_instruction=True,data_group=0)])
    path=tmp_path/'run';run(cfg,path);z=np.load(path/'eval-00001.npz');rows=json.loads((path/'eval-00001.json').read_text())['rows']
    bi=np.arange(8)
    for ci in range(3):
        pred=z[f'c{ci}_pred'][:,-1][bi,z[f'c{ci}_start']]
        assert np.array_equal(pred==z[f'c{ci}_supplied_final_target'],z[f'c{ci}_agreement_supplied_task'])
    assert np.array_equal(z['c1_supplied_final_target'],z['c2_supplied_final_target'])
    assert np.array_equal(z['c0_gold'],z['c2_gold'])
    changed=z['c1_oracle_terminal']!=z['c1_original_terminal']
    assert changed.sum()==rows[1]['changed_terminal_count']
