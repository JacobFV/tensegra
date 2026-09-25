import hashlib
import json
import torch
from tensegra.campaign_attention_records import RecordAttention,tokenize
from tensegra.campaign_attention_selector import generate,oracle_successors,SelectorModel
from tensegra.campaign_attention_confirmation_read import forward,POLICIES
from tensegra.campaign_attention_shared_address import forward as a13_forward
from tensegra.campaign_attention_record_diagnostic import forward as a12_forward
from tensegra.campaign_attention_confirmation_study import run,evaluate


def fixture():
    torch.set_num_threads(2);torch.manual_seed(19)
    model=RecordAttention(width=128,key_dim=8,heads=2)
    batch=generate(2,8,3,groups=2,seed=33,key_dim=8,balanced=True)
    order=torch.rand(2,48).argsort(-1);records=tokenize(batch,order)
    edges=batch.adjacency.bool().nonzero().reshape(2,-1,4)[:,:,1:].gather(1,order[...,None].expand(-1,-1,3))
    return model,batch,records,edges,oracle_successors(batch)


def test_all_five_policies_exact_historical_and_bare_equivalence():
    args=fixture();model=args[0];before={k:v.clone() for k,v in model.state_dict().items()}
    for policy in POLICIES:
        expected=a12_forward(*args,policy) if policy=='both_hard' else a13_forward(*args,policy)
        observed=forward(*args,policy);bare=forward(*args,policy,instrument=False)
        for name in ['logits','routes','weights','edge_mass']:
            assert torch.equal(expected[name],observed[name])
            assert torch.equal(bare[name],observed[name])
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_four_nonoracle_policies_ignore_posthoc_oracle():
    m,b,r,e,s=fixture();wrong=[]
    for t in range(3):
        a=b.adjacency[torch.arange(2),b.relations[:,t]].clone();a.scatter_(2,s[:,t,:,None],0);wrong.append(a.argmax(-1))
    wrong=torch.stack(wrong,1)
    for policy in POLICIES[:-1]:
        assert torch.equal(forward(m,b,r,e,s,policy)['logits'],forward(m,b,r,e,wrong,policy)['logits'])


def test_monitor_does_not_change_training_and_reference_public_pairing(tmp_path):
    # Mechanical two-update/export check only, not confirmation performance.
    cfg=dict(kind='train_confirm',seed=19,steps=2,monitor_steps=[1,2],monitor_examples=2,monitor_order_seed=11,
        examples=2,batch=1,data_seed=43,order_seed=44,benchmark=False,conditions=[dict(nodes=8,depth=2,groups=2)],device='cpu')
    first=tmp_path/'first';second=tmp_path/'second';run(cfg,first);run(dict(cfg,monitor_steps=[2]),second)
    a=json.loads((first/'manifest.json').read_text());b=json.loads((second/'manifest.json').read_text())
    assert a['final_tensor_sha256']==b['final_tensor_sha256']
    sa=torch.load(first/'training-state.pt',weights_only=True);sb=torch.load(second/'training-state.pt',weights_only=True)
    assert torch.equal(sa['cpu_rng'],sb['cpu_rng'])
    for key,value in sa['optimizer']['state'].items():
        for field,tensor in value.items():assert torch.equal(tensor,sb['optimizer']['state'][key][field])
    reference=SelectorModel(width=1024);ref=evaluate(reference,cfg,tmp_path/'reference','cpu','soft')
    assert ref['public_sha256']==a['confirmation']['public_sha256']
    assert all(f'c0_input_{name}' in ref['public_sha256'] for name in ['keys','attributes','instructions','adjacency'])
    assert a['confirmation']['record_order_sha256']==b['confirmation']['record_order_sha256']
    assert a['confirmation']['initial_tensor_sha256']==a['confirmation']['final_tensor_sha256']
