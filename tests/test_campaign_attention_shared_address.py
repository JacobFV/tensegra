import torch
import pytest
from topoformer.campaign_attention_records import RecordAttention,tokenize
from topoformer.campaign_attention_shared_address import forward,POLICIES
from topoformer.campaign_attention_selector import generate,oracle_successors,metrics,targets


def fixture(primary=False):
    # Mechanical dimensions only; production uses the frozen width1024 model.
    torch.set_num_threads(2);torch.manual_seed(17)
    m=RecordAttention() if primary else RecordAttention(width=128,key_dim=8,heads=2)
    b=generate(2,8,3,groups=2,seed=29,key_dim=64 if primary else 8,balanced=True)
    order=torch.rand(2,48).argsort(-1);r=tokenize(b,order)
    edges=b.adjacency.bool().nonzero().reshape(2,-1,4)[:,:,1:].gather(1,order[...,None].expand(-1,-1,3))
    return m,b,r,edges,oracle_successors(b)


@pytest.mark.parametrize('primary',[False,True])
def test_zero_change_exact_and_all_policies_frozen(primary):
    m,b,r,e,s=fixture(primary);state={k:v.clone() for k,v in m.state_dict().items()}
    reference=m(b,records=r);observed=forward(m,b,r,e,s)
    for key in ['logits','routes','weights','edge_mass']:assert torch.equal(reference[key],observed[key])
    for policy in POLICIES:
        z=forward(m,b,r,e,s,policy)
        assert all(torch.equal(value,m.state_dict()[name]) for name,value in state.items())
        assert torch.allclose(z['weights'].sum(-1),torch.ones_like(z['weights'].sum(-1)))
        if policy!='unchanged':assert z['diagnostics']['used_head_address_discrepancy'].eq(0).all()
        if policy in ['shared_hard','oracle_common']:assert z['diagnostics']['payload_mse_to_destination_argmax'].eq(0).all()
        if policy=='shared_soft':assert torch.allclose(z['weights'],reference['weights'])
        if policy=='shared_hard':assert torch.equal(z['weights'].argmax(-1),reference['routes'])
        if policy=='oracle_common':assert z['diagnostics']['query_used_destination_head_correct'].all()
        assert torch.equal(z['diagnostics']['query_mean_route_correct'][:,:,0].all(1),
                           metrics(z,targets(b),b)['exact_pointer_path'])


def test_posthoc_oracle_does_not_route():
    m,b,r,e,s=fixture()
    # Choose the other public attribute-neighbor solely for falsifying metrics.
    wrong=[]
    for t in range(3):
        adjacency=b.adjacency[torch.arange(2),b.relations[:,t]].clone()
        adjacency.scatter_(2,s[:,t,:,None],0)
        wrong.append(adjacency.argmax(-1))
    wrong=torch.stack(wrong,1)
    for policy in ['unchanged','shared_soft','shared_hard']:
        a=forward(m,b,r,e,s,policy);z=forward(m,b,r,e,wrong,policy)
        assert torch.equal(a['logits'],z['logits'])
        assert torch.equal(a['routes'],z['routes'])
        assert not torch.equal(a['diagnostics']['record_soft_target_mass'],z['diagnostics']['record_soft_target_mass'])


def test_mechanical_export_roundtrip(tmp_path):
    import hashlib,json,numpy as np
    from topoformer.campaign_attention_shared_address_study import run
    m,*_=fixture(True);checkpoint=tmp_path/'checkpoint.pt';torch.save(m.state_dict(),checkpoint)
    tensor=hashlib.sha256(b''.join(x.numpy().tobytes() for x in m.state_dict().values())).hexdigest()
    cfg=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),tensor_sha256=tensor,examples=2,batch=1,data_seed=31,order_seed=32,conditions=[dict(nodes=8,depth=2,groups=2)],device='cpu')
    out=tmp_path/'results';run(cfg,out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['initial_tensor_sha256']==manifest['final_tensor_sha256']==tensor
    assert manifest['optimizer_updates']==0
    for policy in POLICIES:
        raw=np.load(out/f'{policy}.npz');row=json.loads((out/f'{policy}.json').read_text())['rows'][0]
        assert raw['c0_pred'].shape==(2,2,8)
        assert raw['c0_diagnostic_record_soft_target_mass'].shape==(2,2,8)
        assert raw['c0_diagnostic_query_used_destination_head_correct'].shape==(2,2,8)
        assert float(raw['c0_task'].mean())==row['task']
