import torch
from topoformer.return_memory import ReturnMemoryModel, FIELDS
from topoformer.retention_data import make_batch
from topoformer.return_memory_study import gate


def model():
    torch.manual_seed(7)
    return ReturnMemoryModel(width=24,heads=4,feature_dim=8)


def test_default_and_capacity_contract():
    import inspect
    assert inspect.signature(ReturnMemoryModel).parameters['width'].default == 1024
    m=model(); b=make_batch(1,2,8)
    assert sum(m.sizes)==24
    assert m.encode(b['public']['event'],'compressed').shape==(2,1,24)
    assert m.encode(b['public']['event'],'factorized').shape==(2,6,24)
    assert set(m(b['public'],0)['logits'])==set(FIELDS)


def test_availability_zero_step_identical_and_gradients():
    m=model(); b=make_batch(2,2,8)
    a=m(b['public'],0,availability='once'); c=m(b['public'],0,availability='persistent')
    assert torch.equal(a['state'],c['state'])
    out=m(b['public'],1)
    sum(v.square().mean() for v in out['logits'].values()).backward()
    assert all(p.grad is not None for enc in m.encoders for p in enc.parameters())
    assert all(block.qkv.weight.grad is not None for block in m.blocks)


def test_lifecycle_changes_only_protected_record():
    m=model(); b=make_batch(3,2,8); replacement=make_batch(4,2,8)['public']['event']
    for command in ('release','overwrite'):
        out=m(b['public'],2,intervention=command,replacement_event=replacement)
        assert torch.equal(out['lifecycle_before'],out['lifecycle_after'])
        if command=='release': assert out['register'] is None
        else: assert torch.equal(out['register']['values'],replacement['values'])
    assert torch.equal(b['public']['event']['values'],make_batch(3,2,8)['public']['event']['values'])


def test_labels_not_actor_input_and_interventions_valid():
    m=model(); b=make_batch(5,2,8)
    baseline=m(b['public'],1)
    b['targets']={k:torch.zeros_like(v) for k,v in b['targets'].items()}
    assert torch.equal(baseline['state'],m(b['public'],1)['state'])
    assert m(b['public'],1,intervention='event_drop')['register'] is None
    for intervention,key in [('wrong_value','values'),('wrong_type','types'),('wrong_provenance','provenance')]:
        out=m(b['public'],0,intervention=intervention)
        assert not torch.equal(out['register'][key],b['public']['event'][key])
    assert not gate([],[],[])
    assert not gate([],[0],[2])


def test_mixed_encoder_and_no_recurrence_at_zero():
    m=ReturnMemoryModel(width=24,heads=4,feature_dim=8,encoding='mixed')
    b=make_batch(1,2,8)
    assert m.encode(b['public']['event'],'mixed').shape==(2,1,24)
    handles=[block.register_forward_hook(lambda *args: (_ for _ in ()).throw(AssertionError('zero step recurrence'))) for block in m.blocks]
    m(b['public'],0,encoding='mixed')
    for handle in handles: handle.remove()


def test_paired_common_backbone_identical_across_encodings():
    states=[]
    for encoding in ('mixed','compressed','factorized'):
        torch.manual_seed(123)
        states.append(ReturnMemoryModel(width=24,heads=4,feature_dim=8,encoding=encoding).state_dict())
    for key in states[0]:
        if not key.startswith('encoders.'):
            assert torch.equal(states[0][key],states[1][key])
            assert torch.equal(states[0][key],states[2][key])
    assert all(torch.equal(states[1][key],states[2][key]) for key in states[1])


def test_chunked_eval_matches_whole_batch():
    from topoformer.return_memory_study import predict_chunked, counts
    m=model(); b=make_batch(7,5,8)
    with torch.no_grad():
        whole=m(b['public'],1)
        chunked=predict_chunked(m,b['public'],1,'factorized','persistent',batch_size=2)
    for key in FIELDS: torch.testing.assert_close(whole['logits'][key],chunked['logits'][key],atol=1e-5,rtol=1e-5)
    assert counts(whole,b['targets'])==counts(chunked,b['targets'])
