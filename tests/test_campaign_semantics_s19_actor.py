"""S19 width16 CPU mechanics; plus bounded synthetic width1024 prefix checks."""
import copy
import pytest
import torch
from topoformer.campaign_semantics_s19_actor import TypedRecordActor,BOS,NODE,EDGE,EOS,PAD
from topoformer.thinking_language import ActorInput,KINDS


def actor(**kwargs):
    torch.manual_seed(1901)
    return TypedRecordActor(value_count=4,width=16,heads=4,node_capacity=8,max_records=12,**kwargs)


def example():
    public=ActorInput('parent alice bob',())
    records=torch.tensor([[[NODE,KINDS.index('ident'),-1,1,-1],
        [NODE,KINDS.index('record'),1,-1,-1],[EDGE,1,0,3,0],[EOS,-1,-1,-1,-1]]])
    bos=torch.full((1,1,5),-1,dtype=torch.long);bos[:,:,0]=BOS
    return public,records,torch.cat((bos,records[:,:-1]),1)


@pytest.mark.parametrize('dtype,tol',[(None,3e-6),('bfloat16',.035)])
def test_cached_prefix_equals_full_teacher_forced(dtype,tol):
    m=actor(autocast_dtype=dtype).eval();public,targets,previous=example()
    full=m([public],previous);shifted=m.teacher_forced([public],targets)
    for k in full:assert torch.equal(full[k],shifted[k])
    cache=m.begin([public]);cross_ids=[(id(k),id(v)) for k,v in cache.cross_kv]
    for t in range(previous.shape[1]):
        step,cache=m.step(previous[:,t],cache)
        assert cache.position==t+1 and all(k.shape[2]==t+1 for k,v in cache.self_kv)
        assert [(id(k),id(v)) for k,v in cache.cross_kv]==cross_ids
        for k in full:assert torch.allclose(full[k][:,t],step[k],atol=tol,rtol=tol),k


def test_future_and_current_targets_cannot_change_earlier_predictions():
    m=actor().eval();public,targets,_=example();changed=targets.clone()
    changed[:,1]=torch.tensor([NODE,KINDS.index('entity'),-1,2,-1]);changed[:,2]=torch.tensor([EDGE,0,1,2,-1])
    original=m.teacher_forced([public],targets);other=m.teacher_forced([public],changed)
    for k in original:assert torch.equal(original[k][:,:2],other[k][:,:2])
    assert any(not torch.equal(original[k][:,2:],other[k][:,2:]) for k in original)


def test_padding_batch_invariance_and_public_copy_mask():
    m=actor().eval();public,targets,previous=example();long=ActorInput('parent alice bob carol dave erin frank',())
    alone=m([public],previous);batch=m([public,long],previous.expand(2,-1,-1).clone())
    for k in alone:
        b=batch[k][:1,:,:alone[k].shape[-1]]
        assert torch.allclose(alone[k],b,atol=4e-6,rtol=4e-6),k
    assert torch.isneginf(batch['copy'][0,:,3:]).all()
    padded=torch.cat((previous,torch.tensor([[[PAD,-1,-1,-1,-1]]])),1)
    out=m([public],padded)
    for k in alone:assert torch.allclose(out[k][:,:4],alone[k],atol=3e-6,rtol=3e-6)
    cache=m.begin([public,long]);bad=torch.tensor([[NODE,5,-1,5,-1],[NODE,5,-1,5,-1]])
    bos=torch.tensor([[BOS,-1,-1,-1,-1]]*2);_,cache=m.step(bos,cache)
    with pytest.raises(ValueError,match='padding'):m.step(bad,cache)


def test_all_layers_pointer_and_previous_copy_are_differentiable():
    m=actor();public,targets,_=example();out=m.teacher_forced([public],targets)
    loss=sum(x.square().mean() for x in out.values());loss.backward()
    for group in (m.encoder,m.decoder):
        for layer in group:assert sum(float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None)>0
    for component in (m.features,m.copy_query,m.copy_key,m.copy_input,m.type_head,m.kind_head,m.source_head,m.target_head,m.role_head,m.slot_head):
        assert component.weight.grad is not None and torch.isfinite(component.weight.grad).all() and component.weight.grad.abs().sum()>0
    assert m.encoder[0].attention.q.weight is not m.encoder[1].attention.q.weight
    assert not torch.equal(m.encoder[0].attention.q.weight,m.encoder[1].attention.q.weight)


def force(m,**heads):
    with torch.no_grad():
        for name,index in heads.items():
            head=getattr(m,name+'_head');head.weight.zero_();head.bias.fill_(-100);head.bias[index]=100


def test_greedy_public_only_eos_and_malformed_failures():
    m=actor();public=ActorInput('alice bob',());force(m,type=2)
    assert m.type_head.out_features==3 # BOS/PAD never legitimate generated classes.
    result=m.greedy([public])[0];assert result['status']=='eos' and result['records']==[[EOS,-1,-1,-1,-1]]
    assert m.training # inference restores caller mode.
    force(m,type=1,source=0,target=0)
    result=m.greedy([public])[0];assert result['status']=='invalid_generated_node_reference'
    force(m,type=0,kind=KINDS.index('record'),value=1)
    result=m.greedy([public])[0];assert result['status']=='node_capacity_overflow' and result['node_count']==8
    force(m,value=0)
    assert m.greedy([public])[0]['records'][0][2]==0
    assert m.value_head.out_features==4 and not hasattr(m,'copy_none')
    with pytest.raises(TypeError):m.greedy([{'text':'alice','graph':'poison'}])
    with pytest.raises(TypeError):m.greedy([public],gold_count=1)


def test_record_overflow_and_prefix_guards():
    m=actor();public,targets,previous=example();m.max_records=2
    force(m,type=0,kind=KINDS.index('record'),value=1)
    assert m.greedy([public])[0]['status']=='record_capacity_overflow'
    with pytest.raises(ValueError,match='capacity'):m([public],previous)
    with pytest.raises(ValueError,match='BOS'):m([public],targets[:,:1])
    with pytest.raises(ValueError,match='int64'):m([public],previous[:,:1].float())


def test_primary_dimensions_parameter_count_without_model_forward():
    with torch.device('meta'):
        m=TypedRecordActor(value_count=4)
        assert len(m.encoder)==len(m.decoder)==2 and m.width==1024 and m.node_capacity==128 and m.max_records==160
        assert all(l.attention.heads==8 for l in m.encoder)
        assert all(l.ff[0].out_features==4096 for l in [*m.encoder,*m.decoder])
        assert not any(isinstance(l,torch.nn.Dropout) for l in m.modules())
        assert m.parameter_count==62_677_315==sum(p.numel() for p in m.parameters())


def test_batched_early_eos_matches_individual_greedy(monkeypatch):
    m=actor().eval();original=m._heads
    def prescribed(x,cache):
        out=original(x,cache)
        # A public-length-based mechanical script forces different stopping
        # times; it does not stand in for the learned actor in any experiment.
        lengths=cache.public_mask.sum(1).tolist()
        for row,length in enumerate(lengths):
            out['type'][row].fill_(-100);out['type'][row,:,2 if cache.position>=length else 0]=100
            out['kind'][row].fill_(-100);out['kind'][row,:,KINDS.index('record')]=100
            out['value'][row].fill_(-100);out['value'][row,:,1]=100
        return out
    monkeypatch.setattr(m,'_heads',prescribed)
    short=ActorInput('alice',());long=ActorInput('alice bob carol',())
    together=m.greedy([short,long]);separate=[m.greedy([p])[0] for p in (short,long)]
    assert together==separate
    assert [len(r['records']) for r in together]==[2,4]
    assert all(r['status']=='eos' for r in together)


def test_incremental_reuses_encoder_cross_and_copy_projections():
    m=actor();public,_,previous=example();counts={};handles=[]
    for name,module in [('features',m.features),('copy_key',m.copy_key),('copy_input',m.copy_input),
        ('cross0',m.decoder[0].cross_attention.k),('cross1',m.decoder[1].cross_attention.k)]:
        counts[name]=0
        def hook(module,args,out,name=name):counts[name]+=1
        handles.append(module.register_forward_hook(hook))
    cache=m.begin([public])
    for t in range(previous.shape[1]):_,cache=m.step(previous[:,t],cache)
    for h in handles:h.remove()
    assert set(counts.values())=={1}


@pytest.mark.parametrize('dtype,tol',[(None,4e-6),('bfloat16',.035)])
@pytest.mark.parametrize('width,heads',[(16,4),(1024,8)])
@torch.no_grad()
def test_cached_prefix_argmax_parity_with_mixed_eos_padding(dtype,tol,width,heads):
    torch.manual_seed(1901)
    m=TypedRecordActor(value_count=4,width=width,heads=heads,autocast_dtype=dtype).eval()
    publics=[ActorInput('alice bob',()),ActorInput('alice bob carol dave',())]
    previous=torch.tensor([[[BOS,-1,-1,-1,-1],[EOS,-1,-1,-1,-1],[PAD,-1,-1,-1,-1]],
        [[BOS,-1,-1,-1,-1],[NODE,KINDS.index('ident'),-1,2,-1],[NODE,KINDS.index('record'),0,-1,-1]]])
    full=m(publics,previous);cache=m.begin(publics)
    for t in range(previous.shape[1]):
        out,cache=m.step(previous[:,t],cache)
        for key in full:
            assert torch.allclose(full[key][:,t],out[key],atol=tol,rtol=tol),key
            assert torch.equal(full[key][:,t].argmax(-1),out[key].argmax(-1)),(width,dtype,t,key)


@torch.no_grad()
def test_public_step_supports_forced_160_step_timing_without_gold_length():
    m=actor().eval();m.max_records=160
    cache=m.begin([ActorInput('alice bob',())])
    previous=torch.tensor([[BOS,-1,-1,-1,-1]])
    for _ in range(160):
        out,cache=m.step(previous,cache)
        # Feed predicted records even past EOS/invalid graph references for timing.
        # No targets, graph sizes, or gold lengths enter this API.
        tag=int(out['type'].argmax(-1))+1
        if tag==NODE:
            kind=int(out['kind'].argmax(-1));lexical=kind in {KINDS.index('ident'),KINDS.index('entity')}
            record=[NODE,kind,-1 if lexical else int(out['value'].argmax(-1)),int(out['copy'].argmax(-1)) if lexical else -1,-1]
        elif tag==EDGE:record=[EDGE,int(out['source'].argmax(-1)),int(out['target'].argmax(-1)),int(out['role'].argmax(-1)),int(out['slot'].argmax(-1))-1]
        else:record=[EOS,-1,-1,-1,-1]
        previous=torch.tensor([record])
    assert cache.position==160


@pytest.mark.parametrize('dtype',[None,'bfloat16'])
@torch.no_grad()
def test_greedy_cached_equals_full_prefix_recomputation(dtype,monkeypatch):
    m=actor(autocast_dtype=dtype).eval()
    publics=[ActorInput('alice bob',()),ActorInput('alice bob carol dave',())]
    cached=m.greedy(publics);history=[];step=m._step
    def full_prefix(previous,cache,*,validate):
        history.append(previous.clone())
        full=m(publics,torch.stack(history,1))
        _,updated=step(previous,cache,validate=validate)
        return {key:value[:,-1] for key,value in full.items()},updated
    monkeypatch.setattr(m,'_step',full_prefix)
    assert m.greedy(publics)==cached
