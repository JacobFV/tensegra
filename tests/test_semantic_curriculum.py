import inspect
import torch
from topoformer import semantic_curriculum as m


def test_default_width_and_exposure_axes():
    assert inspect.signature(m.SemanticCurriculumActor).parameters['width'].default == 1024
    for n in (3,30):
        schedule=[m.exposure_schedule(i,n) for i in range(12)]
        assert [lang for _,lang in schedule]==['english','spanish']*6
        assert len({i for i,_ in schedule})==min(n,6)
    assert m.curriculum_weights(0)['edges']==0
    assert m.curriculum_weights(2000)['edges']==1


def test_queried_edges_equal_dense_and_gradients():
    torch.set_num_threads(1)
    e=m.base.build_tcn_example('variable_binding',13)
    public,g=m.base.surface_input(e,'english')
    vocab=m.base.value_vocabulary([e]); gold=m.base.targets(g,public,128,vocab)
    model=m.SemanticCurriculumActor(value_count=len(vocab),width=16,microsteps=1,workspace_rows=2,edge_width=8)
    pairs=m.sampled_pairs(gold,torch.Generator().manual_seed(0),8)
    dense=model(public); sparse=model(public,pairs=pairs)
    i,j=pairs.unbind(-1)
    assert torch.allclose(dense['edges'][i,j],sparse['edges'],atol=1e-5)
    assert torch.allclose(dense['slots'][i,j],sparse['slots'],atol=1e-5)
    loss=sum(m.sampled_losses(sparse,gold,pairs).values()); loss.backward()
    assert torch.isfinite(loss)
    assert all(block.ff[0].weight.grad.abs().sum()>0 for block in model.blocks)


def test_noinput_is_independent_of_text_content():
    model=m.SemanticCurriculumActor(value_count=2,width=16,microsteps=1,workspace_rows=2,capacity=8,no_input=True,edge_width=8)
    a=model(m.ActorInput('foo bar',())); b=model(m.ActorInput('baz qux',()))
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_smoke_exposure_manifest_and_heldout(tmp_path):
    config=dict(train_count=3,eval_count=2,presentations=2,batch_size=2,seeds=[0],width=16,
                microsteps=1,workspace_rows=2,arms=['semantic','no_input'],frequency_control=False,
                max_attempts=100,output_dir=str(tmp_path/'run'),node_presentations=0,identity_presentations=0)
    rows=m.run(config)
    assert all(row['optimizer_presentations'] in (0,2) for row in rows)
    assert all(row['actual_unique_graphs_seen']==1 for row in rows if row['optimizer_presentations']==2)
    assert all(set(row['evaluation'])=={'english','spanish','symbols','heldout_lexicon'} for row in rows)


def test_frequency_control_vectorized_counts(tmp_path):
    corpus=m.base.CorpusIndex(tmp_path/'corpus.db',requested=2,seed=13,max_attempts=20)
    vocab=m.base.value_vocabulary(corpus[i] for i in range(2))
    fitted=m.frequency_fit(corpus,range(2),vocab,128)
    assert fitted['edges'].shape==(128,128,len(m.ROLES))
    assert fitted['presence'].dtype==torch.bool
    assert fitted['copy'].min()>=-1


def test_true_minibatch_matches_singletons_with_public_padding():
    torch.manual_seed(17)
    model=m.SemanticCurriculumActor(value_count=3,width=16,microsteps=1,workspace_rows=2,capacity=8,edge_width=8)
    publics=[m.ActorInput('a',()),m.ActorInput('b plus c plus d',())]
    singles=[model(p) for p in publics]
    batch=model.forward_batch(publics)
    for a,b in zip(singles,batch):
        assert all(torch.allclose(a[k],b[k],atol=2e-6) for k in a)
    changed=model.forward_batch([publics[0],m.ActorInput('other text',())])[0]
    assert all(torch.allclose(batch[0][k],changed[k],atol=2e-6) for k in batch[0])


def test_bfloat16_autocast_stays_finite_and_close():
    torch.manual_seed(23)
    model=m.SemanticCurriculumActor(value_count=3,width=16,microsteps=1,workspace_rows=2,capacity=8,edge_width=8)
    public=m.ActorInput('a plus b',())
    exact=model(public)
    model.autocast_dtype='bfloat16'
    mixed=model(public)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in mixed.values())
    assert all(torch.allclose(exact[k],mixed[k],atol=.025,rtol=.1) for k in exact)
    sum(v.square().mean() for v in mixed.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_edge_calibration_uses_only_supplied_training_examples():
    examples=[m.base.build_tcn_example('variable_binding',13)]
    vocab=m.base.value_vocabulary(examples); visited=[]
    def scorer(public):
        visited.append(public.text)
        return dict(presence=torch.ones(128),edges=torch.zeros(128,128,len(m.ROLES)))
    thresholds,audit=m.calibrate_edge_thresholds(scorer,examples,vocab,128)
    assert len(visited)==2 and audit['fit_graphs']==1
    assert thresholds.shape==(len(m.ROLES),)
    assert all(v in audit['grid'] for v in thresholds.tolist())


def test_compact_graph_edges_roundtrip_losslessly():
    e=m.base.build_tcn_example('variable_binding',13)
    public,g=m.base.surface_input(e,'english')
    gold=m.base.targets(g,public,128,m.base.value_vocabulary([e]))
    recovered=m.unpack_graph(m.pack_graph(gold))
    assert all(torch.equal(gold[k],recovered[k]) for k in gold)
    assert m.base.metrics(recovered,gold)['semantic_equivalence']==1
