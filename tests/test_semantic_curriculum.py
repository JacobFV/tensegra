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
