import importlib
import pytest
import torch
from topoformer.semantic_graph import compile_term


def module():
    assert importlib.util.find_spec('topoformer.semantic_scaling') is not None, 'Stage7 semantic module is missing'
    return importlib.import_module('topoformer.semantic_scaling')


def test_alpha_equivalence_preserves_order_and_identity():
    m=module()
    def graph(a,b):
        return compile_term({'t':'rel','head':'subtract','args':[{'t':'ident','v':a},{'t':'ident','v':b},{'t':'ident','v':a}]})
    assert m.semantic_key(graph('a','b'))==m.semantic_key(graph('x','y'))
    assert m.semantic_key(graph('a','b'))!=m.semantic_key(graph('a','a'))
    g=compile_term({'t':'rel','head':'subtract','args':[{'t':'num','v':1},{'t':'num','v':2}]})
    h=compile_term({'t':'rel','head':'subtract','args':[{'t':'num','v':2},{'t':'num','v':1}]})
    assert m.semantic_key(g)!=m.semantic_key(h)


def test_disk_corpus_reports_real_cardinality_and_rejects_stale_index(tmp_path):
    m=module(); corpus=m.CorpusIndex(tmp_path/'index.db',requested=12,seed=7,max_attempts=40)
    assert 0<len(corpus)<=12
    assert corpus.audit['actual_unique_graphs']==len(corpus)
    assert corpus.audit['attempts']==len(corpus)+corpus.audit['duplicates']
    assert len({m.semantic_key(corpus[i].privileged.graph) for i in range(len(corpus))})==len(corpus)
    with pytest.raises(ValueError,match='exists'):
        m.CorpusIndex(tmp_path/'index.db',requested=12,seed=7,max_attempts=40)


def test_targets_copy_visible_identifiers_and_ordered_edges():
    m=module(); e=m.build_tcn_example('variable_binding',13)
    public=m.surface_input(e,'english'); vocab=m.value_vocabulary([e])
    target=m.targets(e.privileged.graph,public,128,vocab)
    assert target['copy'].ge(0).any()
    assert target['slots'].ge(0).any()
    model=m.SemanticActor(width=16,capacity=128,workspace_rows=4,microsteps=1,value_count=len(vocab))
    out=model(public); loss=sum(m.losses(out,target).values()); loss.backward()
    assert torch.isfinite(loss)
    perfect={k:v.clone() for k,v in target.items()}
    metrics=m.metrics(perfect,target)
    assert metrics['semantic_equivalence']==1
    changed={k:v.clone() for k,v in target.items()}
    edge=torch.nonzero(changed['slots'].ge(0))[0]
    changed['slots'][tuple(edge)]+=1
    assert m.metrics(changed,target)['semantic_equivalence']==0
    assert m.metrics(changed,target)['ordered_edge']['true_positive'] < metrics['ordered_edge']['true_positive']


def test_runner_step_zero_exposure_and_no_input_baseline(tmp_path):
    m=module(); config=dict(train_count=3,eval_count=2,max_attempts=60,data_seed=17,seeds=[0],
        updates=1,eval_steps=[0,1],batch_size=1,width=16,workspace_rows=4,microsteps=1,
        node_capacity=128,learning_rate=.001,output_dir=str(tmp_path/'run'),threads=1)
    result=m.run(config)
    assert {r['arm'] for r in result}=={'semantic','no_input','frequency'}
    assert all(r['optimizer_examples']==0 for r in result if r['step']==0)
    assert all('symbols' in r['evaluation'] and 'heldout_lexicon' in r['evaluation'] for r in result)
    frequency=next(r for r in result if r['arm']=='frequency')
    assert frequency['fit_label_examples']==3
    assert frequency['actual_unique_graphs_seen']==3
    assert (tmp_path/'run'/'manifest.json').exists()


def test_unknown_values_do_not_count_as_semantic_success():
    m=module(); e=m.build_tcn_example('variable_binding',3)
    public,g=m.surface_input(e,'english'); gold=m.targets(g,public,128,['<unknown>'])
    assert m.metrics({k:v.clone() for k,v in gold.items()},gold)['semantic_equivalence']==0


def test_copy_decode_collapses_repeated_visible_tokens():
    m=module(); public=m.ActorInput('a b a',())
    out={'copy':torch.tensor([[0.,0.,4.]]),'presence':torch.ones(1),'kind':torch.zeros(1,2),'value':torch.zeros(1,2),'edges':torch.zeros(1,1,1),'slots':torch.zeros(1,1,2)}
    assert m.decode(out,public)['copy'].item()==0


def test_all_renderer_targets_copy_rendered_identifiers():
    m=module()
    for seed in range(20):
        e=m.build_tcn_example('variable_binding',700000+seed)
        vocab=m.value_vocabulary([e])
        for language in ('english','spanish','symbols'):
            public,g=m.surface_input(e,language)
            assert m.targets(g,public,128,vocab,language=language)['copy'].ge(0).any()


def test_spurious_ordered_slot_prevents_exact_semantics():
    m=module(); e=m.build_tcn_example('variable_binding',3); p,g=m.surface_input(e,'english')
    gold=m.targets(g,p,128,m.value_vocabulary([e])); pred={k:v.clone() for k,v in gold.items()}
    unordered=torch.nonzero(gold['edges'].any(-1)&gold['slots'].lt(0))[0]
    pred['slots'][tuple(unordered)]=0
    assert m.metrics(pred,gold)['semantic_equivalence']==0


def test_scaling_splits_reserve_same_holdout_before_nested_training():
    m=module()
    small=m.split_indices(3,2); large=m.split_indices(10,2)
    assert small['eval']==large['eval']==range(2)
    assert list(small['train'])==list(large['train'])[:3]
    assert not set(large['train'])&set(small['eval'])


def test_slot_gradient_not_diluted_by_padding():
    m=module(); e=m.build_tcn_example('variable_binding',3); p,g=m.surface_input(e,'english'); vocab=m.value_vocabulary([e])
    masses=[]
    for capacity in (64,128):
        gold=m.targets(g,p,capacity,vocab)
        out={'presence':torch.zeros(capacity,requires_grad=True),'kind':torch.zeros(capacity,len(m.KINDS),requires_grad=True),'value':torch.zeros(capacity,len(vocab),requires_grad=True),'copy':torch.zeros(capacity,len(m.tokens(p)),requires_grad=True),'edges':torch.zeros(capacity,capacity,len(m.ROLES),requires_grad=True),'slots':torch.zeros(capacity,capacity,m.MAX_SLOT+1,requires_grad=True)}
        m.losses(out,gold)['slots'].backward(); mask=gold['slots'].ge(0)
        masses.append(float(out['slots'].grad[mask].abs().sum()))
    assert masses[0]==pytest.approx(masses[1],rel=.01)
    assert masses[0]>.5


def test_pinned_spanish_gender_agreement_is_visible_copy_target():
    m=module(); e=m.build_tcn_example('set_operations',700191,difficulty=.5)
    public,g=m.surface_input(e,'spanish'); tok=m.tokens(public)
    assert 'amarilla' in tok and 'amarillo' not in tok
    gold=m.targets(g,public,128,m.value_vocabulary([e]),language='spanish')
    nodes=[i for i,n in enumerate(g.nodes) if n.kind in ('ident','entity') and n.value=='yellow']
    assert nodes and all(tok[int(gold['copy'][i])]=='amarilla' for i in nodes)
