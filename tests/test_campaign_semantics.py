"""Small CPU contract fixtures, not primary-width acquisition experiments."""
import torch
from topoformer import semantic_scaling as base
from topoformer.campaign_semantics_data import compact_example,target


def test_compact_cache_preserves_compiler_targets():
    examples=[base.build_tcn_example('unification',12000000+i,difficulty=.5) for i in range(3)]
    vocab=base.value_vocabulary(examples)
    for example in examples:
        public,_=base.surface_input(example,'english');row=compact_example(example,public)
        actual=target(row,vocab);expected=base.targets(example.privileged.graph,public,128,vocab,'english')
        assert all(torch.equal(actual[k],expected[k]) for k in expected)


def test_ordered_slot_labels_not_actor_text():
    example=base.build_tcn_example('unification',12000000,difficulty=.5);public,_=base.surface_input(example,'english');row=compact_example(example,public);vocab=base.value_vocabulary([example])
    before=target(row,vocab);ordered=next(e for e in row['edges'] if e[3] is not None);ordered[3]=(ordered[3]+1)%4
    after=target(row,vocab)
    assert row['text']==public.text
    assert torch.equal(before['edges'],after['edges'])
    assert not torch.equal(before['slots'],after['slots'])
