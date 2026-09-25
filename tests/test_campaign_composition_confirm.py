import torch
from tensegra.campaign_composition_study import get_data
from tensegra.campaign_composition_roles import reversed_supervision
from tensegra.campaign_composition_rekey import rekey
from tensegra.campaign_composition_confirm_data import signatures,numeric_overlap,causal_gate


def test_numeric_audit_ignores_nonce_codes_but_preserves_order_and_actual_visits():
    data,pub,labels=get_data(dict(seed=783,count=32))
    rev=reversed_supervision(data['public'])
    new=rekey(pub,torch.Generator().manual_seed(4))
    assert signatures(pub,labels,True)==signatures(new,labels,True)
    visits=torch.zeros(32,dtype=torch.long); visits[0]=3
    swaps=torch.zeros_like(visits);swaps[0]=1
    audit=numeric_overlap({'train':(pub,labels,data['public']),'validation':(pub,labels,data['public'])},{'train':rev,'validation':rev},visits,swaps)
    for info in audit['audits'].values():
        assert info['actual_training_presentations']==3
        assert info['actual_training_distinct_signatures']<=2
        assert info['populations']['validation/clean']['events']==32
        assert info['populations']['validation/clean']['training_overlap_events']>=1


def test_causal_denominators_include_refusals_and_undefined_gain_fails():
    truth=torch.tensor([0,1,0,1])
    clean=torch.tensor([0,1,-1,-1]);drop=torch.tensor([0,-1,-1,-1])
    query=torch.tensor([1,0,-1,-1]);copy=torch.tensor([0,1,0,-1])
    z=causal_gate(clean,drop,query,copy,truth,{})
    assert z['attempted']==4 and z['clean_accuracy']==.5 and z['copy_accuracy']==.75
    assert z['clean_minus_drop']==.25 and z['normalized_gain']==2/3
    z=causal_gate(clean,drop,clean,clean,truth,{})
    assert z['normalized_gain'] is None and not z['passed']


def test_causal_gate_requires_both_changed_controls_and_sufficient_support():
    truth=torch.zeros(512,dtype=torch.long)
    supplied=torch.ones_like(truth)
    control=(supplied,supplied)
    good=dict(wrong=control,swap=control)
    assert causal_gate(truth,supplied,supplied,truth,truth,good)['passed']
    assert not causal_gate(truth,supplied,supplied,truth,truth,{})['passed']
    assert not causal_gate(truth,supplied,supplied,truth,truth,dict(wrong=control))['passed']
    assert not causal_gate(truth[:255],supplied[:255],supplied[:255],truth[:255],truth[:255],{k:(a[:255],b[:255]) for k,(a,b) in good.items()})['passed']


def test_requested_labels_replay_clean_and_reversed_runtime_without_gold_reset():
    from tensegra.campaign_composition_confirm_data import requested_labels
    from tensegra.campaign_composition_acquire import controlled_rows
    data,public,labels=get_data(dict(seed=784,count=32))
    actual,bundle=requested_labels(data['public'])
    assert all(torch.equal(actual[k],labels[k]) for k in actual)
    reversed_rows=controlled_rows(data['public'],'reverse_roles',0)
    actual,bundle=requested_labels(reversed_rows)
    _,reference=reversed_supervision(data['public'])
    assert all(torch.equal(actual[k],reference[k]) for k in actual)
    assert len(bundle['indices'])==32
