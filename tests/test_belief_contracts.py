import torch
from topoformer.belief_state import BeliefModel,collate,oracle
from topoformer.belief_contracts import PriorContract,contract_episodes,empty_ledger_frames

def test_prior_public_only_and_full_retraction():
    batch=collate(contract_episodes(3,71,condition='full_retract'))
    for mode in ('protected','recurrent'):
        base=BeliefModel(mode,width=16,inner=32,observation_id_features=True)
        raw=base(batch['public'])['logits']
        exact=PriorContract(base,True)(batch['public'])['logits']
        mask=empty_ledger_frames(batch['public'])
        assert mask[:,0].all() and mask[:,-1].all() and not mask[:,1:-1].any()
        assert torch.equal(raw[~mask],exact[~mask])
        assert torch.equal(exact.softmax(-1)[mask],batch['targets']['posterior'][mask])

def test_all_contract_targets_match_public_oracle():
    for c in ('full_retract','distinct_equal','candidate_permutation','id_rename'):
        b=collate(contract_episodes(4,97,condition=c))
        assert torch.equal(oracle(b['public']),b['targets']['posterior'])

def test_distinct_same_content_is_not_idempotent_ledger():
    b=collate(contract_episodes(2,9,condition='distinct_equal'))
    assert len(set(b['public']['ids'][0].tolist()))==9
    # Deterministic noiseless constraints imply equal support, not doubled evidence confidence.
    q=b['targets']['posterior'];assert torch.equal(q[:,1],q[:,2])

def test_public_ledger_noops_padding_first_write_and_absent_retraction():
    p=dict(ids=torch.tensor([[-1,7,7,9,7,3]]),actions=torch.tensor([[0,1,1,-1,-1,1]]),frames=torch.tensor([[1,1,1,1,1,0]],dtype=torch.bool))
    assert empty_ledger_frames(p).tolist()==[[True,False,False,False,True,True]]
