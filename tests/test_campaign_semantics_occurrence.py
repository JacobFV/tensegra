"""Small contract fixtures, not reduced-width experiments."""
import pytest
import torch
from topoformer.campaign_semantics_occurrence import occurrence_targets


def test_occurrences_preserve_canonical_identity_and_other_fields():
    row={'text':'parent: bob, bob; parent: erin, bob.', 'nodes':[['ident','bob'],['entity','bob'],['ident','bob'],['ident','erin'],['entity','erin'],['ident','bob']]}
    gold={'copy':torch.tensor([2,2,2,8,8,2]),'edges':torch.zeros(6,6,dtype=torch.bool)}
    changed=occurrence_targets(row,gold)
    assert changed['copy'].tolist()==[2,2,4,8,8,10]
    assert gold['copy'].tolist()==[2,2,2,8,8,2]
    assert changed['edges'] is gold['edges']
    tokens=['parent',':','bob',',','bob',';','parent',':','erin',',','bob','.']
    first={t:tokens.index(t) for t in tokens}
    assert [first[tokens[i]] for i in changed['copy'].tolist()]==gold['copy'].tolist()


def test_mismatched_occurrence_order_rejected():
    row={'text':'erin bob', 'nodes':[['ident','bob'],['ident','erin']]}
    with pytest.raises(ValueError):occurrence_targets(row,{'copy':torch.tensor([1,0])})
