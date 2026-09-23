import inspect,unittest
import torch
from topoformer.campaign_semantics_contract_decode import decode,allowed
from topoformer.thinking_language import KINDS,ROLES

class ContractTests(unittest.TestCase):
    def example(self):
        return dict(presence=torch.ones(5,dtype=torch.bool),kind=torch.tensor([KINDS.index(x) for x in ('scope','pred','ident','entity','num')]),value=torch.zeros(5,dtype=torch.long),copy=torch.tensor([-1,-1,7,7,-1]),edges=torch.zeros(5,5,len(ROLES),dtype=torch.bool),slots=torch.full((5,5),-1,dtype=torch.long))
    def test_no_target_interface_and_no_mutation(self):
        self.assertEqual(set(inspect.signature(decode).parameters),{'pred','mask','bookkeeping'})
        p=self.example();original={k:v.clone() for k,v in p.items()};decode(p,mask=True,bookkeeping=True)
        self.assertTrue(all(torch.equal(p[k],v) for k,v in original.items()))
    def test_bookkeeping_preserves_associations(self):
        p=self.example();p['edges'][1,2,ROLES.index('argument')]=True;p['slots'][1,2]=2
        q,info=decode(p,bookkeeping=True)
        self.assertTrue(q['edges'][1,2,ROLES.index('argument')]);self.assertEqual(int(q['slots'][1,2]),2)
        self.assertTrue(q['edges'][2,3,ROLES.index('refers_to')]);self.assertEqual(int(q['edges'][:,:,ROLES.index('contains')].sum()),3)
    def test_scope_ambiguity_retains(self):
        p=self.example();p['kind'][4]=KINDS.index('scope');p['edges'][0,2,0]=True
        q,info=decode(p,bookkeeping=True);self.assertTrue(info['bookkeeping_abstained']);self.assertTrue(torch.equal(q['edges'][:,:,0],p['edges'][:,:,0]))
    def test_duplicate_reference_and_schema(self):
        p=self.example();p['kind'][4]=KINDS.index('entity');p['copy'][4]=7;p['edges'][3,2,ROLES.index('argument')]=True
        q,info=decode(p,mask=True,bookkeeping=True)
        self.assertEqual(info['multiple_references'],1);self.assertEqual(int(q['edges'][2,:,ROLES.index('refers_to')].sum()),2)
        self.assertFalse(q['edges'][3,2,ROLES.index('argument')]);self.assertTrue(allowed('list','num','item'))
    def test_permutation_and_identity_renaming(self):
        p=self.example();q,_=decode(p,mask=True,bookkeeping=True);perm=torch.tensor([3,1,4,0,2])
        pp={k:(v[perm][:,perm] if k in ('edges','slots') else v[perm]) for k,v in p.items()};pp['copy'][pp['copy']>=0]+=99
        qq,_=decode(pp,mask=True,bookkeeping=True)
        self.assertTrue(torch.equal(qq['edges'],q['edges'][perm][:,perm]));self.assertTrue(torch.equal(qq['slots'],q['slots'][perm][:,perm]))
