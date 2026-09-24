import unittest
import torch
from topoformer.campaign_returns_balanced_diversity import nested_indices,prefix_cache


class BalancedDiversityTests(unittest.TestCase):
    def test_exact_nested_balancing_without_value_edit(self):
        strata=[(t,y) for t,ys in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))) for y in ys]
        pairs=torch.tensor(strata*9)
        batch={'targets':{'type':pairs[:,0],'value':pairs[:,1]}}
        large=nested_indices(batch,413,31);small=nested_indices(batch,109,31)
        self.assertTrue(torch.equal(small,large[:109]))
        self.assertEqual(len(large.unique()),413)
        for ids in (small,large):
            counts=[int(((pairs[ids,0]==t)&(pairs[ids,1]==y)).sum()) for t,y in strata]
            self.assertLessEqual(max(counts)-min(counts),1)

    def test_insufficient_support_rejected(self):
        batch={'targets':{'type':torch.zeros(52,dtype=torch.long),'value':torch.zeros(52,dtype=torch.long)}}
        with self.assertRaises(ValueError):nested_indices(batch,52,1)

    def test_prefix_features_targets_and_hashes_match(self):
        b={'features':{0:torch.arange(24).reshape(6,4)},'labels':torch.arange(6),
           'targets':{'value':torch.arange(6)},'event_row_hashes':list('abcdef')}
        p=prefix_cache(b,2)
        self.assertEqual(p['features'][0].data_ptr(),b['features'][0].data_ptr())
        self.assertEqual(p['event_row_hashes'],['a','b'])
        self.assertEqual(p['labels'].tolist(),[0,1])

if __name__=='__main__':unittest.main()
