import copy
import unittest
import numpy as np
import torch
from topoformer.campaign_returns_linear_continuation import update,require_exact_replay,cpu_tree


class ContinuationTests(unittest.TestCase):
    def test_replayed_and_split_updates_preserve_optimizer_and_rng(self):
        # Mathematical CPU fixture, not a primary experimental width.
        torch.manual_seed(3);x=torch.randn(64,4);y=torch.arange(64)%3
        head=torch.nn.Linear(4,3);full=copy.deepcopy(head)
        opt=torch.optim.AdamW(head.parameters(),lr=.003);fullopt=torch.optim.AdamW(full.parameters(),lr=.003)
        rng=torch.Generator().manual_seed(9);fullrng=torch.Generator().manual_seed(9)
        a,_=update(head,opt,rng,x,y,4,8)
        b,_=update(head,opt,rng,x,y,4,8)
        c,_=update(full,fullopt,fullrng,x,y,8,8)
        self.assertEqual(a+b,c)
        self.assertTrue(all(torch.equal(v,full.state_dict()[k])for k,v in head.state_dict().items()))
        self.assertTrue(torch.equal(rng.get_state(),fullrng.get_state()))

    def test_mismatch_stops_replay(self):
        h=torch.nn.Linear(4,3);m=torch.zeros(4);s=torch.ones(4);v=torch.ones(4,dtype=torch.bool)
        ref={'state':cpu_tree(h.state_dict()),'mean':m,'scale':s}
        old={'visited_row_bits_little_endian':np.packbits(v.numpy(),bitorder='little').tobytes().hex()}
        require_exact_replay(h,m,s,v,ref,old)
        ref['state']['bias'][0]+=1
        with self.assertRaises(ValueError):require_exact_replay(h,m,s,v,ref,old)

if __name__=='__main__':unittest.main()
