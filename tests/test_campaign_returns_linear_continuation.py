import copy
import unittest
import numpy as np
import torch
from tensegra.campaign_returns_linear_continuation import update,require_exact_replay,cpu_tree


class ContinuationTests(unittest.TestCase):
    def test_replayed_and_split_updates_preserve_optimizer_and_rng(self):
        # Mathematical CPU fixture, not a primary experimental width.
        torch.manual_seed(3);x=torch.randn(64,4);y=torch.arange(64)%3
        head=torch.nn.Linear(4,3);full=copy.deepcopy(head)
        opt=torch.optim.AdamW(head.parameters(),lr=.003);fullopt=torch.optim.AdamW(full.parameters(),lr=.003)
        rng=torch.Generator().manual_seed(9);fullrng=torch.Generator().manual_seed(9)
        a,_,_=update(head,opt,rng,x,y,4,8)
        b,_,_=update(head,opt,rng,x,y,4,8)
        c,_,_=update(full,fullopt,fullrng,x,y,8,8)
        self.assertEqual(a+b,c)
        self.assertTrue(all(torch.equal(v,full.state_dict()[k])for k,v in head.state_dict().items()))
        self.assertTrue(torch.equal(rng.get_state(),fullrng.get_state()))
        for parameter, state in opt.state_dict()['state'].items():
            for name,value in state.items():
                self.assertTrue(torch.equal(value,fullopt.state_dict()['state'][parameter][name]))
        fork=copy.deepcopy(head);forkopt=torch.optim.AdamW(fork.parameters(),lr=.003)
        forkopt.load_state_dict(copy.deepcopy(opt.state_dict()))
        forkopt.param_groups[0]['lr']=.0003
        forkrng=torch.Generator();forkrng.set_state(rng.get_state())
        _,_,first_order=update(head,opt,rng,x,y,4,8)
        _,_,fork_order=update(fork,forkopt,forkrng,x,y,4,8)
        self.assertEqual(first_order,fork_order)
        self.assertTrue(torch.equal(rng.get_state(),forkrng.get_state()))

    def test_mismatch_stops_replay(self):
        h=torch.nn.Linear(4,3);m=torch.zeros(4);s=torch.ones(4);v=torch.ones(4,dtype=torch.bool)
        ref={'state':cpu_tree(h.state_dict()),'mean':m,'scale':s}
        old={'visited_row_bits_little_endian':np.packbits(v.numpy(),bitorder='little').tobytes().hex()}
        require_exact_replay(h,m,s,v,ref,old)
        ref['state']['bias'][0]+=1
        with self.assertRaises(ValueError):require_exact_replay(h,m,s,v,ref,old)

if __name__=='__main__':unittest.main()
