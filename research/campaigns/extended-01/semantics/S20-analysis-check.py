"""Synthetic S20 checks, no artifact or confirmation reads."""
import copy,importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('S20-analysis.py');spec=importlib.util.spec_from_file_location('s20',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Checks(unittest.TestCase):
 def test_integer_boundaries(self):
  c={'3x3':410,'3x4':52,'4x3':461,'4x4':512}
  self.assertTrue(all(a.count_gate(c).values()))
  self.assertFalse(a.count_gate({**c,'4x3':460})['known_macro_competence'])
  self.assertFalse(a.count_gate({**c,'3x3':409,'4x3':462})['each_known_cell_competence'])
  self.assertFalse(a.count_gate({**c,'3x4':51})['heldout_competence'])
 def test_all_seed_both_comparator(self):
  c={s:dict.fromkeys(a.CELLS,512) for s in a.SEEDS};d={s:dict.fromkeys(a.COMPARATORS,1) for s in a.SEEDS};ci={s:{b:[.1,2.] for b in a.COMPARATORS} for s in a.SEEDS}
  self.assertTrue(a.decisions(c,d,ci)['replicated_known_claim'])
  for seed in a.SEEDS:
   for arm in a.COMPARATORS:
    z=copy.deepcopy(ci);z[seed][arm]=[0.,2.];self.assertFalse(a.decisions(c,d,z)['replicated_known_claim'])
  c[701]['3x4']=0;self.assertTrue(a.decisions(c,d,ci)['replicated_known_claim']);self.assertFalse(a.decisions(c,d,ci)['heldout_claim'])
 def test_transitions(self):self.assertEqual(a.transitions([0,0,1,1],[0,1,0,1]),dict(zip(('0->0','0->1','1->0','1->1'),(1,1,1,1))))
 def test_bootstrap_constant(self):
  draws=np.zeros((3,10000,512),dtype=np.int16)
  for v in (-1,0,1):
   ci,dist=a.paired_interval(np.full((3,512),v),draws);self.assertEqual(ci,[100*v]*2);self.assertTrue(np.all(dist==v))
 def test_support_rejection(self):
  with self.assertRaises(ValueError):a.validate({})
  with self.assertRaises(ValueError):a.count_gate(dict.fromkeys(a.CELLS,513))
  with self.assertRaises(ValueError):a.count_gate(dict.fromkeys(a.CELLS,410.))
 def test_identity(self):
  cells={cell:{format(i+512*j,'064x'):False for i in range(512)} for j,cell in enumerate(a.CELLS)}
  f={s:{'record':{'categorical':copy.deepcopy(cells)},**{arm:{p:copy.deepcopy(cells) for p in a.POLICIES} for arm in a.COMPARATORS}} for s in a.SEEDS}
  a.validate(f)
  f[701]['original']['raw']['3x3'][format(0,'064x')]=0
  with self.assertRaises(ValueError):a.validate(f)
if __name__=='__main__':unittest.main()
