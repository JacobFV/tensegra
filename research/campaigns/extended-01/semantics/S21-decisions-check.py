import importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S21-decisions.py');s=importlib.util.spec_from_file_location('d',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
class Checks(unittest.TestCase):
 def test_boundaries(self):
  a=dict.fromkeys(d.CELLS,256);b=dict(a);a['3x4']=26;b['3x4']=52;b['3x3']-=76
  self.assertTrue(d.decisions(a,b)['advance'])
  self.assertFalse(d.decisions(a,{**b,'3x3':b['3x3']-1})['advance'])
  self.assertFalse(d.decisions(a,{**b,'3x4':51})['advance'])
  self.assertFalse(d.decisions({**a,'3x4':27},b)['advance'])
  for cell in ('2x4','5x3','5x4'):self.assertFalse(d.decisions(a,{**b,cell:255})['advance'])
 def test_support(self):
  with self.assertRaises(ValueError):d.decisions({},dict.fromkeys(d.CELLS,0))
if __name__=='__main__':unittest.main()
