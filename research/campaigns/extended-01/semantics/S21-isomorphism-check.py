"""Synthetic fixtures only, before inspecting strict-isomorphism outcomes."""
import importlib.util,time,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('S21-isomorphism.py');spec=importlib.util.spec_from_file_location('iso',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture():return m.graph([(7,1,-1),(5,-1,0),(5,-1,1)],[(0,1,3,0),(0,2,3,1)],['alice','bob','alice'])
class Tests(unittest.TestCase):
 def test_identity(self):self.assertEqual(m.compare(fixture(),fixture())[0],'permutation_equivalent')
 def test_permutation(self):self.assertEqual(m.compare(fixture(),m.nx.relabel_nodes(fixture(),{0:2,1:0,2:1}))[0],'permutation_equivalent')
 def test_duplicate_public_occurrence(self):g=m.graph([(7,1,-1),(5,-1,2),(5,-1,1)],[(0,1,3,0),(0,2,3,1)],['alice','bob','alice']);self.assertEqual(m.compare(fixture(),g)[0],'permutation_equivalent')
 def test_copied_identity(self):g=fixture();g.nodes[1]['attribute']=('ident','public_identity','carol');self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_type(self):g=fixture();g.nodes[0]['attribute']=('record','finite_json','"parent"');self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_value(self):g=fixture();g.nodes[0]['attribute']=('pred','finite_json','"unify"');self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_scalar_value(self):
  a=m.nx.MultiDiGraph();a.add_node(0,attribute=('num','finite_json','1'));b=a.copy();b.nodes[0]['attribute']=('num','finite_json','2');self.assertEqual(m.compare(a,b)[0],'non_isomorphic')
 def test_direction(self):g=fixture();g.remove_edge(0,1);g.add_edge(1,0,relation='argument',slot=0);self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_slots(self):g=fixture();g[0][1][0]['slot']=1;g[0][2][0]['slot']=0;self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_relation(self):g=fixture();g[0][1][0]['relation']='item';self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_multiplicity(self):g=fixture();g.add_edge(0,1,relation='argument',slot=0);self.assertEqual(m.compare(fixture(),g)[0],'non_isomorphic')
 def test_edge_match_multiset(self):self.assertFalse(m.edge_match({0:dict(relation='x',slot=0),1:dict(relation='x',slot=0)},{0:dict(relation='x',slot=0)}))
 def test_same_counts_not_isomorphic(self):
  a=m.graph([(12,3,-1)]*6,[(i,(i+1)%6,0,-1) for i in range(6)],[]);b=m.graph([(12,3,-1)]*6,[(i,(i//3)*3+(i+1)%3,0,-1) for i in range(6)],[]);self.assertEqual(m.compare(a,b),('non_isomorphic','multidigraph_matcher'))
 def test_deadline(self):self.assertEqual(m.compare(fixture(),fixture(),0)[0],'unresolved')
 def test_match_timeout(self):
  with patch.object(m.nx.algorithms.isomorphism.MultiDiGraphMatcher,'is_isomorphic',lambda _:time.sleep(.03)):self.assertEqual(m.compare(fixture(),fixture(),.001)[0],'unresolved')
if __name__=='__main__':
 start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));print(dict(tests=r.testsRun,passed=r.wasSuccessful(),cpu_wall_seconds=time.monotonic()-start));raise SystemExit(not r.wasSuccessful())
