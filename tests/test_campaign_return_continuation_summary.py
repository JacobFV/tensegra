import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/tools'))
from campaign_return_continuation_summary import endpoint_gate,select_candidate


class ContinuationSummaryTests(unittest.TestCase):
    def test_exact_boundaries(self):
        self.assertTrue(endpoint_gate(.98,122/128,65405)['advance'])
        self.assertFalse(endpoint_gate(.98,121/128,65405)['advance'])
        self.assertFalse(endpoint_gate(.98,122/128,65404)['advance'])

    def test_mixture_veto(self):
        self.assertFalse(endpoint_gate(.979,1,65536)['advance'])

    def test_constant_wins_exact_tie(self):
        g=endpoint_gate(.99,125/128,65536)
        self.assertEqual(select_candidate({'constant':g,'decay':g})['selected'],'constant')

    def test_validation_cannot_select(self):
        g={'constant':endpoint_gate(.99,123/128,65536),'decay':endpoint_gate(.99,124/128,65536)}
        baseline=select_candidate(g)
        g['constant']['validation_accuracy']=1.;g['decay']['validation_accuracy']=0.
        self.assertEqual(select_candidate(g),baseline)

    def test_lower_lr_credit_requires_four(self):
        g={'constant':endpoint_gate(.99,120/128,65536),'decay':endpoint_gate(.99,123/128,65536)}
        self.assertFalse(select_candidate(g)['lower_lr_specific_gain'])
        g['decay']=endpoint_gate(.99,124/128,65536)
        self.assertTrue(select_candidate(g)['lower_lr_specific_gain'])

if __name__=='__main__':unittest.main()
