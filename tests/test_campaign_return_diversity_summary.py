import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'research/tools/campaign_return_diversity_summary.py'
spec=importlib.util.spec_from_file_location('r10_summary',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row(head,split,correct):
    return dict(head=head,split=split,target_delay=16,distractors=8,event_sha256='shared',
                targets={'value':[1]*128,'type':[1]*128},
                predictions={'value':[1]*correct+[0]*(128-correct),'type':[1]*128})


class SummaryTests(unittest.TestCase):
    def test_exact_advancement_boundary_and_paired_denominator(self):
        rows=[row(h,s,n) for h,grid in [('balanced_16384',118),('balanced_65536',122)]
              for s,n in [('calibration',128),('calibration_grid',grid)]]
        result=module.summarize(rows,require_main=False)
        self.assertTrue(result['development_gate']['advance'])
        pair=next(r for r in result['paired'] if r['split']=='calibration_grid' and r['field']=='value')
        self.assertEqual(pair['outcomes'],{'0->0':6,'0->1':4,'1->0':0,'1->1':118})

    def test_worse_mixture_cannot_be_hidden_by_grid(self):
        rows=[row(h,s,n) for h,grid in [('balanced_16384',118),('balanced_65536',128)]
              for s,n in [('calibration',125),('calibration_grid',grid)]]
        self.assertFalse(module.summarize(rows,require_main=False)['development_gate']['advance'])

    def test_missing_or_profile_cells_reject_main_gate(self):
        with self.assertRaises(ValueError):module.summarize([])
        with self.assertRaises(ValueError):module.summarize([row("balanced_65536","calibration_grid",128)])

    def test_unpaired_population_rejected(self):
        a=row('balanced_16384','calibration_grid',128);b=row('balanced_65536','calibration_grid',128)
        b['event_sha256']='different'
        with self.assertRaises(ValueError):module.summarize([a,b],require_main=False)

if __name__=='__main__':unittest.main()
