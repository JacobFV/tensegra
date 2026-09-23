"""Artifact-only tests: load directly so aggregation never imports Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).parents[1] / 'src/topoformer/thinking_analysis.py'
spec = importlib.util.spec_from_file_location('analysis', PATH)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def row(seed=1, variant='local', correct=True):
    return dict(kind='evaluation', seed=seed, variant=variant, step=2, depth=8,
        condition='depth', source_hash='source', initial_hash=f'init{seed}', n=1,
        examples=[dict(task_correct=correct, exact_semantic_correct=False,
            trajectory_exact=False, numeric_correct=correct, decision_correct=True,
            microsteps=3, halted=True, max_forced=False, post_event_pass=True,
            input_hash=f'input{seed}', data_hash=f'data{seed}', trace=[])],
        readiness_calibration=[dict(probability=.75,target=0)], counts={})


class AnalysisTests(unittest.TestCase):
    def test_seed_stats_pairing_and_undefined_conditional(self):
        result=a.summarize_controlled(iter([row(),row(2),row(1,'fixed',False),row(2,'fixed',True)]))
        local=next(x for x in result['aggregates'] if x['variant']=='local')
        self.assertEqual(local['metrics']['task_accuracy'],dict(n=2,mean=1.,sd=0.))
        self.assertIsNone(local['metrics']['task_given_exact']['mean'])
        self.assertEqual(local['counts']['examples'],2)
        self.assertEqual(result['paired_contrasts'][0]['differences']['task_accuracy']['mean'],-.5)
        self.assertNotIn('examples',local)
        self.assertAlmostEqual(local['calibration']['brier'],.5625)
        self.assertEqual(local['calibration']['risk_coverage'][5]['accepted'],2)

    def test_duplicate_mismatched_data_and_initialization_rejected(self):
        for mutation in ('duplicate','data','init'):
            left=row();right=row(1,'fixed')
            if mutation=='duplicate':right=left
            elif mutation=='data':right['examples'][0]['data_hash']='different'
            else:right['initial_hash']='different'
            with self.assertRaises(ValueError):a.summarize_controlled([left,right])

    def test_language_metrics_separate_oracles_and_denominators(self):
        records=[dict(seed=1,arm='single_pass',step=0,evaluation={'symbols':dict(
            task_choice_accuracy=.5,node_precision=.2,task_counts={'a':2},
            per_task_accuracy={'a':.5},oracle_majority_position_baseline=.5)})]
        result=a.summarize_language(records)
        cell=result['aggregates'][0]
        self.assertEqual(cell['metrics']['node_precision']['mean'],.2)
        self.assertNotIn('oracle_majority_position_baseline',cell['metrics'])
        self.assertEqual(cell['privileged_baselines']['oracle_majority_position_baseline']['mean'],.5)
        self.assertEqual(cell['counts']['examples'],2)

    def test_runtime_populations_distribution_and_duplicate_training(self):
        record=row()
        record['examples'][0]['trace']=[dict(proposals=[{},{}],events=[
            dict(status='refused',reason='argument unavailable or not numeric'),
            dict(status='executed',reason='')],projection_overlap=.25,
            event_roundtrips=[dict(stage='encoder',value_error=2.,type_correct=True,operation_correct=False)])]
        cell=a.summarize_controlled([record])['aggregates'][0]
        self.assertEqual(cell['metrics']['runtime_refused_rate']['mean'],.5)
        self.assertEqual(cell['counts']['runtime_executed'],1)
        self.assertEqual(cell['microstep_histogram'],{'3':1})
        self.assertEqual(cell['metrics']['roundtrip_encoder_value_error']['mean'],2.)
        train=dict(kind='training',seed=1,variant='local',step=1,losses={'task':2},data_hash='x')
        with self.assertRaises(ValueError):a.summarize_controlled([train,train])

    def test_expected_grid_reports_incomplete_artifacts(self):
        summary=a.summarize_controlled([row()])
        audit=a.coverage_audit(summary,{'seeds':[1,2],'variants':['local'],
            'steps':2,'checkpoints':[0,2],'eval_depths':[8]})
        self.assertFalse(audit['complete'])
        self.assertEqual(len(audit['missing']),3)

    def test_file_hash_audit_and_stream(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl';path.write_text('{"n":1}\n\n{"n":2}\n')
            self.assertEqual(list(a.read_rows(path)),[{'n':1},{'n':2}])
            self.assertTrue(a.verify_files({'x.jsonl':a.file_hash(path)},d)['ok'])
            self.assertFalse(a.verify_files({'x.jsonl':'bad'},d)['ok'])


if __name__=='__main__':unittest.main()
