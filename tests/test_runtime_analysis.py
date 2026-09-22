"""Artifact analysis can be tested without importing Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('runtime_analysis', Path(__file__).parents[1]/'src/topoformer/runtime_analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

CONFIG = dict(seeds=[0, 1, 2], variants=['oracle', 'supervised'], eval_depths=[4], eval_sizes=[8], extra_evaluations=False, warmup_steps=2)


def rows():
    for variant in CONFIG['variants']:
        for seed in CONFIG['seeds']:
            counts = dict(examples=2, steps=10, binding_correct=8, primitive_correct=10,
                          all_binding_correct=1, all_primitive_correct=2, all_lowering_correct=1,
                          valid=2, execution_correct=1, task_correct=1, complete_trajectory=1,
                          task_given_binding_correct=1, execution_given_lowering_correct=1,
                          task_given_execution_correct=1, oracle_lift_correct=2)
            cell = dict(condition='n8_d4', settings=dict(nodes=8,depth=4), data_hash=f'data{seed}', counts=counts,
                        confidence=[dict(threshold=.8,examples=2,answered=0,correct=0,invoked=0,deferred=2,rejected=0,
                                         accepted_wrong=0,accepted=0,deferred_correct=10,correct_lowerings=10)])
            yield dict(variant=variant, seed=seed, source={'commit':'abc'}, config_hash=a.hashlib.sha256(json.dumps(CONFIG,sort_keys=True,separators=(',', ':')).encode()).hexdigest(),
                       training=dict(schedule_hash=f'schedule{seed}', initial_state_hash=f'init{seed}',curve=[]),
                       evaluations=[cell],initial_evaluations=[], resources={})


class AnalysisTests(unittest.TestCase):
    def test_counts_conditionals_and_no_abstention_success(self):
        out=a.summarize(rows(), CONFIG)
        cell=out['aggregates'][0]
        self.assertAlmostEqual(cell['metrics']['binding_accuracy']['mean'], .8)
        self.assertEqual(cell['pooled_rates']['execution_given_lowering_correct'], 1.)
        self.assertEqual(cell['counts']['examples'], 6)
        risk=out['confidence'][0]
        self.assertIsNone(risk['risk'])
        self.assertEqual(risk['unconditional_execution_accuracy'],0.)
        self.assertEqual(risk['coverage'],0.)
        self.assertEqual(len(out['paired_contrasts']),1)

    def test_pairing_and_complete_grid(self):
        broken=list(rows()); broken[-1]['evaluations'][0]['data_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'unpaired'):
            a.summarize(broken,CONFIG)
        with self.assertRaisesRegex(ValueError,'missing|coverage'):
            a.summarize(list(rows())[:-1],CONFIG)

    def test_undefined_and_invalid_counts(self):
        self.assertIsNone(a.ratio(0,0))
        with self.assertRaises(ValueError): a.ratio(1,0)
        with self.assertRaises(ValueError): a.ratio(-1,2)
        with self.assertRaises(ValueError): a.ratio(3,2)

    def test_gzip_stream_and_provenance(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.jsonl.gz'
            with gzip.open(path,'wt') as f:
                for row in rows(): f.write(json.dumps(row)+'\n')
            self.assertEqual(len(list(a.read_rows(path))),6)
            self.assertEqual(len(a.file_hash(path)),64)

    def test_neural_execution_undefined_and_runtime_sizes_preserved(self):
        data=list(rows())
        for row in data:
            row['metric_semantics']={'execution_counts':'auxiliary audit'}
            cell=row['evaluations'][0]
            cell['execution_accuracy']=None
            cell['complete_trajectory_accuracy']=None
            cell['oracle_lifting_accuracy']=None
            cell['runtime_nodes']={'min':30,'max':40,'mean':35.}
        result=a.summarize(data,CONFIG)
        cell=result['aggregates'][0]
        self.assertIsNone(cell['pooled_rates']['execution_accuracy'])
        self.assertEqual(cell['pooled_rates']['result_accuracy'],.5)
        self.assertEqual(cell['runtime_nodes']['min'],30)
        self.assertEqual(cell['runtime_nodes']['max'],40)
        self.assertEqual(result['provenance'][0]['metric_semantics']['execution_counts'],'auxiliary audit')

    def test_style_breakdown_and_undefined_outcomes(self):
        data=list(rows())
        for row in data:
            cell=row['evaluations'][0]
            cell['style_breakdown']={
                'comparison':dict(examples=2,defined_examples=2,task_correct=1,task_accuracy=.5,oracle_lift_correct=2,oracle_lifting_accuracy=1.),
                'sign':dict(examples=0,defined_examples=0,task_correct=0,task_accuracy=None,oracle_lift_correct=None,oracle_lifting_accuracy=None)}
        out=a.summarize(data,CONFIG)
        style=next(s for s in out['styles'] if s['style']=='comparison')
        self.assertEqual(style['counts']['defined_examples'],6)
        self.assertEqual(style['pooled_rates']['oracle_lifting_accuracy'],1.)
        empty=next(s for s in out['styles'] if s['style']=='sign')
        self.assertIsNone(empty['pooled_rates']['task_accuracy'])
        self.assertIsNone(empty['pooled_rates']['oracle_lifting_accuracy'])
        cell={'counts':dict(examples=2,defined_examples=0,all_lowering_correct=2,execution_given_lowering_correct=0,oracle_lift_correct=2)}
        self.assertIsNone(a.rates(cell)['execution_given_lowering_correct'])
        self.assertIsNone(a.rates(cell)['oracle_lifting_accuracy'])

    def test_mixed_source_rejected(self):
        broken=list(rows()); broken[-1]['source']['commit']='different'
        with self.assertRaisesRegex(ValueError,'source'):
            a.summarize(broken,CONFIG)


if __name__=='__main__': unittest.main()
