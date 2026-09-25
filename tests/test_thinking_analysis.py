"""Artifact-only tests: load directly so aggregation never imports Torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).parents[1] / 'src/tensegra/thinking_analysis.py'
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

    def test_language_micro_counts_null_metrics_and_hash_mismatch(self):
        row1=dict(seed=1,arm='single_pass',step=0,initial_hash='init1',data_hash='data',config_hash='config',
            evaluation={'symbols':dict(node_precision=None,node_recall=0.,task_counts={'a':2},
            graph_counts={'node':dict(true_positive=0,predicted_count=0,gold_count=5)},
            graph_defined_examples={'node':dict(precision=0,recall=2)})})
        result=a.summarize_language([row1])
        cell=result['aggregates'][0]
        self.assertIsNone(cell['metrics']['node_precision']['mean'])
        self.assertIsNone(cell['pooled_graph_rates']['node_precision'])
        self.assertEqual(cell['pooled_graph_rates']['node_recall'],0.)
        self.assertEqual(cell['counts']['node_gold_count'],5)
        self.assertEqual(cell['counts']['node_precision_defined_examples'],0)
        bad=dict(row1,arm='recurrent',initial_hash='different')
        with self.assertRaises(ValueError):a.summarize_language([row1,bad])

    def test_language_manifest_checks_renamed_data_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'model.pt').write_bytes(b'checkpoint')
            audit=dict(examples=[{'id':1}],renamed_examples=[{'id':2}],config={'seeds':[1]},
                       semantic_train=['train'],semantic_eval=['eval'])
            audit['data_hash']=a.digest(audit['examples']+audit['renamed_examples'])
            audit['config_hash']=a.digest(audit['config'])
            manifest=dict(data_hash=audit['data_hash'],config_hash=audit['config_hash'],
                runs=[dict(seed=1,arm='single_pass',initial_hash='init',checkpoint='model.pt',checkpoint_sha256=a.file_hash(root/'model.pt'))])
            result=a.audit_language(audit,manifest,root)
            self.assertTrue(result['data_hash_matches'])
            self.assertTrue(result['checkpoint_hashes']['ok'])
            (root/'model.pt').write_bytes(b'changed')
            self.assertFalse(a.audit_language(audit,manifest,root)['checkpoint_hashes']['ok'])

    def test_controlled_protocol_difference_is_explicit(self):
        left=row();left['observation_protocol']='progressive_exogenous_context'
        right=row(1,'neural_recurrent');right['observation_protocol']='complete_from_start'
        result=a.summarize_controlled([left,right])
        self.assertFalse(result['paired_contrasts'][0]['matched_observation_protocol'])
        self.assertEqual(result['aggregates'][0]['observation_protocols'],['progressive_exogenous_context'])

    def make_shard(self, root, name, seed, config=None):
        shard=root/name;shard.mkdir()
        config=config or dict(seeds=[seed],variants=['local'],steps=0,checkpoints=[0],eval_depths=[8],save_checkpoints=True)
        sources={'source.py':'sourcehash'}
        manifest=dict(config=config,sources=sources,source_hash=a.digest(sources))
        (shard/'manifest.json').write_text(json.dumps(manifest))
        record=row(seed);record.update(step=0,source_hash=manifest['source_hash'])
        (shard/'metrics.jsonl').write_text(json.dumps(record)+'\n')
        (shard/f'local-seed{seed}-step0.pt').write_bytes(f'checkpoint {seed}'.encode())
        return shard

    def test_merge_disjoint_shards_preserves_raw_identity_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);left=self.make_shard(root,'left',1);right=self.make_shard(root,'right',2)
            output=root/'merged'
            manifest=a.merge_shards({'seeds':[1,2],'variants':['local'],'steps':0},[left,right],output)
            self.assertEqual(manifest['config']['seeds'],[1,2])
            self.assertEqual(len(manifest['combined_from']),2)
            self.assertEqual([r['seed'] for r in a.read_rows(output/'metrics.jsonl.gz')],[1,2])
            self.assertEqual((output/'local-seed2-step0.pt').read_bytes(),b'checkpoint 2')
            self.assertEqual((output/'shards/000/manifest.json').read_bytes(),(left/'manifest.json').read_bytes())
            self.assertEqual(manifest['combined_from'][0]['metrics_uncompressed_sha256'],a.file_hash(left/'metrics.jsonl'))
            summary=a.summarize_controlled(a.read_rows(output/'metrics.jsonl.gz'))
            self.assertTrue(a.coverage_audit(summary,manifest['config'])['complete'])

    def test_merge_rejects_overlap_missing_seed_config_mismatch_and_gap(self):
        for problem in ('overlap','missing_seed','config_mismatch','source_mismatch','gap','checkpoint_gap'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);left=self.make_shard(root,'left',1)
                right=self.make_shard(root,'right',1 if problem=='overlap' else 2)
                if problem=='config_mismatch':
                    path=right/'manifest.json';manifest=json.loads(path.read_text());manifest['config']['steps']=1;path.write_text(json.dumps(manifest))
                if problem=='source_mismatch':
                    path=right/'manifest.json';manifest=json.loads(path.read_text());manifest['sources']['source.py']='different';manifest['source_hash']=a.digest(manifest['sources']);path.write_text(json.dumps(manifest))
                if problem=='gap':(right/'metrics.jsonl').write_text('')
                if problem=='checkpoint_gap':(right/'local-seed2-step0.pt').unlink()
                shards=[left] if problem=='missing_seed' else [left,right]
                with self.assertRaises(ValueError):a.merge_shards({'seeds':[1,2],'variants':['local'],'steps':0},shards,root/'merged')
                self.assertFalse((root/'merged').exists())

    def test_economy_grid_tracks_anchors_final_ood_and_privileged_oracles(self):
        config=dict(seeds=[1],variants=['local','fixed'],steps=2,checkpoints=[0,1,2],
                    eval_depths=[2,8,32],train_depth=2,evaluation_schedule='economy_v1',extra_evaluations=True)
        expected=a.expected_controlled_evaluations(config)
        self.assertEqual(len(expected),46)
        self.assertNotIn(('local',1,0,8,'depth','none'),expected)
        self.assertIn(('local',1,2,32,'oracle_minimal','oracle_minimal'),expected)
        self.assertIn(('fixed',1,2,2,'heldout_wordorder','none'),expected)
        cells=[dict(variant=v,seeds=[seed],step=step,depth=depth,condition=condition,intervention=intervention)
               for v,seed,step,depth,condition,intervention in expected]
        summary=dict(track='controlled_execution',aggregates=[c for c in cells if not c['condition'].startswith('oracle')],
                     privileged_oracles=[c for c in cells if c['condition'].startswith('oracle')])
        self.assertTrue(a.coverage_audit(summary,config)['complete'])
        summary['privileged_oracles'].pop()
        self.assertFalse(a.coverage_audit(summary,config)['complete'])

    def test_merge_preserves_modern_shard_provenance_and_verifies_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shards=[self.make_shard(root,'left',1),self.make_shard(root,'right',2)]
            for seed,shard in enumerate(shards,1):
                path=shard/'manifest.json';manifest=json.loads(path.read_text());name=f'local-seed{seed}-step0.pt'
                manifest.update(config_hash=a.digest(manifest['config']),git_commit='samecommit',started_utc=f'2026-09-22T00:00:0{seed}Z',
                    elapsed_seconds=seed,initializations={f'local:seed{seed}':f'init{seed}'},
                    checkpoints={name:dict(state_hash=f'state{seed}',file_sha256=a.file_hash(shard/name))})
                path.write_text(json.dumps(manifest))
                r=json.loads((shard/'metrics.jsonl').read_text());r['checkpoint_hash']=f'state{seed}';(shard/'metrics.jsonl').write_text(json.dumps(r)+'\n')
            combined=a.merge_shards({'seeds':[1,2]},shards,root/'combined')
            self.assertEqual(len(combined['checkpoints']),2)
            self.assertEqual(combined['initializations']['local:seed2'],'init2')
            self.assertEqual(combined['config_hash'],a.digest(combined['config']))
            (shards[1]/'local-seed2-step0.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):a.merge_shards({'seeds':[1,2]},shards,root/'bad')

    def test_file_hash_audit_and_stream(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl';path.write_text('{"n":1}\n\n{"n":2}\n')
            self.assertEqual(list(a.read_rows(path)),[{'n':1},{'n':2}])
            self.assertTrue(a.verify_files({'x.jsonl':a.file_hash(path)},d)['ok'])
            self.assertFalse(a.verify_files({'x.jsonl':'bad'},d)['ok'])


if __name__=='__main__':unittest.main()
