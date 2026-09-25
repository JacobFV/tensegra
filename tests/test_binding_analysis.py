import pytest
from tensegra.binding_analysis import trajectory_summary, stability_gate, pearson, feature_associations


def test_conditional_transition_denominator_and_nonduplicated_product():
    result = trajectory_summary(['111', '100', '001'])
    assert result['transition_counts'] == {'00': 2, '01': 1, '10': 1, '11': 2}
    assert result['error_persistence'] == pytest.approx(2/3)
    assert result['error_recovery'] == pytest.approx(1/3)
    assert result['complete_path'] == pytest.approx(1/3)
    assert result['independent_marginal_product'] == pytest.approx(4/27)
    assert result['first_error_survival'] == pytest.approx([2/3, 1/3, 1/3])


def test_no_errors_has_undefined_conditional_rates():
    result = trajectory_summary(['111', '111'])
    assert result['error_persistence'] is None
    assert result['error_recovery'] is None


def test_gate_requires_each_seed_not_mean():
    rows = [dict(seed=s, task_accuracy=1., exact_path_completion=1.) for s in range(3)]
    assert stability_gate(rows)['passed']
    rows[-1]['exact_path_completion'] = .94
    assert not stability_gate(rows)['passed']
    with pytest.raises(ValueError, match='all declared seeds'):
        stability_gate(rows[:2])


def test_constant_feature_or_outcome_correlation_is_null():
    assert pearson([1, 1], [0, 1]) is None
    assert pearson([0, 1], [1, 1]) is None
    result = feature_associations([dict(canonical_correct='11', margin=1), dict(canonical_correct='00', margin=1)], ['margin'])
    assert result['margin']['pearson_with_path_failure'] is None


def fixture_rows():
    rows = []
    for variant in ('stage3_soft4', 'cosine_mixed'):
        for seed in range(3):
            e = dict(nodes=16, depth=2, examples=2, task_accuracy=.5, data_hash=f'data{seed}',
                     metrics={'canonical_complete_path': .5},
                     canonical_correct=['111', '100'],
                     per_step={'pre_correct': ['11', '10'], 'post_correct': ['11', '00']},
                     per_example={'mean_margin': [.9, .1], 'distinct_gold_nodes': [2, 3]})
            rows.append(dict(variant=variant, seed=seed, source={'commit': 'abc'}, config_hash='cfg',
                             training=dict(steps=4, schedule_hash=f'schedule{seed}', initial_parameter_hashes={'q':'same'}),
                             resources={'seconds': 1}, initial_evaluations=[e], evaluations=[e]))
    return rows


CONFIG = dict(seeds=[0, 1, 2], eval_sizes=[16], eval_depths=[2], steps=4)


def test_aggregation_bitstrings_grid_and_paired_data():
    from tensegra.binding_analysis import summarize
    result = summarize(fixture_rows(), CONFIG)
    assert result['aggregates'][0]['metrics']['exact_path_completion']['mean'] == .5
    assert result['trajectories'][0]['complete_path'] == .5
    assert result['associations'][0]['associations']['mean_margin']['pearson_with_path_failure'] == pytest.approx(-1)
    assert result['paired_contrasts'][0]['task_difference']['mean'] == 0
    rows = fixture_rows()
    rows[-1]['evaluations'][0]['data_hash'] = 'mismatch'
    with pytest.raises(ValueError, match='unpaired evaluation'):
        summarize(rows, CONFIG)
    with pytest.raises(ValueError, match='variant/seed'):
        summarize(fixture_rows()[:-1], CONFIG)
    with pytest.raises(ValueError, match='depth/size'):
        summarize(fixture_rows(), dict(CONFIG, eval_sizes=[16, 32]))


def test_source_and_budget_validation():
    from tensegra.binding_analysis import summarize
    rows = fixture_rows()
    rows[-1]['source']['commit'] = 'changed'
    with pytest.raises(ValueError, match='source/config'):
        summarize(rows, CONFIG)
    with pytest.raises(ValueError, match='budget'):
        summarize(fixture_rows(), dict(CONFIG, steps=5))


def test_cli_provenance_and_report(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import tensegra.binding_analysis as analysis
    metrics = tmp_path/'metrics.jsonl'
    metrics.write_text('\n'.join(json.dumps(r) for r in fixture_rows()))
    (tmp_path/'config.json').write_text(json.dumps(CONFIG))
    output = tmp_path/'analysis'
    result = analysis.main([str(metrics), str(output), '--no-plots'])
    assert result['artifact']['analyzer_sha256'] == hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()
    assert result['artifact']['metrics_sha256'] == hashlib.sha256(metrics.read_bytes()).hexdigest()
    report = (output/'report.md').read_text()
    assert 'Gate unavailable' in report
    assert 'does not prove an attractor' in report
    assert 'same nonduplicated states' in report


def test_gzip_input_hashes_compressed_file_and_preserves_results(tmp_path):
    import gzip
    import hashlib
    import json
    from tensegra.binding_analysis import main
    metrics = tmp_path/'metrics.jsonl.gz'
    with gzip.open(metrics, 'wt') as stream:
        stream.write('\n'.join(json.dumps(r) for r in fixture_rows()))
    (tmp_path/'config.json').write_text(json.dumps(CONFIG))
    result = main([str(metrics), str(tmp_path/'out'), '--no-plots'])
    assert result['artifact']['metrics_sha256'] == hashlib.sha256(metrics.read_bytes()).hexdigest()
    assert result['artifact']['metrics_file'] == 'metrics.jsonl.gz'
    assert result['aggregates'][0]['metrics']['exact_path_completion']['mean'] == .5


def test_standalone_help_without_site_packages(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import tensegra.binding_analysis as analysis
    result = subprocess.run([sys.executable, '-S', str(Path(analysis.__file__)), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--no-plots' in result.stdout
