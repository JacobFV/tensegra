import math
import pytest
from topoformer.grounding_analysis import describe, paired_difference, validate_finite


def test_sample_sd_and_single_seed_is_not_zero_uncertainty():
    assert describe([1, 2, 3])["sd"] == 1
    assert describe([1])["sd"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_nested_metrics(value):
    with pytest.raises(ValueError, match="nonfinite"):
        validate_finite({"evaluations": [{"accuracy": value}]})
    with pytest.raises(ValueError):
        describe([value])


def test_pairing_preserves_direction_and_exposes_missing_seeds():
    result = paired_difference({0: .8, 1: .9}, {0: .6, 2: .5})
    assert result["summary"]["mean"] == pytest.approx(.2)
    assert result["summary"]["n"] == 1
    assert result["missing_control"] == [1]
    assert result["missing_treatment"] == [2]


def fixture_rows():
    from topoformer.grounding_analysis import METRICS
    rows = []
    for variant in ('soft', 'none'):
        for seed in (0, 1, 2):
            metrics = {metric: .5 for metric in METRICS}
            metrics['task_accuracy'] = .8 if variant == 'soft' else .6
            rows.append(dict(variant=variant, seed=seed, source={'commit': 'abc'}, config_hash='config',
                             training={'schedule_hash': f'seed{seed}', 'initial_parameter_hashes': {'weight': 'same'}},
                             resources={'seconds': 1}, model={'parameters': 1},
                             evaluations=[dict(condition='depth_4', depth=4, nodes=16, distractors=4,
                                               composition='train', corruption=0., data_hash=f'eval{seed}',
                                               **metrics, step_diagnostics=[dict(step=i, grounding_accuracy=.5, clean_next_attention_mass=.5) for i in range(4)], example=[])]))
    return rows


def test_summarize_requires_complete_matched_seed_grid():
    from topoformer.grounding_analysis import summarize
    rows = fixture_rows()
    result = summarize(rows, expected_seeds=[0, 1, 2])
    assert result['paired_task_differences'][0]['summary']['mean'] == pytest.approx(.2)
    assert result['paired_task_differences'][0]['initial_common_parameters_match']
    with pytest.raises(ValueError, match='missing expected seeds'):
        summarize(rows[:-1], expected_seeds=[0, 1, 2])
    with pytest.raises(ValueError, match='duplicate run'):
        summarize(rows + [rows[0]])
    rows[-1]['evaluations'][0]['data_hash'] = 'different'
    with pytest.raises(ValueError, match='unpaired data/schedule'):
        summarize(rows)


def test_no_distractors_is_unavailable_not_zero_accuracy():
    from topoformer.grounding_analysis import summarize, render_report
    rows = fixture_rows()
    for row in rows:
        row['evaluations'][0].update(distractors=0, distractor_null_accuracy=None)
    result = summarize(rows)
    assert result['aggregates'][0]['metrics']['distractor_null_accuracy'] is None
    assert 'N/A' in render_report(result)


def test_mixed_source_or_config_rejected():
    from topoformer.grounding_analysis import summarize
    rows = fixture_rows()
    rows[-1]['source']['commit'] = 'other'
    with pytest.raises(ValueError, match='mixed configuration or source'):
        summarize(rows)


def test_report_keeps_task_and_attention_path_separate():
    from topoformer.grounding_analysis import summarize, render_report
    report = render_report(summarize(fixture_rows()))
    assert 'Complete attention trajectory' in report
    assert 'Exact pre-step grounding' in report
    assert 'not a lexical-language test' in report
    assert 'sample SD' in report
