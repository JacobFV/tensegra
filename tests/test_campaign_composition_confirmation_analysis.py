"""Analysis contracts; no model execution or GPU access."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np

_source=Path(__file__).resolve().parents[1]/'research/tools/campaign_composition_confirmation_analysis.py'
_spec=importlib.util.spec_from_file_location('c04_analysis',_source)
analysis=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def test_refusals_stay_in_original_and_changed_denominators():
    n=512; original=np.zeros(n,dtype=int); clean=original.copy(); drop=np.ones(n,dtype=int)
    supplied=np.ones(n,dtype=int); prediction=supplied.copy();prediction[:64]=-1
    report=analysis.causal_counts(clean,drop,drop,clean,original,
        {'wrong':(prediction,supplied),'swap':(prediction,supplied)})
    assert report['attempted']==512
    assert report['changed_controls']['wrong']['changed_support']==512
    assert report['changed_controls']['wrong']['changed_supplied_correct']==448
    assert report['changed_controls']['wrong']['prediction_refused']==64
    assert report['passed'] is False


def test_nonpositive_normalization_is_undefined_and_failed():
    original=np.zeros(512,dtype=int); wrong=np.ones(512,dtype=int)
    report=analysis.causal_counts(original,wrong,original,original,original,
        {'wrong':(wrong,wrong),'swap':(wrong,wrong)})
    assert report['normalized_gain'] is None
    assert report['passed'] is False


def test_bootstrap_keeps_pairing_and_fixed_seed_dispersion():
    zero=analysis.bootstrap({0:np.zeros(100),1:np.zeros(100),2:np.zeros(100)},100)
    assert zero['event_bootstrap_percentile95']==[0.,0.]
    difference={0:np.ones(100),1:-np.ones(100),2:np.zeros(100)}
    result=analysis.bootstrap(difference,100)
    assert result['equal_lineage_mean']==0
    assert result['event_bootstrap_percentile95']==[0.,0.]
    assert result['seed_sample_sd']==1
    assert result==analysis.bootstrap(difference,100)


def test_missing_lineages_are_pending_not_failed_or_passing():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        result=analysis.run(root,root/'pending.json',100)
        assert result['status']=='pending'
        assert result['aggregate_gates']=='pending'
        assert len(result['lineages'])==3
        assert result['comparisons']=={}
        assert all(x['status']=='pending' for x in result['lineages'].values())
