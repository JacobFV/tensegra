"""Decision and bootstrap fixtures; import helper without opening main results."""
import importlib.util,itertools,json
from pathlib import Path
import numpy as np
path=Path('research/campaigns/extended-01/semantics/S15-analyze.py');spec=importlib.util.spec_from_file_location('s15_analysis',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=[52,26,103,-51,52,26];d=m.decisions(*base);assert d['confirmation_eligible'] and not d['one_paired_extension_eligible']
for i in range(6):
 args=base.copy();args[i]-=1;d=m.decisions(*args)
 if i in (4,5):assert d['one_paired_extension_eligible'] and not d['confirmation_eligible']
 else:assert not d['one_paired_extension_eligible'] and not d['confirmation_eligible']
count=0
for values in itertools.product((51,52,53),(25,26,27),(102,103,104),(-52,-51,-50),(51,52,53),(25,26,27)):
 d=m.decisions(*values);assert not(d['confirmation_eligible'] and d['one_paired_extension_eligible']);count+=1
m.validate_visits([8]*4096)
for visits in ([8]*4095,[8]*4097,[7]+[8]*4095):
 try:m.validate_visits(visits)
 except AssertionError:pass
 else:raise AssertionError('bad visits accepted')
delta=np.tile(np.array([-1,0,1,0]),128);rng=np.random.default_rng(15015);samples=m.bootstrap_samples(delta,rng)
reference_rng=np.random.default_rng(15015);draws=reference_rng.integers(0,512,size=(10000,512));expected=np.sum(delta[draws],axis=1)/512
assert np.array_equal(samples,expected) and rng.integers(2**31)==reference_rng.integers(2**31)
assert np.array_equal(m.bootstrap_samples(np.zeros(512),np.random.default_rng(15015)),np.zeros(10000))
assert np.array_equal(m.bootstrap_samples(np.ones(512),np.random.default_rng(15015)),np.ones(10000))
print(json.dumps(dict(decision_boundary_checks=7,mutually_exclusive_grid_checks=count,visit_length_and_count_guards=True,bootstrap_fixed_draw_replay=True,bootstrap_degenerate_controls=True,no_main_results_opened=True)))
