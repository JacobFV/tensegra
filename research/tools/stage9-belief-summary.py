"""Aggregate frozen per-cell metrics without selecting models or thresholds."""
import json,argparse
from pathlib import Path
ORIGINAL={'clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty'}
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('research/results/stage9/beliefs'));a=p.parse_args()
rows=[];manifests=[]
for folder in sorted(a.root.glob('*-[012]')):
 if (folder/'metrics.json').exists():rows+=json.loads((folder/'metrics.json').read_text());manifests.append(json.loads((folder/'manifest.json').read_text()))
gates=[]
for mode in ('protected','recurrent'):
 for arm in ('learned_prior','supplied_empty_prior'):
  for scope in ('original','expanded','additional_n32'):
   for seed in range(3):
    subset=[r for r in rows if r['mode']==mode and r['arm']==arm and r['seed']==seed and r['split']=='validation' and ((r['candidates']==32) if scope=='additional_n32' else r['candidates'] in (8,16)) and (scope!='original' or r['condition'] in ORIGINAL)]
    expected=16 if scope=='original' else 24 if scope=='expanded' else 12
    fail=[{k:r[k] for k in ('candidates','condition','mean_l1','mean_impossible','final_support_accuracy')} for r in subset if not r['passed']]
    gates.append(dict(mode=mode,arm=arm,scope=scope,seed=seed,cells=len(subset),expected=expected,passed=len(subset)==expected and not fail,failures=fail))
summary=dict(kind='frozen_checkpoint_architectural_intervention_no_training',cells=len(rows),episode_evaluations=sum(r['count'] for r in rows),optimizer_presentations=0,actual_width=1024,parameters=16867395,total_seconds=sum(m['total_seconds'] for m in manifests),peak_cuda_bytes=max((m['cuda_peak_bytes'] for m in manifests),default=0),peak_rss_kib=max((m['rss_peak_kib'] for m in manifests),default=0),gates=gates,composition_allowed=False)
(a.root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='gates'},indent=2))
for mode in ('protected','recurrent'):
 for arm in ('learned_prior','supplied_empty_prior'):
  print(mode,arm,[(g['scope'],g['seed'],g['passed'],len(g['failures'])) for g in gates if g['mode']==mode and g['arm']==arm])
