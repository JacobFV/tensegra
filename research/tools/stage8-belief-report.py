"""Render isolated belief JSON without loading checkpoints."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('--gates',action='store_true'); p.add_argument('--status',default='isolated_belief'); a=p.parse_args(); root=Path(a.directory)
runs={path.parent.name:json.loads(path.read_text()) for path in root.glob('*/metrics.json')}
summary=[]
for name,rows in runs.items():
    for r in rows:
        frames=r['frames']; summary.append(dict(run=name,seed=r['seed'],split=r['split'],condition=r['condition'],candidates=r['candidates'],count=r['count'],accuracy=frames[-1]['support_accuracy'],mean_l1=sum(f['posterior_l1'] for f in frames)/len(frames),mean_impossible=sum(f['impossible_mass'] for f in frames)/len(frames)))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,4))
for name,rows in runs.items():
    for r in rows:
        if r['condition']!='clean' or r['split']!='validation': continue
        label=f'{name}, N={r["candidates"]}'
        for ax,key in zip(axes,('support_accuracy','posterior_l1','impossible_mass')):
            ax.plot([f['frame'] for f in r['frames']],[f[key] for f in r['frames']],marker='.',label=label); ax.set_title(key); ax.set_xlabel('Evidence frame')
axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(root/'belief-trajectories.png',dpi=160)

if a.gates:
    from topoformer.belief_study import gate
    modes={}
    conditions=('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty')
    for mode in ('protected','recurrent'):
        rows=sum([runs.get(f'{mode}-{seed}',[]) for seed in range(3)],[])
        failures=[]
        for r in rows:
            if r['split']!='validation': continue
            fs=r['frames']; reasons=[]
            if fs[-1]['support_accuracy'] <= (.98 if r['regime']=='iid' else .95): reasons.append('final_support_accuracy')
            if sum(f['posterior_l1'] for f in fs)/len(fs)>=.05: reasons.append('posterior_l1')
            if sum(f['impossible_mass'] for f in fs)/len(fs)>=.01: reasons.append('impossible_mass')
            if reasons: failures.append(dict(seed=r['seed'],candidates=r['candidates'],condition=r['condition'],reasons=reasons))
        modes[mode]={'passed':gate(rows,[(seed,c,n) for seed in range(3) for c in conditions for n in (8,16)]),'expected_validation_cells':48,'failed_cells':failures}
    (root/'gates.json').write_text(json.dumps({'status':a.status,'composition_allowed':False,'modes':modes},indent=2))
