"""Render isolated belief JSON without loading checkpoints or torch."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('directory'); a=p.parse_args(); root=Path(a.directory)
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
