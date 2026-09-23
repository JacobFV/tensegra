"""Static figures from auditable per-run manifests, no model inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
fig, axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for manifest in sorted(args.root.glob('*/manifest.json')):
    data=json.loads(manifest.read_text())
    curves=data['curves']
    for row,ci in enumerate([0,3]):
        for col,key in enumerate(['task','exact_pointer_path','edge_mass']):
            axes[row,col].plot([c['step'] for c in curves], [c['rows'][ci][key] for c in curves],
                               marker='.',label=manifest.parent.name)
            axes[row,col].set_ylim(-.03,1.03)
            axes[row,col].set_title(f"N{curves[0]['rows'][ci]['condition']['nodes']} D{curves[0]['rows'][ci]['condition']['depth']} — {key}")
            axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('optimizer updates')
axes[0,0].legend(fontsize=8)
fig.suptitle('A01 exploratory seed101: value accuracy versus mean-head routing diagnostics')
fig.tight_layout()
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output)
