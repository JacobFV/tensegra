"""A03 static learning/generalization plots from frozen compact results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
seeds=[201,202,203];arms=['soft4','soft8','context','message','hard']
summary=json.loads((a.root/'analysis.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4))
labels=['16/4','32/4','16/8','64/16','16/4 new','64/4','16/16','32/8']
for arm in arms:
 for ax,key in zip(axes,['seed_task','seed_pointer_path','seed_edge_mass']):
  y=np.array([r['arms'][arm][key] for r in summary['rows'][:8]])
  ax.plot(range(8),y.mean(1),marker='.',label=arm)
  ax.fill_between(range(8),y.min(1),y.max(1),alpha=.12)
  ax.set_xticks(range(8),labels,rotation=45);ax.set_xlabel('Nodes / depth');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
for ax,title in zip(axes,['Exact payload task','Mean-head argmax path diagnostic','Clean-edge attention mass']):ax.set_title(title)
axes[0].legend(fontsize=8);fig.suptitle('A03 confirmation: mean and initialization-seed range, 1024 paired events per cell');fig.tight_layout();fig.savefig(a.output/'a03-generalization.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for arm in arms:
 for ci,ax in enumerate(axes):
  runs=[]
  for seed in seeds:
   m=json.loads((a.root/f'{arm}-{seed}'/'manifest.json').read_text())
   runs.append([c['rows'][ci]['task'] for c in m['curves'] if c['step']<500])
  y=np.array(runs);steps=[c['step'] for c in m['curves'] if c['step']<500]
  ax.plot(steps,y.mean(0),marker='.',label=arm);ax.fill_between(steps,y.min(0),y.max(0),alpha=.12)
  ax.set_ylim(-.03,1.03);ax.set_xlabel('Optimizer updates (batch16)');ax.set_ylabel('Exact task accuracy');ax.grid(alpha=.2)
axes[0].set_title('N16 / D4');axes[1].set_title('N32 / D8');axes[0].legend(fontsize=8)
fig.suptitle('A03 intermediate monitoring data: 256 events; independent of final confirmation');fig.tight_layout();fig.savefig(a.output/'a03-learning.svg')
