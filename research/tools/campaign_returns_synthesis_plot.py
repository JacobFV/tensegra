"""Plot paired development acquisition and scalar-boundary limits."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();rows=json.load(open(a.results/'comparison.json'))['rows'];x=np.arange(len(rows));w=.35
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for off,split,label in [(-w/2,'train_balanced','Each arm’s fitting events'),(w/2,'validation_grid','Fresh balanced contexts')]:
 c=[r['counts']['train_original' if split=='train_balanced' and r['arm']=='original' else split]['16']['value'] for r in rows];axes[0].bar(x+off,[v['correct']/v['total'] for v in c],w,label=label)
for off,key,label in [(-w/2,'calibration_grid_minimum','Calibration'),(w/2,'validation_grid_minimum','Reused development validation')]:
 c=[r[key] for r in rows];axes[1].bar(x+off,[v['correct']/v['total'] for v in c],w,label=label)
for ax in axes:ax.set_xticks(x,[r['label'].replace(' ','\n') for r in rows]);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
axes[0].set(ylim=(.98,1.001),ylabel='Exact scalar accuracy',title='Delay16: acquisition versus fresh contexts\nTruncated axis; aggregate grid accuracy can hide whole-value tails')
axes[1].set(ylim=(0,1.03),ylabel='Worst legal type/value cell accuracy',title='Worst stratum across delays0/1/16\n64 events per stratum; calibration drives advancement')
fig.suptitle('Return diagnostics: shared frozen backbone and reused R06 populations')
fig.savefig(a.results/'return-diagnostics.png',dpi=160)
