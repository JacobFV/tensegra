"""Plot frozen R03 raw outcomes; no fitting or selection."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
rows=json.load(gzip.open(a.results/'predictions.json.gz','rt'))
steps=[900,1800,3600]
fig,axes=plt.subplots(1,2,figsize=(10,3.7))
for split,label in [('train','Fixed fitting pool'),('calibration','Calibration'),('validation','Reused development validation')]:
    values=[]
    for step in steps:
        matches=[r for r in rows if r['head']==f'ce_{step}' and r['split']==split and r['target_delay']==16 and r['distractors']==(2 if split=='train' else 8)]
        c=matches[0]['counts']['value'];values.append(100*c['correct']/c['total'])
    axes[0].plot(steps,values,'o-',label=label)
for split,label in [('calibration','Calibration minimum'),('validation','Validation minimum')]:
    values=[]
    for step in steps:
        cs=[r['counts']['value'] for r in rows if r['head']==f'ce_{step}' and r['split']==split]
        values.append(min(100*c['correct']/c['total'] for c in cs))
    axes[1].plot(steps,values,'o-',label=label)
for ax in axes:
    ax.set_xlabel('CE optimizer updates');ax.set_ylabel('Exact scalar accuracy (%)');ax.set_xticks(steps);ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_title('Delay 16: fitting improves more than fresh accuracy')
axes[1].set_title('Worst covered cell: selection retains 900')
fig.suptitle('R03 development: same frozen backbone,16k events and continuous optimizer trajectory',fontsize=10)
fig.tight_layout();fig.savefig(a.results/'exposure-curve.png',dpi=160)
