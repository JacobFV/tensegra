"""Static scientific figures; no fitting or threshold selection."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
rows=[]
for path in root.glob('*-predictions.json.gz'):rows.extend(json.load(gzip.open(path,'rt')))
delays=[0,1,2,4,8,16,32]
styles={'frozen_original':('Frozen original','#777777'),
        'frozen_shared_readout':('Frozen + historical shared ridge','#bc8f00'),
        'short_continue':('Short continuation (trained through 4)','#3679ba'),
        'wide_continue':('Wide continuation (trained through 16)','#db6237')}
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for col,d in enumerate((2,8)):
 for row,field in enumerate(('value','joint')):
  ax=axes[row,col]
  for head,(label,color) in styles.items():
   selected=[r for r in rows if r['split']=='test' and r['head']==head and r['distractors']==d]
   values=[]
   for delay in delays:
    cells=[r['counts'][field] for r in selected if r['target_delay']==delay]
    values.append(100*sum(c['correct'] for c in cells)/sum(c['total'] for c in cells))
    ax.scatter([delays.index(delay)]*len(cells),[100*c['correct']/c['total'] for c in cells],s=10,color=color,alpha=.45)
   ax.plot(range(len(delays)),values,'o-',label=label,color=color,lw=1.6,ms=4)
  ax.axvline(5.5,color='black',linestyle=':',lw=1)
  ax.set_title(f'{"Exact scalar" if field=="value" else "All-six-field joint"}, {d} distractors')
  ax.set_xticks(range(len(delays)),delays);ax.set_ylim(0,102);ax.grid(alpha=.2)
  if col==0:ax.set_ylabel('Correct (%)')
  if row==1:ax.set_xlabel('Recurrent updates (32 is OOD for both continuations)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005))
fig.suptitle('Stage 11: same frozen initialization, matched fresh-example exposure, different horizons')
fig.tight_layout(rect=(0,.085,1,.96));fig.savefig(root/'return-horizons.png',dpi=180);fig.savefig(root/'return-horizons.pdf')
