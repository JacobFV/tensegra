"""Static return-retention plots from the auditable aggregate summary."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
root=Path(args.directory); rows=json.loads((root/'summary.json').read_text())['cells']
rows=[r for r in rows if r['split']=='test' and r['intervention']=='none']
steps=[0,1,2,4,8,16,32]
encodings=['mixed','compressed','factorized']
colors=dict(mixed='#a33b35',compressed='#426ab3',factorized='#268657')
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for ax,distractors in zip(axes,[2,8]):
    for encoding in encodings:
        for availability in ['once','persistent']:
            groups=[[r['counts']['joint']['correct']/r['counts']['joint']['total'] for r in rows if r['distractors']==distractors and r['encoding']==encoding and r['availability']==availability and r['steps']==step] for step in steps]
            means=[np.mean(g) for g in groups]
            ax.plot(range(len(steps)),means,label=f'{encoding} / {availability}',color=colors[encoding],linestyle='-' if availability=='persistent' else '--',marker='o',markersize=3)
            ax.fill_between(range(len(steps)),[min(g) for g in groups],[max(g) for g in groups],color=colors[encoding],alpha=.07)
    ax.set_title(f'{distractors} unrelated memory rows'); ax.set_xticks(range(len(steps)),steps)
    ax.set_xlabel('Recurrent microsteps (0 = initial learned read)'); ax.set_ylim(0,1);ax.grid(alpha=.2)
axes[0].set_ylabel('All six semantic fields correct')
axes[1].legend(fontsize=8)
fig.suptitle('Width 1024 return retention — mean and range over three paired seeds')
fig.tight_layout();fig.savefig(root/'retention.png',dpi=160);plt.close(fig)
fields=['value','type','operation','argument0','argument1_required','provenance']
fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
labels=[f'{e}\n{a}' for e in encodings for a in ['once','persistent']]
for ax,field in zip(axes.flat,fields):
    for j,(encoding,availability) in enumerate((e,a) for e in encodings for a in ['once','persistent']):
        group=[r['counts'][field]['correct']/r['counts'][field]['total'] for r in rows if r['distractors']==8 and r['steps']==16 and r['encoding']==encoding and r['availability']==availability]
        ax.bar(j,np.mean(group),color=colors[encoding],alpha=.8 if availability=='persistent' else .45)
        ax.plot([j,j],[min(group),max(group)],color='black',linewidth=1)
    ax.axhline(.99 if field in ('type','operation') else .98,color='gray',linestyle=':',linewidth=1)
    ax.set_title(field);ax.set_xticks(range(6),labels,fontsize=7);ax.set_ylim(0,1.04);ax.grid(axis='y',alpha=.2)
fig.suptitle('16-step field retention with eight distractors — mean and seed range; dotted competence threshold')
fig.tight_layout();fig.savefig(root/'fields16.png',dpi=160)
