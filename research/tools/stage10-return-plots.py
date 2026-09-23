#!/usr/bin/env python3
"""Static scientific plots; all seed cells remain visible."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('output');a=p.parse_args()
data=json.loads(Path(a.summary).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
for matrix in data['matrices']:
    ax=axes[0 if matrix['distractors']==2 else 1,matrix['seed']-10]
    values=[[c['correct']/c['total'] for c in row] for row in matrix['cells']]
    im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='equal')
    for i,row in enumerate(values):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=7,color='black' if value>.55 else 'white')
    ax.set_xticks(range(7),matrix['targets']);ax.set_yticks(range(7),matrix['sources'])
    ax.set_xlabel('Evaluation delay');ax.set_ylabel('Decoder fitting delay')
    ax.set_title(f"Frozen seed {matrix['seed']}, {matrix['distractors']} distractors")
fig.colorbar(im,ax=axes.ravel().tolist(),label='Exact scalar accuracy',shrink=.85)
fig.suptitle('Cross-delay readout transfer: 512 paired test events per cell\nThe source-32 row is a trained-delay diagnostic, not extrapolation',fontsize=13)
fig.savefig(out/'cross-delay-matrices.png',dpi=160)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,distractors in zip(axes,(2,8)):
    for head in ('original','short_pool','shared_pool'):
        means=[];low=[];high=[]
        for delay in (0,1,2,4,8,16,32):
            rows=[r for r in data['curves'] if r['head']==head and r['distractors']==distractors and r['delay']==delay]
            values=[r['counts']['value']['correct']/r['counts']['value']['total'] for r in rows]
            means.append(sum(values)/len(values));low.append(min(values));high.append(max(values))
        line=ax.plot(range(7),means,marker='o',label=head)[0]
        ax.fill_between(range(7),low,high,color=line.get_color(),alpha=.13)
    ax.axhline(.98,color='gray',linestyle='--',linewidth=.7,label='Scalar threshold')
    ax.axvline(3.5,color='gray',linestyle=':',linewidth=.7)
    ax.axvline(5.5,color='black',linestyle=':',linewidth=.7)
    ax.set_xticks(range(7),(0,1,2,4,8,16,32));ax.set_ylim(.6,1.01);ax.grid(alpha=.2)
    ax.set_xlabel('Evaluation delay');ax.set_ylabel('Exact scalar accuracy');ax.set_title(f'{distractors} distractors')
    ax.legend(fontsize=8)
fig.suptitle('Matched 8,192-row pooled readouts: mean and seed range\nShort pool trains through 4; shared pool through 16; 32 is unseen by both pooled fits',fontsize=11)
fig.savefig(out/'shared-readout-curves.png',dpi=160)
