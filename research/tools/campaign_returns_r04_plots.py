"""Static R04 scientific figures reconstructed from raw counts."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
rows={s:json.load(gzip.open(a.results/str(s)/'predictions.json.gz','rt')) for s in (10,11,12)}
fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True)
for i,seed in enumerate(rows):
 for j,field in enumerate(('value','nonvalue_joint','joint')):
  ax=axes[i,j]
  for head,label in [('unchanged','Frozen original'),('ce_4096','CE:4k events'),('ce_16384','CE:16k events')]:
   data=sorted((r for r in rows[seed] if r['split']=='test' and r['head']==head and r['distractors']==8),key=lambda r:r['target_delay'])
   ax.plot(range(len(data)),[100*r['counts'][field]['correct']/r['counts'][field]['total'] for r in data],'o-',label=label)
  ax.axvline(5.5,color='gray',ls=':');ax.set_title(f'Backbone {seed}: {field.replace("_"," ")}');ax.set_xticks(range(7),[0,1,2,4,8,16,32]);ax.grid(alpha=.2)
  if j==0:ax.set_ylabel('Accuracy (%)')
  if i==2:ax.set_xlabel('Recurrent updates')
  if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('R04 fresh test, eight distractors: 0–16 covered;32 extrapolation',fontsize=13)
fig.tight_layout();fig.savefig(a.results/'readout-curves.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(13,7),sharex=True,constrained_layout=True)
for ax,seed in zip(axes,rows):
 data=[]
 for delay in (0,1,16,32):
  r=next(r for r in rows[seed] if r['split']=='balanced' and r['head']=='ce_16384' and r['target_delay']==delay)
  row=[]
  for value in range(33):
   ids=[i for i,(v,t) in enumerate(zip(r['targets']['value'],r['targets']['type'])) if v==value and t==1]
   assert len(ids)==64
   row.append(sum(r['predictions']['value'][i]!=value for i in ids))
  data.append(row)
 im=ax.imshow(data,vmin=0,vmax=64,cmap='magma',aspect='auto');ax.set_yticks(range(4),[0,1,16,32]);ax.set_ylabel(f'Backbone {seed}\ndelay')
 for i,row in enumerate(data):
  for j,count in enumerate(row):
   if count:ax.text(j,i,str(count),ha='center',va='center',fontsize=7,color='black' if count>40 else 'white')
axes[-1].set_xticks(range(33),[f'{(v-16)/2:g}' for v in range(33)],rotation=45);axes[-1].set_xlabel('True float value (64 fresh events per cell)')
fig.colorbar(im,ax=axes,label='Exact-value errors /64');fig.suptitle('R04 16k consumer: balanced float grid exposes localized blind spots')
fig.savefig(a.results/'balanced-float-errors.png',dpi=160)
