#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('output');a=p.parse_args()
rows=json.loads(Path(a.summary).read_text())['rows'];fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,field,label in zip(axes,['value','joint'],['Exact scalar','Full semantic joint']):
 for arm in ['unchanged','ridge','ce_refit']:
  means=[];low=[];high=[];delays=[0,1,2,4,8,16,32]
  for step in delays:
   v=[r['counts'][field]['correct']/r['counts'][field]['total'] for r in rows if r['split']=='test' and r['arm']==arm and r['distractors']==8 and r['intervention']=='none' and r['steps']==step]
   means.append(sum(v)/len(v));low.append(min(v));high.append(max(v))
  line=ax.plot(range(len(delays)),means,marker='o',label=arm)[0];ax.fill_between(range(len(delays)),low,high,color=line.get_color(),alpha=.12)
 ax.set_xticks(range(len(delays)),delays);ax.set_xlabel('Recurrent updates');ax.set_ylabel(label+' accuracy');ax.set_ylim(.5,1.01);ax.grid(alpha=.2)
 if field=='value':ax.axhline(.98,color='gray',linestyle='--',linewidth=.8,label='16-step value threshold')
 ax.axvline(3.5,color='gray',linestyle=':',linewidth=.8);ax.legend(fontsize=8)
fig.suptitle('Frozen-workspace readout repair: mean and seed range, 512 test events, 8 distractors')
fig.tight_layout();fig.savefig(a.output,dpi=150)
