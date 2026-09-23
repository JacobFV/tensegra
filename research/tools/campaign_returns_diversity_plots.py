"""R02 fixed-update diversity curve from archived exact counts."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
rows=json.load(gzip.open(root/'predictions.json.gz','rt'));sizes=[4096,8192,16384]
fit=[];fresh=[];floats=[]
for count in sizes:
 arm=f'ce_{count}'
 train=next(r for r in rows if r['head']==arm and r['split']=='train' and r['target_delay']==16)
 val=next(r for r in rows if r['head']==arm and r['split']=='validation' and r['target_delay']==16 and r['distractors']==8)
 fit.append(100*train['in_pool_value']['correct']/count)
 fresh.append(100*val['counts']['value']['correct']/2048)
 cell=val['groups']['type']['1'];floats.append(100*cell['correct']/cell['total'])
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].plot(sizes,fit,'o-',label='Actual fitting prefix');axes[0].plot(sizes,fresh,'o-',label='Fresh validation, all types')
axes[1].plot(sizes,fresh,'o-',label='All types');axes[1].plot(sizes,floats,'o-',label='Float type only')
for ax in axes:
 ax.set_xscale('log',base=2);ax.set_xticks(sizes,[f'{n:,}' for n in sizes]);ax.set_ylim(93,100.4);ax.set_ylabel('Exact scalar (%)');ax.set_xlabel('Unique fitting events; 900 updates fixed');ax.grid(alpha=.2);ax.legend()
fig.suptitle('R02 selected backbone 11: sixteen-step acquisition versus fresh-context tails')
fig.tight_layout();fig.savefig(root/'diversity-curve.png',dpi=180)
