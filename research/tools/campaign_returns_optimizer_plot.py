"""Static scientific R08 plots; every screened rate remains visible."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
m=json.load(gzip.open(a.results/'manifest.json.gz','rt'));s=json.load(open(a.results/'summary.json'))
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for f in m['fits']:
 label=f['arm'].replace('residual_lr_','LR ')
 if not f['trained_here']:continue
 loss=np.asarray(f['losses']);axes[0].plot(np.arange(25,len(loss)+1),np.convolve(loss,np.ones(25)/25,mode='valid'),label=label)
 steps=[c['step'] for c in f['curve']]
 for ax,key in zip(axes[1:],['grid','cells']):ax.plot(steps,[min(c['correct']/c['total'] for c in cp[key]) for cp in f['curve']],marker='.',label=label)
for f in m['fits']:
 if f['trained_here']:continue
 for ax,key in zip(axes[1:],['grid','cells']):ax.axhline(min(c['correct']/c['total'] for c in f['curve'][0][key]),ls='--',label=f['arm'].replace('_',' '))
axes[0].set(yscale='log',ylabel='Cross-entropy (25-update mean)',title='Fitting loss, all screened rates')
axes[1].set(ylabel='Minimum exact accuracy',title='Calibration type/value grid')
axes[2].set(ylabel='Minimum exact accuracy',title='Original-mixture calibration')
for ax in axes:ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('R08: fixed residual consumer, optimizer-only development screen')
fig.savefig(a.results/'optimizer-curves.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
names=list(s['arms']);x=np.arange(len(names));w=.35
for offset,split,label in [(-w/2,'calibration_grid','Calibration'),(w/2,'validation_grid','Reused development validation')]:
 ax.bar(x+offset,[s['arms'][n][split]['correct']/s['arms'][n][split]['total'] for n in names],w,label=label)
ax.set_xticks(x,[n.replace('residual_lr_','LR ').replace('_','\n') for n in names]);ax.set(ylabel='Worst legal type/value cell accuracy',ylim=(0,1.03),title='Fixed900 endpoints; no validation selection');ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(a.results/'optimizer-grid-minima.png',dpi=160)
