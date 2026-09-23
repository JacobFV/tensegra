"""Reproduce A06 learning and condition curves without GPU."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
colors=dict(soft='#2563eb',hard='#16a34a',context='#d97706',none='#6b7280')
fig,axs=plt.subplots(1,3,figsize=(13,3.5))
for mode,color in colors.items():
 ms=[json.loads((root/str(s)/mode/'manifest.json').read_text()) for s in [601,602,603]]
 for j,cell in enumerate([0,1]):
  xx=[c['step'] for c in ms[0]['curves'][:-1]]
  yy=np.array([[c['rows'][cell]['task'] for c in m['curves'][:-1]] for m in ms]);axs[j].plot(xx,yy.mean(0),label=mode,color=color);axs[j].fill_between(xx,yy.min(0),yy.max(0),color=color,alpha=.15)
 rows=[m['curves'][-1]['rows'] for m in ms]
 yy=np.array([[r[i]['task'] for i in [0,2,6]] for r in rows]);axs[2].plot([4,8,32],yy.mean(0),'o-',label=mode,color=color);axs[2].fill_between([4,8,32],yy.min(0),yy.max(0),color=color,alpha=.15)
for ax,title in zip(axs,['Monitoring IID','Monitoring N64/D8','Final N32/K4: depth shift']):ax.set_title(title);ax.set_ylim(0,1.03);ax.set_ylabel('Exact task accuracy');ax.grid(alpha=.2)
for ax in axs[:2]:ax.set_xscale('symlog',linthresh=10);ax.set_xlabel('Optimizer updates')
axs[2].set_xlabel('Supplied path depth');axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'a06-learning-depth.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.5))
for mode,color in colors.items():
 rows=[json.loads((root/str(s)/mode/'eval-01000.json').read_text())['rows'] for s in [601,602,603]]
 vals=np.array([[r[i]['task'] for i in [1,5]] for r in rows]);axs[0].plot([4,8],1-vals.mean(0),'o-',label=mode,color=color)
 idx=[8,18]+([21] if mode=='context' else [])
 vals=np.array([[r[i]['task'] for i in idx] for r in rows]);axs[1].plot(range(len(idx)),1-vals.mean(0),'o-',label=mode,color=color);axs[1].fill_between(range(len(idx)),1-vals.max(0),1-vals.min(0),color=color,alpha=.15)
axs[0].set(title='N64/D4: neighbor count',xlabel='K');axs[1].set(title='N128/D32/K8: frozen score policies',xticks=[0,1,2],xticklabels=['Learned','Content16','Both16'])
for ax in axs:ax.set_ylabel('Task error rate');ax.grid(alpha=.2);ax.set_ylim(-.02,1)
axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'a06-neighbors-concentration.svg')
