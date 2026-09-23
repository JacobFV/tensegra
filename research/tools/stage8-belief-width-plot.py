"""Matched-task width contrast; CPU32 versus CUDA1024 is explicitly disclosed."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]); fig,axes=plt.subplots(1,3,figsize=(12,3.7))
for mode,color in [('protected','#247ba0'),('recurrent','#e76f51')]:
    for ax,condition,key,title in zip(axes,['clean','retract','long_duplicate'],['support_accuracy']*3,['Clean','Retraction','Eight-fold duplicate IDs']):
        means=[]; lows=[]; highs=[]
        for width,folder in [(32,'historical32'),(1024,'idmatched')]:
            values=[]
            for seed in range(3):
                rows=json.loads((root/folder/f'{mode}-{seed}'/'metrics.json').read_text())
                row=next(r for r in rows if r['split']=='test' and r['candidates']==16 and r['condition']==condition)
                values.append(row['frames'][-1][key])
            mean=sum(values)/3;means.append(mean);lows.append(mean-min(values));highs.append(max(values)-mean)
        ax.errorbar([32,1024],means,yerr=[lows,highs],color=color,label=mode,marker='o',capsize=4)
        ax.set_xscale('log',base=2);ax.set_xticks([32,1024],['32 (historical)','1024 (primary)']);ax.set_ylim(-.04,1.05);ax.set_title(title);ax.set_ylabel('Final proposal accuracy, N=16');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Same task, supervision and exposures; mean and seed range (3 seeds)')
fig.tight_layout();fig.savefig(root/'width-comparison.png',dpi=170)
