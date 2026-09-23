"""Static scientific acquisition/transfer summary; archived data only."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('acquisition');p.add_argument('summary');p.add_argument('output');a=p.parse_args()
x=json.load(gzip.open(a.acquisition,'rt'));summary=json.loads(Path(a.summary).read_text())
plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
colors=['#0072B2','#D55E00','#009E73']
for i,run in enumerate(x['runs']):
 points=run['curves'];axes[0].plot([p['update'] for p in points],[p['raw_exact']/8 for p in points],label=f"Seed {run['seed']}",color=colors[i],marker=['o','s','^'][i],markersize=7-i,linewidth=2-i*.4,linestyle=['-','--',':'][i])
axes[0].set(title='Fixed TRAIN set: 8 graphs per seed',xlabel='Optimizer updates (8 presentations/update)',ylabel='Complete canonical graph accuracy',ylim=(-.03,1.06));axes[0].legend(loc='lower right');axes[0].grid(alpha=.2)
axes[0].text(.04,.88,'Raw = calibrated at every checkpoint',transform=axes[0].transAxes,fontsize=9)
labels=['Complete\ngraph','Type','Exact\ncopy','Typed\nedge F1','Ordered\nedge F1'];locations=np.arange(len(labels))
for i,seed in enumerate([30,31,32]):
 row=next(r for r in summary if r['seed']==seed and r['decoder']=='raw')
 values=[row['semantic_equivalence'],row['node_type_accuracy'],row['identity_copy_accuracy'],row['typed_edge']['f1'],row['ordered_edge']['f1']]
 axes[1].bar(locations+(i-1)*.25,values,width=.24,color=colors[i],label=f'Seed {seed}')
axes[1].set(title='Fresh known English: 512 graphs per seed',xticks=locations,xticklabels=labels,ylabel='Accuracy / micro F1',ylim=(0,1.06));axes[1].text(-.37,.045,'0/512\neach seed',fontsize=9);axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
fig.suptitle('Stage 11: complete acquisition does not transfer from eight constructions',fontsize=13)
fig.savefig(a.output);plt.close(fig)
