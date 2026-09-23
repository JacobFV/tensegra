"""Render supplemental controlled-study figures from the compact summary only."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
summary=json.loads((root/'summary.json').read_text())
output=root/'figures';output.mkdir(exist_ok=True)
final=max(c['step'] for c in summary['aggregates'])
variants=['local','global','neural_fixed','neural_recurrent','fixed_compute','runtime_off','protected_learned','task_only_cold','task_only_warm','anneal_all']
for metric,name in [('microsteps','microsteps-by-depth'),('task_accuracy','task-accuracy-by-depth')]:
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for variant in variants:
        cells=sorted((c for c in summary['aggregates'] if c['variant']==variant and c['condition']=='depth' and c['step']==final),key=lambda c:c['depth'])
        ax.plot([c['depth'] for c in cells],[c['metrics'][metric]['mean'] for c in cells],marker='o',label=variant,alpha=.75)
    ax.set(xlabel='Depth',ylabel=metric,xticks=[4,8,16,32]);ax.legend(fontsize=6,ncol=2)
    fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
for variant in variants:
    cells=sorted((c for c in summary['aggregates'] if c['variant']==variant and c['condition']=='depth' and c['depth']==4),key=lambda c:c['step'])
    ax.plot([c['step'] for c in cells],[c['metrics']['task_accuracy']['mean'] for c in cells],marker='o',label=variant,alpha=.75)
ax.set(xlabel='Optimizer updates',ylabel='Task accuracy, depth 4',ylim=(-.01,.2));ax.legend(fontsize=6,ncol=2)
fig.savefig(output/'task-accuracy-over-training.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,metric in zip(axes,['exact_semantic_accuracy','task_accuracy']):
    for condition,label in [('depth','local free policy'),('oracle_minimal','privileged minimal trace'),('oracle_trace','privileged 80-step trace')]:
        cells=[c for c in summary['aggregates']+summary['privileged_oracles'] if c['variant']=='local' and c['condition']==condition and c['step']==final and c['depth'] in (4,32)]
        ax.plot([c['depth'] for c in cells],[c['metrics'][metric]['mean'] for c in cells],marker='o',label=label)
    ax.set(xlabel='Depth',ylabel=metric,xticks=[4,32],ylim=(-.03,1.03))
axes[0].legend(fontsize=7);fig.savefig(output/'privileged-oracle-vs-policy.png',dpi=150);plt.close(fig)
