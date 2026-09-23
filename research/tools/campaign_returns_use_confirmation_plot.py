"""Three-run R05 confirmation, preserving paired causal denominators."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
for i,seed in enumerate([10,11,12]):
 rows=json.load(gzip.open(a.results/str(seed)/'predictions.json.gz','rt'));summary=json.load(open(a.results/str(seed)/'summary.json'))
 for arm,label in [('learned','Learned access'),('oracle','Exact-value neural ceiling'),('query_only','Query only')]:
  cells=sorted((x for x in rows if x['arm']==arm and x['key']=='test/8'),key=lambda x:x['delay'])
  axes[i,0].plot(range(7),[100*x['original_correct']/x['total'] for x in cells],'o-',label=label)
 axes[i,0].set_xticks(range(7),[0,1,2,4,8,16,32]);axes[i,0].axvline(5.5,color='gray',ls=':');axes[i,0].set_title(f'Backbone {seed}: fresh task test');axes[i,0].set_ylabel('Accuracy (%)')
 cells=sorted((x for x in summary['causal'] if 'drop' in x['key']),key=lambda x:x['delay'])
 axes[i,1].bar(range(3),[100*x['paired_original_accuracy_drop'] for x in cells],color='tab:purple');axes[i,1].axhline(15,color='gray',ls=':');axes[i,1].set_xticks(range(3),[0,1,16]);axes[i,1].set_ylim(0,36);axes[i,1].set_title('Paired event-removal effect');axes[i,1].set_ylabel('Original-answer drop (pp)')
 for key,label in [('intervention_wrong/8','Wrong value'),('intervention_swap/8','Swapped return')]:
  cells=sorted((x for x in summary['causal'] if x['key']==key),key=lambda x:x['delay'])
  axes[i,2].plot(range(3),[100*x['changed_supplied_accuracy'] for x in cells],'o-',label=label)
 axes[i,2].set_xticks(range(3),[0,1,16]);axes[i,2].set_ylim(90,100.5);axes[i,2].set_title('Follow supplied, changed answer');axes[i,2].set_ylabel('Conditional accuracy (%)')
 for j in range(3):axes[i,j].grid(axis='y',alpha=.2);axes[i,j].set_xlabel('Recurrent updates')
axes[0,0].legend(fontsize=7,loc='center left');axes[0,2].legend(fontsize=7,loc='lower left')
fig.suptitle('R05 restricted confirmation: learned finite-domain scalar consumer; teacher first return',fontsize=14)
fig.savefig(a.results/'confirmation-use.png',dpi=160)
