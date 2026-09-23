"""Plot narrow R05 decision accuracy and paired intervention semantics."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=json.load(gzip.open(a.results/'predictions.json.gz','rt'))
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for arm,label in [('learned','Learned scalar access'),('oracle','Exact-value neural ceiling'),('query_only','Query only')]:
 cells=sorted([x for x in r if x['key']=='validation/8' and x['arm']==arm],key=lambda x:x['delay'])
 axes[0].plot(range(len(cells)),[100*x['original_correct']/x['total'] for x in cells],'o-',label=label)
axes[0].set_xticks(range(6),[0,1,2,4,8,16]);axes[0].set_title('Clean decisions, registered mixture');axes[0].set_xlabel('Recurrent updates');axes[0].set_ylabel('Task accuracy (%)');axes[0].legend(fontsize=7)
for key,label in [('validation/8','Correct event'),('intervention_drop/8','Event absent before ingestion')]:
 cells=sorted([x for x in r if x['key']==key and x['arm']=='learned' and x['delay'] in (0,1,16)],key=lambda x:x['delay'])
 axes[1].plot(range(3),[100*x['original_correct']/x['total'] for x in cells],'o-',label=label)
axes[1].set_xticks(range(3),[0,1,16]);axes[1].set_title('Paired event removal');axes[1].set_xlabel('Recurrent updates');axes[1].set_ylabel('Original-answer accuracy (%)');axes[1].legend(fontsize=7)
for arm,label in [('learned','Learned access'),('query_only','Query only')]:
 cells=[next(x for x in r if x['key']==key and x['arm']==arm and x['delay']==16) for key in ('intervention_wrong/8','intervention_swap/8')]
 x=[0,1];offset=-.16 if arm=='learned' else .16
 axes[2].bar([v+offset for v in x],[100*z['changed_supplied_correct']/z['changed_total'] for z in cells],width=.3,label=label)
 for v,z in zip(x,cells):axes[2].text(v+offset,100*z['changed_supplied_correct']/z['changed_total']+1,f"{z['changed_supplied_correct']}/{z['changed_total']}",ha='center',fontsize=7)
axes[2].set_xticks([0,1],['Wrong value','Swapped return']);axes[2].set_ylim(0,110);axes[2].set_title('Follow supplied fact when answer changes');axes[2].set_ylabel('Supplied-answer accuracy (%)');axes[2].legend(fontsize=7,loc='lower left')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('R05 development: one historical backbone; frozen return interface; learned comparator',fontsize=11)
fig.savefig(a.results/'use-and-interventions.png',dpi=160)
