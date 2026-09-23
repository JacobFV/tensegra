"""Scientific plots of R01 confirmation; no model selection or fitting."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
rows=[]
for directory in sorted(root.iterdir()):
 if directory.is_dir():rows.extend(json.load(gzip.open(directory/'predictions.json.gz','rt')))
delays=[0,1,2,4,8,16,32];styles={'unchanged':('Original wide head','#777777'),'ridge':('Fixed ridge .01','#b68a00'),'ce':('Fixed CE 900','#256cad')}
fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey='row')
for col,seed in enumerate((10,11,12)):
 for row,field in enumerate(('value','nonvalue_joint','joint')):
  ax=axes[row,col]
  for arm,(label,color) in styles.items():
   values=[]
   for delay in delays:
    record=next(r for r in rows if r['seed']==seed and r['head']==arm and r['split']=='test' and r['distractors']==8 and r['target_delay']==delay)
    cell=record['counts'][field];values.append(100*cell['correct']/cell['total'])
   ax.plot(range(7),values,'o-',label=label,color=color,markersize=3)
  ax.axvline(5.5,color='black',ls=':',lw=1);ax.set_xticks(range(7),delays);ax.grid(alpha=.2)
  ax.set_ylim(70,101)
  if row==0:ax.set_title(f'Frozen historical backbone {seed}')
  if col==0:ax.set_ylabel({'value':'Exact scalar (%)','nonvalue_joint':'Non-value joint (%)','joint':'All-six-field joint (%)'}[field])
  if row==2:ax.set_xlabel('Updates; 32 is out of fitting coverage')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
fig.suptitle('R01 fresh confirmation test, 1,024 paired events per backbone, eight distractors')
fig.tight_layout(rect=(0,.045,1,.97));fig.savefig(root/'r01-confirmation-curves.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
confusion={}
for ax,typ,name in zip(axes,(0,1),('Integer','Float')):
 support=[0]*33;errors=[0]*33;matrix=[[0]*33 for _ in range(33)]
 for record in rows:
  if record['head']!='ce' or record['split']!='test' or record['distractors']!=8 or record['target_delay']!=16:continue
  for t,p,tau in zip(record['targets']['value'],record['predictions']['value'],record['targets']['type']):
   if tau==typ:support[t]+=1;errors[t]+=int(t!=p);matrix[t][p]+=1
 x=[(i-16)/2 for i in range(33)];rates=[100*e/n if n else float('nan') for e,n in zip(errors,support)]
 ax.bar(x,support,width=.35,color='#cccccc',label='Support (left axis)');ax.set_ylabel('Observed examples');ax.set_title(f'{name} returns; exact CE reconstruction at 16 updates')
 other=ax.twinx();other.plot(x,rates,'o-',color='#bd4430',label='Error percentage (right axis)');other.set_ylabel('Errors (%)',color='#bd4430');other.set_ylim(0,max(10,max(v for v in rates if v==v)+3))
 for value,error,rate in zip(x,errors,rates):
  if error:other.annotate(str(error),(value,rate),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
 ax.grid(axis='y',alpha=.2);confusion[name]=dict(support=support,errors=errors,counts=matrix)
axes[-1].set_xticks([i/2 for i in range(-16,17,2)]);axes[-1].set_xlabel('True half-unit value; labels above red points are error counts')
fig.suptitle('R01 value tails: pooled descriptive counts over three distinct confirmation populations')
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/'r01-value-tails.png',dpi=180)
(root/'value-confusions.json').write_text(json.dumps(confusion,indent=2))
