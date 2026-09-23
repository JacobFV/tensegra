"""Paired R05 consumption metrics; no model fitting or endpoint selection."""
import argparse,gzip,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
r=json.load(gzip.open(a.results/'predictions.json.gz','rt'))
def get(arm,key,delay):return next(x for x in r if(x['arm'],x['key'],x['delay'])==(arm,key,delay))
def acc(row):return row['original_correct']/row['total']
out={'clean':[],'causal':[],'interpretation':'Per-cell paired event support; no independence assumed across delays.'}
for row in r:
 if row['arm']!='learned' or not row['key'].startswith('validation/'):continue
 oracle=get('oracle',row['key'],row['delay']);prior=get('query_only',row['key'],row['delay']);gain=acc(oracle)-acc(prior)
 out['clean'].append({'key':row['key'],'delay':row['delay'],'learned_correct':row['original_correct'],'oracle_correct':oracle['original_correct'],'query_only_correct':prior['original_correct'],'total':row['total'],'normalized_gain':None if gain<=0 else (acc(row)-acc(prior))/gain})
for row in r:
 if row['arm']!='learned' or not row['key'].startswith('intervention_'):continue
 clean=get('learned','validation/8',row['delay']);n=row['total'];target=row['original_targets'];d=[float(x==t)-float(y==t) for x,y,t in zip(clean['predictions'][:n],row['predictions'],target)]
 mean=sum(d)/n;var=sum((x-mean)**2 for x in d)/(n-1);se=math.sqrt(var/n)
 out['causal'].append({'key':row['key'],'delay':row['delay'],'original_correct':row['original_correct'],'total':n,'paired_original_accuracy_drop':mean,'descriptive_normal95':[mean-1.96*se,mean+1.96*se],'changed_total':row.get('changed_total'),'changed_supplied_correct':row.get('changed_supplied_correct'),'changed_supplied_accuracy':None if not row.get('changed_total') else row['changed_supplied_correct']/row['changed_total']})
(a.results/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
