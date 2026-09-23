#!/usr/bin/env python3
"""Reconstruct readout-repair gates and paired improvements from raw examples."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
FIELDS=('value','type','operation','argument0','argument1','provenance')
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory)
rows=[]
for file in sorted(d.glob('*-predictions.json.gz')):rows.extend(json.loads(gzip.decompress(file.read_bytes())))
gates=[];summary=[];pairs=[];index={}
for row in rows:
    matching={k:[p==t for p,t in zip(row['predictions'][k],row['targets'][k])] for k in FIELDS}
    nonvalue=[all(v) for v in zip(*(matching[k] for k in FIELDS[1:]))]
    error=[(p-t)/2 for p,t in zip(row['predictions']['value'],row['targets']['value'])]
    meta={k:row[k] for k in ('seed','split','arm','distractors','steps','intervention')}
    summary.append(dict(**meta,counts=row['counts'],nonvalue_joint=dict(correct=sum(nonvalue),total=len(nonvalue)),value_given_nonvalue=dict(correct=sum(a and b for a,b in zip(nonvalue,matching['value'])),total=sum(nonvalue)),mae=sum(abs(e) for e in error)/len(error),signed_error=sum(error)/len(error),within_half=sum(abs(e)<=.5 for e in error),confusion=[[x,y,z] for (x,y),z in sorted(Counter(zip(row['targets']['value'],row['predictions']['value'])).items())]))
    index[(row['seed'],row['split'],row['arm'],row['distractors'],row['steps'],row['intervention'])]=(row,matching)
for seed in (10,11,12):
 for arm in ('unchanged','ridge','ce_refit'):
    failures=[]
    for dist in (2,8):
        item=index.get((seed,'validation',arm,dist,16,'none'))
        if item is None:failures.append(dict(distractors=dist,reason='missing'));continue
        row,_=item
        if row['counts']['joint']['total']<512: failures.append(dict(distractors=dist,reason='tiny population'))
        for field in FIELDS:
            key='argument1_required' if field=='argument1' else field;c=row['counts'][key];threshold=.99 if field in ('type','operation') else .98
            if not c['total'] or c['correct']/c['total']<=threshold:failures.append(dict(distractors=dist,field=field,counts=c,threshold=threshold))
    gates.append(dict(seed=seed,arm=arm,passed=not failures,failures=failures))
for key,(row,match) in index.items():
 if key[2]!='unchanged':continue
 for arm in ('ridge','ce_refit'):
    other,after=index[(*key[:2],arm,*key[3:])]
    assert row['event_sha256']==other['event_sha256'] and row['targets']==other['targets']
    for field in FIELDS[1:]:assert row['predictions'][field]==other['predictions'][field]
    outcomes=dict(Counter(('correct' if before else 'wrong')+'_to_'+('correct' if end else 'wrong') for before,end in zip(match['value'],after['value'])))
    pairs.append(dict(seed=key[0],split=key[1],arm=arm,distractors=key[3],steps=key[4],intervention=key[5],outcomes=outcomes))
(d/'summary.json').write_text(json.dumps(dict(gates=gates,rows=summary,paired=pairs),separators=(',',':')))
