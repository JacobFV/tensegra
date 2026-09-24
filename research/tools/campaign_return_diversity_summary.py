"""Independently derive R10 counts and the fixed development decision from raw arrays."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def counts(row, indices=None):
    pred, target = row['predictions'], row['targets']
    ids = range(len(target['value'])) if indices is None else indices
    correct = {k: sum(pred[k][i] == target[k][i] for i in ids) for k in target}
    correct['nonvalue_joint'] = sum(all(pred[k][i] == target[k][i] for k in target if k != 'value') for i in ids)
    correct['joint'] = sum(all(pred[k][i] == target[k][i] for k in target) for i in ids)
    return {'correct': correct, 'total': len(ids)}


def summarize(rows):
    cells, strata, paired = [], [], []
    by_cell = {}
    for row in rows:
        cell = {k: row[k] for k in ('head','split','target_delay','distractors')}
        cell.update(counts(row))
        if row['split']=='train' and row['head'].startswith('balanced_'):
            n=int(row['head'].split('_')[1]);cell['in_pool']=counts(row,range(n))
            if n<len(row['targets']['value']):cell['heldout_prefix_tail']=counts(row,range(n,len(row['targets']['value'])))
        cells.append(cell)
        key = (row['split'],row['target_delay'],row['distractors'])
        by_cell.setdefault(key,{})[row['head']] = row
        if row['split'].endswith('_grid'):
            labels = sorted(set(zip(row['targets']['type'],row['targets']['value'])))
            for typ,value in labels:
                ids = [i for i,pair in enumerate(zip(row['targets']['type'],row['targets']['value'])) if pair == (typ,value)]
                strata.append(dict(cell, type=typ, value_label=value, **counts(row,ids)))
    for key, arms in by_cell.items():
        names = sorted((h for h in arms if h.startswith('balanced_')),key=lambda h:int(h.split('_')[1]))
        if len(names) != 2: raise ValueError('Expected exactly two paired learned heads')
        a,b = [arms[n] for n in names]
        if a['targets'] != b['targets'] or a['event_sha256'] != b['event_sha256']:raise ValueError('Unpaired events')
        for field in ('value','nonvalue_joint','joint'):
            def outcome(r,i):
                fs = ['value'] if field=='value' else [k for k in r['targets'] if field=='joint' or k!='value']
                return all(r['predictions'][k][i]==r['targets'][k][i] for k in fs)
            table={f'{i}->{j}':0 for i in (0,1) for j in (0,1)}
            for i in range(len(a['targets']['value'])):table[f'{int(outcome(a,i))}->{int(outcome(b,i))}']+=1
            paired.append(dict(split=key[0],delay=key[1],distractors=key[2],field=field,arms=names,outcomes=table))
    names=sorted({r['head'] for r in rows if r['head'].startswith('balanced_')},key=lambda h:int(h.split('_')[1]))
    minimum={name:min(r['correct']['value']/r['total'] for r in strata if r['head']==name and r['split']=='calibration_grid') for name in names}
    mixture_pass=all(r['correct']['value']/r['total']>=.98 for r in cells if r['head']==names[1] and r['split']=='calibration')
    gate=dict(mixture_pass=mixture_pass,calibration_grid_minima=minimum,minimum_at_least_95pct=minimum[names[1]]>=.95,
              minimum_gain_at_least_4_of_128=minimum[names[1]]-minimum[names[0]]>=4/128)
    gate['advance']=all(gate[k] for k in ('mixture_pass','minimum_at_least_95pct','minimum_gain_at_least_4_of_128'))
    return dict(cells=cells,strata=strata,paired=paired,development_gate=gate)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=Path(a.predictions).read_bytes();result=summarize(json.loads(gzip.decompress(raw)))
    result['predictions_sha256']=hashlib.sha256(raw).hexdigest()
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
