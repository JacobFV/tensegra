"""Reconstruct Stage 11 return metrics and paired outcomes from raw rows."""
import argparse
import gzip
import json
from pathlib import Path


def summarize(directory):
    rows=[]
    for path in sorted(directory.glob('*-predictions.json.gz')):
        rows.extend(json.load(gzip.open(path,'rt')))
    gates=[];paired=[];cells=[]
    for row in rows:
        if row['target_delay']==16:
            required={'type':.99,'operation':.99,'value':.98,'argument0':.98,'argument1_required':.98,'provenance':.98}
            failed=[field for field,threshold in required.items() if row['counts'][field]['total']==0 or
                    row['counts'][field]['correct']/row['counts'][field]['total']<=threshold]
            gates.append({key:row[key] for key in ('seed','head','split','distractors')}|dict(passed=not failed,failed_fields=failed))
        p,t=row['predictions'],row['targets']
        nonvalue=[all(p[field][i]==t[field][i] for field in t if field!='value') for i in range(len(t['value']))]
        nv=sum(nonvalue);both=sum(ok and p['value'][i]==t['value'][i] for i,ok in enumerate(nonvalue))
        cells.append({key:row[key] for key in ('seed','head','split','distractors','target_delay')}|
                     dict(counts=row['counts'],metrics=row['metrics'],value_given_nonvalue=dict(correct=both,total=nv)))
    lookup={(r['seed'],r['head'],r['split'],r['distractors'],r['target_delay']):r for r in rows}
    for key,left in lookup.items():
        seed,head,split,d,delay=key
        if head!='short_continue':continue
        right=lookup[(seed,'wide_continue',split,d,delay)]
        assert left['targets']==right['targets'] and left['event_sha256']==right['event_sha256']
        for subset in ('value','nonvalue','joint'):
            fields=['value'] if subset=='value' else [f for f in left['targets'] if subset=='joint' or f!='value']
            counts={k:0 for k in ('correct_correct','correct_wrong','wrong_correct','wrong_wrong')}
            for i in range(len(left['targets']['value'])):
                lc=all(left['predictions'][f][i]==left['targets'][f][i] for f in fields)
                rc=all(right['predictions'][f][i]==right['targets'][f][i] for f in fields)
                counts[('correct' if lc else 'wrong')+'_'+('correct' if rc else 'wrong')]+=1
            paired.append(dict(seed=seed,split=split,distractors=d,delay=delay,fields=subset,
                               orientation='short_to_wide',outcomes=counts))
    return dict(cells=cells,gates=gates,paired=paired)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();directory=Path(a.directory)
    result=summarize(directory);(directory/'summary.json').write_text(json.dumps(result,indent=2))
