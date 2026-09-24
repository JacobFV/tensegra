"""R11 fixed-endpoint gate with complete matrix and paired reference checks."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from campaign_return_diversity_summary import counts,validate_main_contract


def summarize(rows):
    aliases={'reference':'unchanged','constant':'balanced_16384','decay':'balanced_65536'}
    translated=[dict(r,head=aliases[r['head']]) for r in rows]
    validate_main_contract(translated)
    cells=[];strata=[];by_cell={}
    for r in rows:
        meta={k:r[k]for k in ('head','split','target_delay','distractors')}
        cells.append(dict(meta,**counts(r)))
        by_cell.setdefault((r['split'],r['target_delay'],r['distractors']),{})[r['head']]=r
        if r['split'].endswith('_grid'):
            for typ,value in sorted(set(zip(r['targets']['type'],r['targets']['value']))):
                ids=[i for i,pair in enumerate(zip(r['targets']['type'],r['targets']['value']))if pair==(typ,value)]
                strata.append(dict(meta,type=typ,value_label=value,**counts(r,ids)))
    paired=[]
    for key,arms in by_cell.items():
        ref=arms['reference']
        for name in ('constant','decay'):
            r=arms[name]
            if r['targets']!=ref['targets'] or r['event_sha256']!=ref['event_sha256']:raise ValueError('Unpaired events')
            if any(r['predictions'][k]!=ref['predictions'][k]for k in ref['targets']if k!='value'):raise ValueError('Changed non-value output')
            table={f'{a}->{b}':0 for a in (0,1)for b in (0,1)}
            for i,y in enumerate(ref['targets']['value']):
                table[f'{int(ref["predictions"]["value"][i]==y)}->{int(r["predictions"]["value"][i]==y)}']+=1
            paired.append(dict(split=key[0],delay=key[1],distractors=key[2],arm=name,reference='reference',outcomes=table))
    gates={}
    for name in aliases:
        mixture=min(r['correct']['value']/r['total'] for r in cells if r['head']==name and r['split']=='calibration')
        grid=min(r['correct']['value']/r['total']for r in strata if r['head']==name and r['split']=='calibration_grid')
        train=next(r['correct']['value']for r in cells if r['head']==name and r['split']=='train' and r['target_delay']==16)
        gates[name]=endpoint_gate(mixture,grid,train)
    selection=select_candidate(gates)
    return dict(cells=cells,strata=strata,paired=paired,gates=gates,**selection)


def endpoint_gate(mixture,grid,train):
    return dict(mixture_minimum=mixture,grid_minimum=grid,training16_correct=train,
                advance=mixture>=.98 and grid>=122/128 and train>=65405)


def select_candidate(gates):
    eligible=[a for a in ('constant','decay')if gates[a]['advance']]
    selected=max(eligible,key=lambda a:(gates[a]['grid_minimum'],gates[a]['mixture_minimum'],a=='constant'))if eligible else None
    return dict(selected=selected,lower_lr_specific_gain=gates['decay']['grid_minimum']-gates['constant']['grid_minimum']>=4/128)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=Path(a.predictions).read_bytes();s=summarize(json.loads(gzip.decompress(raw)));s['predictions_sha256']=hashlib.sha256(raw).hexdigest()
    Path(a.output).write_text(json.dumps(s,indent=2)+'\n')
