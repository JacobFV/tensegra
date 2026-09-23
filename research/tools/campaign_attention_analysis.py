"""Paired summary of A03 compact records; inference is never rerun here."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
seeds=[201,202,203]
arms=['soft4','soft8','context','message','hard']
loaded={}
for arm in arms:
    for seed in seeds:
        path=args.root/f'{arm}-{seed}'
        summary=json.loads((path/'eval-00500.json').read_text())
        manifest=json.loads((path/'manifest.json').read_text())
        arrays=np.load(path/'eval-00500.npz')
        loaded[arm,seed]=(summary,manifest,arrays)
rng=np.random.default_rng(67131)
bootstrap=rng.integers(0,1024,size=(2000,1024))
rows=[]
for ci in range(18):
    row={'condition':loaded['soft8',201][0]['rows'][ci]['condition'],'arms':{}}
    for arm in arms:
        vals=[loaded[arm,s][2][f'c{ci}_task'].astype(float) for s in seeds]
        summaries=[loaded[arm,s][0]['rows'][ci] for s in seeds]
        row['arms'][arm]={'seed_task':[float(v.mean()) for v in vals],
            'mean_task':float(np.mean(vals)),
            'seed_pointer_path':[x['exact_pointer_path'] for x in summaries],
            'seed_forward_seconds':[x['forward_seconds'] for x in summaries],
            'seed_edge_mass':[x['edge_mass'] for x in summaries],
            'seed_supplied_edge_mass':[x['supplied_edge_mass'] for x in summaries]}
    row['paired_comparisons']={}
    for comparator in ['soft4','context','message','hard']:
        differences=np.stack([loaded['soft8',s][2][f'c{ci}_task'].astype(float)-loaded[comparator,s][2][f'c{ci}_task'].astype(float) for s in seeds])
        eventmean=differences.mean(0)
        samples=eventmean[bootstrap].mean(1)
        row['paired_comparisons']['soft8_minus_'+comparator]={'seed_differences':differences.mean(1).tolist(),
            'mean':float(eventmean.mean()),'event_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()}
    rows.append(row)
overrides=[]
for ci,base in [(18,2),(19,3)]:
    per=[]
    for seed in seeds:
        z=loaded['soft4',seed][2]
        before=z[f'c{base}_task'].astype(bool);after=z[f'c{ci}_task'].astype(bool)
        per.append({'seed':seed,'before':float(before.mean()),'after':float(after.mean()),
            'correct_to_correct':int((before&after).sum()),'correct_to_wrong':int((before&~after).sum()),
            'wrong_to_correct':int((~before&after).sum()),'wrong_to_wrong':int((~before&~after).sum())})
    overrides.append({'condition':loaded['soft4',201][0]['rows'][ci]['condition'],'seeds':per})
result={'seeds':seeds,'rows':rows,'frozen_override':overrides,
        'uncertainty_note':'Resample shared event indices, averaging paired differences across fixed initialization seeds. Seed variability is separately listed. Zero empirical bootstrap width does not establish exact population equivalence.','search':'A01 seed101 five arms; A02 seed102 three arms plus three frozen seed101 interventions; confirmation arms selected before fresh A03 evaluation.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
