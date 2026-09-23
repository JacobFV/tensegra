"""Read exact archived score trajectories for new seed21's 300-update errors."""
import argparse,gzip,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
steps=[0,25,100,300,600,1200]
records={t:json.load(gzip.open(a.directory/f'scores-seed21-u{t}.json.gz','rt')) for t in steps}
failures=[]
for graph in records[300]['graphs']:
    for pair,(scores,truth) in enumerate(zip(graph['scores'],graph['truth'])):
        for relation,(score,target) in enumerate(zip(scores,truth)):
            threshold=records[300]['relations'][relation]['threshold']
            if (score>threshold)==target:continue
            trace=[]
            for step in steps:
                row=next(g for g in records[step]['graphs'] if g['graph_seed']==graph['graph_seed'])
                value=row['scores'][pair][relation];current=records[step]['relations'][relation]['threshold']
                assert row['truth'][pair][relation]==target
                trace.append(dict(update=step,logit=value,threshold=current,raw_correct=(value>0)==target,calibrated_correct=(value>current)==target,frozen_300_threshold_correct=(value>threshold)==target))
            failures.append(dict(seed=21,graph_seed=graph['graph_seed'],source=pair//graph['nodes'],target=pair%graph['nodes'],relation_index=relation,gold=target,trajectory=trace))
a.output.write_text(json.dumps(dict(scope='Newseed21 errors only, distinct from historicalseed10five; observational checkpoints, no selection',failures=failures),indent=2)+'\n')
