"""Render count-weighted retention plots and auditable failure examples."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(directory)
    rows=[]
    training=[]
    for path in root.glob('seed*-*.jsonl'):
        mode=path.stem.split('-',1)[1]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if 'step' in row: training.append(dict(row,mode=mode))
            if 'split' in row: rows.append(dict(row,mode=row.get('mode',mode),initialization=int(path.stem.split('-')[0][4:])))
    fields=('value','type','operation','argument0','argument1','provenance')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for field,axis in zip(fields,axes.flat):
        for mode in sorted({r['mode'] for r in training}):
            aggregate=defaultdict(list)
            for row in training:
                if row['mode']==mode: aggregate[row['step']].append(row['loss'][field])
            x=sorted(aggregate)
            axis.plot(x,[sum(aggregate[step])/len(aggregate[step]) for step in x],label=mode)
        axis.set_title(field);axis.set_xlabel('Optimizer updates');axis.set_ylabel('Cross-entropy');axis.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('C1 reconstruction losses, including step0; paired initializations averaged')
    fig.tight_layout();fig.savefig(root/'training-loss.png',dpi=140);fig.savefig(root/'training-loss.svg');plt.close(fig)
    splits=sorted({r['split'] for r in rows})
    failure=[]
    summary=[]
    identity_audit=[]
    for row in rows:
        if row['intervention']!='none': continue
        required=[i for i,op in enumerate(row['targets']['operation']) if op!=3]
        second=dict(correct=sum(row['targets']['argument1'][i]==row['predictions']['argument1'][i] for i in required),total=len(required))
        ordered=dict(correct=sum(all(row['targets'][name][i]==row['predictions'][name][i] for name in ('argument0','argument1')) for i in range(len(row['targets']['value']))),total=len(row['targets']['value']))
        identity_audit.append({k:row[k] for k in ('initialization','mode','split','seed','distractors','delay')}|dict(argument1_required=second,ordered_arguments=ordered))
    (root/'identity-denominators.json').write_text(json.dumps(identity_audit,indent=2))
    for split in splits:
        for distractors in sorted({r['distractors'] for r in rows}):
            fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
            for field,axis in zip(fields,axes.flat):
                for mode in sorted({r['mode'] for r in rows if r['intervention']=='none'}):
                    aggregate=defaultdict(lambda:[0,0])
                    for r in rows:
                        if r['split']==split and r['distractors']==distractors and r['intervention']=='none' and r['mode']==mode:
                            for i,k in enumerate(('correct','total')): aggregate[r['delay']][i]+=r['counts'][field][k]
                    x=sorted(aggregate)
                    axis.plot(x,[aggregate[d][0]/aggregate[d][1] for d in x],marker='o',label=mode)
                    for d in x: summary.append(dict(split=split,distractors=distractors,mode=mode,delay=d,field=field,correct=aggregate[d][0],total=aggregate[d][1]))
                axis.set_title(field); axis.set_ylim(0,1.02); axis.grid(alpha=.2); axis.set_xlabel('Retention updates'); axis.set_ylabel('Accuracy')
            axes[0,0].legend();fig.suptitle(f'{split}; {distractors} unrelated distractors per update; workspace reconstruction')
            fig.tight_layout();fig.savefig(root/f'retention-{split}-distractors{distractors}.png',dpi=140);fig.savefig(root/f'retention-{split}-distractors{distractors}.svg');plt.close(fig)
    for mode in sorted({r['mode'] for r in rows}):
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for field,axis in zip(('value','joint'),axes):
            for intervention in sorted({r['intervention'] for r in rows}):
                aggregate=defaultdict(lambda:[0,0])
                for r in rows:
                    if r['split']=='validation' and r['mode']==mode and r['intervention']==intervention:
                        for i,k in enumerate(('correct','total')):aggregate[r['delay']][i]+=r['counts'][field][k]
                x=sorted(aggregate)
                if x:axis.plot(x,[aggregate[d][0]/aggregate[d][1] for d in x],marker='o',label=intervention)
            axis.set_title(field);axis.set_xlabel('Retention updates');axis.set_ylim(0,1.02);axis.grid(alpha=.2)
        axes[0].set_ylabel('Accuracy against declared targets');axes[1].legend(fontsize=7,bbox_to_anchor=(1.02,1));fig.suptitle(f'{mode}: frozen interventions (only measured delays), validation cohorts pooled')
        fig.tight_layout();fig.savefig(root/f'interventions-{mode}.png',dpi=140);fig.savefig(root/f'interventions-{mode}.svg');plt.close(fig)
    for r in rows:
        if r['intervention']!='none' or r['delay']!=16:continue
        for i in range(len(r['targets']['value'])):
            truth={f:r['targets'][f][i] for f in fields};pred={f:r['predictions'][f][i] for f in fields}
            if truth!=pred:
                failure.append(dict(initialization=r['initialization'],mode=r['mode'],split=r['split'],seed=r['seed'],distractors=r['distractors'],delay=16,index=i,target=truth,prediction=pred,data_hash=r['data_hash']))
                break
    (root/'summary-counts.json').write_text(json.dumps(summary,indent=2))
    (root/'failure-examples.json').write_text(json.dumps(failure,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();report(args.directory)
