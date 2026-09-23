"""Render standalone proposal/calibration figures from immutable raw results."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('seed*-raw.json'))
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    derived = {}
    for path in paths:
        raw = json.loads(path.read_text())
        name = path.stem
        for split,style in [('iid_validation','-'),('ood_validation','--')]:
            axes[0].plot([r['step'] for r in raw['curves']], [r['validation'][split]['full'] for r in raw['curves']],style,label=f'{name} {split[:3]}')
        partition = 'test' if 'test' in raw['readiness'] else 'validation'
        ready = raw['readiness'][partition]
        bins = [b for b in ready['local']['reliability'] if b['count']]
        axes[1].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=name)
        rows = ready['records']
        scores = [r['calibrated'] for r in rows]
        truth = [r['correct'] for r in rows]
        ordered = sorted(zip(scores,truth),reverse=True)
        tp = 0; precision = []; recall = []
        for i,(_, y) in enumerate(ordered,1):
            tp += y
            precision.append(tp/i); recall.append(tp/max(1,sum(truth)))
        axes[2].plot(recall,precision,label=name)
        reversed_metrics = {}
        for split,controls in raw['evaluations'].items():
            r = controls['reverse']['raw']
            flags = []
            for op, pred_op, target, pred in zip(r['target_primitive'],r['primitive'],r['target_pointers'],r['pointers']):
                changed = [target[0],target[2],target[1]]
                flags.append(op == pred_op and pred[0] == changed[0] and pred[1] == changed[1] and (op==3 or pred[2] == changed[2]))
            reversed_metrics[split] = {'original_target_full':controls['reverse']['metrics']['full'],
                                       'changed_valid_target_full':sum(flags)/len(flags),'count':len(flags)}
        derived[name] = {'reverse_control':reversed_metrics,'readiness_partition':partition,
                         'frozen_threshold':raw['readiness']['threshold'],'frozen_local':ready['local'],
                         'frozen_global':ready['global'],'delta_accuracy':raw['delta']['heldout_insert_value_accuracy']}
    axes[0].set(xlabel='optimizer steps',ylabel='full ordered proposal accuracy',ylim=(0,1.02))
    axes[1].plot([0,1],[0,1],'k:',linewidth=1)
    axes[1].set(xlabel='mean calibrated confidence',ylabel='empirical correctness',xlim=(0,1),ylim=(0,1))
    axes[2].axhline(.99,color='k',linestyle=':')
    axes[2].set(xlabel='executable recall',ylabel='precision',xlim=(0,1),ylim=(0,1.02))
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory/'proposal-readiness.svg')
    fig.savefig(directory/'proposal-readiness.png',dpi=160)
    plt.close(fig)
    result = {'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
              'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'results':derived}
    (directory/'derived-report.json').write_text(json.dumps(result,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory')
    render(parser.parse_args().directory)
