"""Streaming, artifact-only Stage 5 analysis. Run this file without importing Torch."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

# Numerators and their actual populations. Undefined conditionals remain null.
RATE_COUNTS = {
    'binding_accuracy': ('binding_correct', 'steps'),
    'primitive_accuracy': ('primitive_correct', 'steps'),
    'all_binding_accuracy': ('all_binding_correct', 'examples'),
    'lowering_accuracy': ('all_lowering_correct', 'examples'),
    'type_validity_rate': ('valid', 'examples'),
    'execution_accuracy': ('execution_correct', 'defined_examples'),
    'result_accuracy': ('execution_correct', 'defined_examples'),
    'lowering_audit_complete_trajectory_accuracy': ('lowering_audit_complete_trajectory', 'examples'),
    'task_accuracy': ('task_correct', 'defined_examples'),
    'complete_trajectory_accuracy': ('complete_trajectory', 'defined_examples'),
    'task_given_binding_correct': ('task_given_binding_correct', 'all_binding_correct'),
    'execution_given_lowering_correct': ('execution_given_lowering_correct', 'all_lowering_correct'),
    'task_given_execution_correct': ('task_given_execution_correct', 'execution_correct'),
    'oracle_lifting_accuracy': ('oracle_lift_correct', 'defined_examples'),
}


def ratio(numerator, denominator):
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError('invalid count numerator/denominator')
    return numerator/denominator if denominator else None


def describe(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None,
                values=values)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def rates(cell):
    counts = cell['counts']
    out = {}
    for metric, (numerator, denominator) in RATE_COUNTS.items():
        if metric in cell and cell[metric] is None:
            out[metric] = None
            continue
        if numerator not in counts or counts[numerator] is None:
            continue
        denominator = 'examples' if denominator == 'defined_examples' and denominator not in counts else denominator
        if denominator not in counts or counts[denominator] is None: continue
        value = ratio(counts[numerator], counts[denominator])
        # Explicit nulls distinguish actual execution from counterfactual audits.
        if metric in cell and cell[metric] is None: value = None
        if metric in cell and cell[metric] is not None and value is not None and not math.isclose(cell[metric],value,abs_tol=1e-9):
            raise ValueError(f'inconsistent stored rate {metric}')
        out[metric] = value
    return out


def _aggregate(items):
    counts = defaultdict(int)
    for _,cell in items:
        for key,value in cell['counts'].items():
            if value is not None:
                if not isinstance(value,int) or value<0: raise ValueError('counts must be nonnegative integers')
                counts[key]+=value
    all_rates=[rates(cell) for _,cell in items]
    metrics={key:describe([r.get(key) for r in all_rates]) for key in set().union(*all_rates)}
    pooled_cell={'counts':dict(counts)}
    for key in metrics:
        if metrics[key]['n']==0: pooled_cell[key]=None
    sizes=[cell['runtime_nodes'] for _,cell in items if cell.get('runtime_nodes')]
    runtime_nodes=None if not sizes else dict(min=min(s['min'] for s in sizes),max=max(s['max'] for s in sizes),
                                             seed_means=describe([s['mean'] for s in sizes]))
    return dict(seeds=[seed for seed,_ in items],counts=dict(counts),metrics=metrics,pooled_rates=rates(pooled_cell),runtime_nodes=runtime_nodes)


def summarize(rows, config):
    """Consume one decoded run at a time; retain only compact cell statistics."""
    groups=defaultdict(list); curves=defaultdict(list); initial=defaultdict(list)
    confidence=defaultdict(list); identities={}; seen=set(); sources=set(); configs=set()
    provenance=[]; resources=[]; family_groups=defaultdict(list)
    seeds=set(config['seeds']); variants=set(config['variants'])
    required={f'n{n}_d{d}' for n in config['eval_sizes'] for d in config['eval_depths']}
    if config.get('extra_evaluations'):
        required.update(('renamed','paraphrase','noise_medium','noise_high','ambiguous','invalid','permuted','wrong_graph'))
    for row in rows:
        key=row['variant'],row['seed']
        if key in seen or key[0] not in variants or key[1] not in seeds: raise ValueError('duplicate/unexpected run')
        seen.add(key); sources.add(json.dumps(row['source'],sort_keys=True)); configs.add(row['config_hash'])
        training=row['training']; identities[key]=(training.get('schedule_hash'),training.get('initial_state_hash'))
        provenance.append(dict(variant=key[0],seed=key[1],source=row['source'],config_hash=row['config_hash'],
                               schedule_hash=training.get('schedule_hash'),initial_state_hash=training.get('initial_state_hash'),metric_semantics=row.get('metric_semantics')))
        resources.append(dict(variant=key[0],seed=key[1],**row.get('resources',{})))
        cells=row['evaluations']; names=[cell['condition'] for cell in cells]
        if len(names)!=len(set(names)) or not required.issubset(names): raise ValueError('missing/duplicate evaluation grid')
        for cell in cells:
            groups[(key[0],cell['condition'])].append((key[1],{k:v for k,v in cell.items() if k in ('settings','data_hash','counts','runtime_nodes',*RATE_COUNTS)}))
            for risk in cell.get('confidence',[]):
                confidence[(key[0],cell['condition'],risk['threshold'])].append((key[1],risk))
            for family,part in cell.get('family_breakdown',{}).items():
                counts = dict(part)
                counts['execution_correct'] = counts.pop('result_correct', 0)
                counts['all_lowering_correct'] = counts.pop('lowering_correct', 0)
                wrapped = {'counts': counts, 'execution_accuracy': None}
                family_groups[(key[0],cell['condition'],family)].append((key[1],wrapped))
        for item in training.get('curve',[]):
            curves[(key[0],item['step'])].append((key[1],item['diagnostics']))
        for cell in row.get('initial_evaluations',[]): initial[(key[0],cell['condition'])].append((key[1],cell))
    if seen!={(v,s) for v in variants for s in seeds}: raise ValueError('missing run coverage')
    if len(sources)!=1 or len(configs)!=1: raise ValueError('mixed source/config identity')
    expected_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    if configs != {expected_hash}: raise ValueError('configuration hash does not match supplied config')
    # Every evaluated condition must be paired across ALL declared variants/seeds.
    condition_names={condition for _,condition in groups}
    for condition in condition_names:
        for variant in variants:
            if {s for s,_ in groups[(variant,condition)]}!=seeds: raise ValueError('missing condition seed coverage')
        for seed in seeds:
            hashes={cell['data_hash'] for variant in variants for s,cell in groups[(variant,condition)] if s==seed}
            if len(hashes)!=1: raise ValueError('unpaired evaluation data')
    aggregates=[dict(variant=v,condition=c,settings=items[0][1]['settings'],**_aggregate(sorted(items)))
                for (v,c),items in sorted(groups.items())]
    contrasts=[]
    for variant in sorted(variants-{'supervised'}):
        if 'supervised' not in variants: break
        for condition in sorted(condition_names):
            left=dict(groups[(variant,condition)]);right=dict(groups[('supervised',condition)])
            diffs={metric:describe([rates(left[s]).get(metric)-rates(right[s]).get(metric)
                                  for s in sorted(seeds) if rates(left[s]).get(metric) is not None and rates(right[s]).get(metric) is not None])
                   for metric in RATE_COUNTS}
            contrasts.append(dict(variant=variant,reference='supervised',condition=condition,differences=diffs,
                                  matched_initialization=all(identities[(variant,s)][1]==identities[('supervised',s)][1] for s in seeds),
                                  matched_schedule=all(identities[(variant,s)][0]==identities[('supervised',s)][0] for s in seeds)))
    risks=[]
    for (variant,condition,threshold),items in sorted(confidence.items()):
        counts={k:sum(r.get(k,0) for _,r in items) for k in set().union(*(r.keys() for _,r in items))-{'threshold'}}
        answered=counts['answered'];correct=counts['correct']
        risks.append(dict(variant=variant,condition=condition,threshold=threshold,counts=counts,
                          coverage=ratio(answered,counts['examples']),risk=ratio(answered-correct,answered),
                          unconditional_execution_accuracy=ratio(correct,counts['examples']),
                          false_positive_crystallization=ratio(counts.get('accepted_wrong',0),counts.get('accepted',0)),
                          false_negative_crystallization=ratio(counts.get('deferred_correct',0),counts.get('correct_lowerings',0))))
    return dict(schema_version=5,provenance=provenance,aggregates=aggregates,confidence=risks,
                paired_contrasts=contrasts,resources=resources,
                curves=[dict(variant=v,step=s,warmup_boundary=config.get('warmup_steps'),**_aggregate(sorted(items))) for (v,s),items in sorted(curves.items())],
                initial=[dict(variant=v,condition=c,**_aggregate(sorted(items))) for (v,c),items in sorted(initial.items())],
                families=[dict(variant=v,condition=c,family=f,**_aggregate(sorted(items))) for (v,c,f),items in sorted(family_groups.items())],
                caveats=['D counts operations after initial resolve; total schedule length is D+1.',
                         'Grid size denotes distractor objects, not total runtime nodes.',
                         'Exact execution is supplied semantics, not learned symbolic reasoning.',
                         'Neural-control exact-lowering audits are counterfactual, not executed model trajectories.',
                         'Undefined conditional rates are null; abstention never counts as correct.'])


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for variant in sorted({r['variant'] for r in summary['aggregates']}):
        cells=[r for r in summary['aggregates'] if r['variant']==variant and r['condition'].startswith('n') and '_d' in r['condition']]
        if not cells: continue
        sizes=sorted({r['settings']['nodes'] for r in cells});depths=sorted({r['settings']['depth'] for r in cells})
        fig,axes=plt.subplots(1,4,figsize=(15,3.4),constrained_layout=True)
        for ax,metric in zip(axes,('task_accuracy','execution_accuracy','binding_accuracy','complete_trajectory_accuracy')):
            values={(r['settings']['nodes'],r['settings']['depth']):r['metrics'].get(metric,{}).get('mean') for r in cells}
            matrix=[[values.get((n,d)) if values.get((n,d)) is not None else float('nan') for n in sizes] for d in depths]
            im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            ax.set(xticks=range(len(sizes)),xticklabels=sizes,yticks=range(len(depths)),yticklabels=depths,
                   xlabel='Distractor objects',ylabel='Post-resolve depth D',title=metric.replace('_',' '))
            for i,row in enumerate(matrix):
                for j,v in enumerate(row):
                    if math.isfinite(v): ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<.5 else 'black',fontsize=8)
        fig.colorbar(im,ax=axes,shrink=.7);fig.suptitle(variant);fig.savefig(output/f'{variant}-matrix.png',dpi=150);plt.close(fig)
    for condition in ('noise_medium','noise_high','ambiguous','invalid'):
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for variant in sorted({r['variant'] for r in summary['confidence']}):
            points=[r for r in summary['confidence'] if r['variant']==variant and r['condition']==condition and r['risk'] is not None]
            if points: ax.plot([r['coverage'] for r in points],[r['risk'] for r in points],marker='o',label=variant)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Answered fraction',ylabel='Wrong result | answered',title=condition)
        if ax.lines: ax.legend(fontsize=7)
        fig.savefig(output/f'{condition}-risk-coverage.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,metric in zip(axes,('task_accuracy','lowering_accuracy')):
        for variant in sorted({r['variant'] for r in summary['curves']}):
            points=[r for r in summary['curves'] if r['variant']==variant and r['metrics'].get(metric,{}).get('mean') is not None]
            if points: ax.plot([r['step'] for r in points],[r['metrics'][metric]['mean'] for r in points],label=variant)
        ax.set(xlabel='Optimizer updates',ylabel=metric,ylim=(0,1))
        boundaries={r['warmup_boundary'] for r in summary['curves'] if r['warmup_boundary'] is not None}
        for boundary in boundaries: ax.axvline(boundary,color='gray',linestyle='--',alpha=.5)
    if axes[0].lines: axes[0].legend(fontsize=7)
    fig.savefig(output/'learning-curves.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics');parser.add_argument('output');parser.add_argument('--config');parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();path=Path(args.metrics);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config or path.parent/'config.json').read_text())
    summary=summarize(read_rows(path),config)
    summary['artifact']=dict(name=path.name,sha256=file_hash(path),bytes=path.stat().st_size)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not args.no_plots: plots(summary,out/'figures')


if __name__=='__main__': main()
