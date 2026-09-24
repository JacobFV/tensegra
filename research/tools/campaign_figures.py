#!/usr/bin/env python3
"""Extract audited campaign counts (stdlib) and render with matplotlib; no model imports.

python3 research/tools/campaign_figures.py --root . --extract-only
python3 research/tools/campaign_figures.py --render-data <figures>/plotted-data.json.gz --output <figures>
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import time
from pathlib import Path


def composition_rows(summary, source):
    """Lossless count adapter; caller must verify a summary's audit/hash first."""
    rows=[]
    cfg=summary['config']; lineage=cfg['data']['train']['seed']//1000000
    common=dict(source=source,lineage=lineage,replicate=cfg['replicate'],seed=cfg['seed'])
    def emit(panel,metric,counts,**keys):
        assert 0 <= counts['correct'] <= counts['total']
        rows.append(dict(common,panel=panel,metric=metric,**counts,**keys))
    if summary.get('phase')=='hybrid':
        for cell in summary['cells']:
            keys={k:cell[k] for k in ('view','distractors','path','delay')}
            for metric in ('original','supplied','changed_original','changed_supplied','joint'):
                if metric in cell: emit('c04_hybrid',metric,cell[metric],**keys)
            emit('c04_hybrid','refused',dict(correct=cell['refused'],total=cell['original']['total']),**keys)
        for view in summary['causal']:
            for cell in view['cells']:
                keys=dict(view=view['view'],path=cell['path'],delay=cell['delay'],intervention=cell['kind'])
                for metric in ('original','supplied','changed_original','changed_supplied'):
                    emit('c04_causal',metric,cell[metric],**keys)
                emit('c04_causal','refused',dict(correct=cell['refused'],total=cell['original']['total']),**keys)
    else:
        for endpoint,results in summary['results'].items():
            for split in ('validation','validation_reversed'):
                for target in ('supplied',) if split.endswith('_reversed') else ('original',):
                    cell=results[split][target] if split.endswith('_reversed') else results[split]
                    for metric in ('answer','value'):
                        emit('c04_neural',metric,cell[metric],arm=summary['arm'],endpoint=endpoint,target=target,
                             step=summary['primary_endpoint_step'] if endpoint=='endpoint' else summary['selected_step'],
                             view='reversed' if split.endswith('_reversed') else 'clean')
    return rows


def extract(root, c04_audit=None):
    rows, inputs, status = [], {}, []
    def read(path):
        p = root / path
        inputs[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
            return json.load(f)
    base = 'research/results/campaign-01/'
    review = 'research/campaigns/extended-01/review/'
    def add(panel, source, **kw):
        rows.append(dict(panel=panel, source=source, **kw))
    for pattern in ('s01-*/manifest.json.gz', 's04-*/manifest.json.gz', 's11-*/manifest.json.gz'):
        for p in sorted((root / (base+'semantics')).glob(pattern)):
            source=str(p.relative_to(root)); m=read(source)
            if 'profile' in p.parent.name:
                status.append(dict(experiment=p.parent.name,status='mechanical profile; excluded from scientific curves'))
                continue
            for c in m.get('curves', []):
                for policy in ('raw','calibrated'):
                    k='dev_'+policy
                    if k in c:
                        add('semantic_learning',source,run=p.parent.name,policy=policy,split='development',
                            presentations=c.get('presentations',c['update']*m['config']['batch_size']),
                            correct=int(c[k]['exact']),total=c[k]['examples'])
    for seed in (701,702,703):
        source=review+f'S12-pair-{seed}-audit.json'
        if not (root/source).exists():
            status.append(dict(experiment='S12',seed=seed,status='pending audit'));continue
        m=read(source)
        assert m['paired_population_exact']
        for policy, counts in m['results'].items():
            for arm in ('constant','decay'):
                add('semantic_confirmation',source,seed=seed,policy=policy,arm=arm,split='confirmation',
                    correct=counts[arm],total=m['examples'],threshold=m['decay_competence_threshold'])
    for directory in ('s10-schema-decode-compact','s10-reuse-s11'):
        source=base+'semantics/'+directory+'/summary.json'
        for c in read(source)['summary']:
            if c['split']=='development':
                add('semantic_supplied',source,checkpoint=c['checkpoint'],policy=c['policy'],arm=c['variant'],
                    correct=int(c['exact']),total=c['examples'])
    for p in sorted((root/(base+'attention/a06')).glob('*/*/config.json')):
        source=str(p.relative_to(root)); config=read(source)
        for f in sorted(p.parent.glob('eval-*.json')):
            source=str(f.relative_to(root)); m=read(source); step=int(f.stem.split('-')[1])
            population='curve' if m['eval_seed']==config['curve_eval_seed'] else 'final'
            assert m['eval_seed'] in (config['curve_eval_seed'],config['eval_seed'])
            for c in m['rows']:
                n=c['examples']; exact=c['task']*n
                assert abs(exact-round(exact))<1e-6
                add('attention_'+population,source,seed=config['seed'],arm=config['mode'],
                    condition=c['condition'],presentations=step*config['batch'],correct=round(exact),total=n,
                    supplied_correct=round(c['agreement_supplied_task']*n))
    source=review+'R04-uncertainty.json'; m=read(source)
    for result in m['results']:
        for c in result['cells']:
            add('return_recovery',source,seed=result['seed'],split=result['split'],**c)
    source=review+'R05-confirmation-uncertainty.json'; m=read(source)
    for result in m['seeds']:
        for c in result['clean']:
            split,distractors=c['key'].split('/')
            add('return_use',source,seed=result['seed'],split=split,distractors=int(distractors),**c)
        for c in result['causal']:
            for arm in ('wrong','swap'):
                add('return_intervention',source,seed=result['seed'],delay=c['delay'],arm=arm,**c[arm])
    source=review+'C03-paired-audit.json'
    for c in read(source)['results']:
        if c['endpoint']=='endpoint':
            total=sum(c[k] for k in ('both_correct','static_only','roles_only','both_wrong'))
            for arm in ('static','roles'):
                add('composition',source,arm=arm,split=c['split'],metric=c['field'],correct=c[arm],total=total)
    source=review+'A08-main-audit.json'
    if (root/source).exists():
        for c in read(source)['cells']:
            seed,arm=c['model'].split('-')
            for target in ('original','supplied'):
                add('attention_corruption',source,seed=int(seed),arm=arm,target=target,
                    condition=c['condition'],correct=c[target+'_correct'],total=c['examples'])
    source=base+'attention/a09/results/config.json'
    if (root/source).exists():
        config=read(source)
        for f in sorted((root/(base+'attention/a09/results')).glob('eval-*.json')):
            source=str(f.relative_to(root)); m=read(source)
            population='curve' if m['eval_seed']==config['curve_eval_seed'] else 'final'
            for c in m['rows']:
                add('records_'+population,source,seed=config['seed'],arm='records',condition=c['condition'],
                    presentations=int(f.stem.split('-')[1])*config['batch'],
                    correct=round(c['task']*c['examples']),total=c['examples'])
    for experiment in ('a10','a11'):
        audit_source=review+experiment.upper()+'-development-audit.json'
        if not (root/audit_source).exists():
            status.append(dict(experiment=experiment.upper(),status='pending audited eval bindings'));continue
        audit=read(audit_source);bindings=audit.get('input_sha256',{})
        source=base+f'attention/{experiment}/results/config.json'
        if source not in bindings:
            status.append(dict(experiment=experiment.upper(),status='pending audited config binding'));continue
        config=read(source);assert inputs[source]==bindings[source]
        for f in sorted((root/(base+f'attention/{experiment}/results')).glob('eval-*.json')):
            source=str(f.relative_to(root))
            if source not in bindings: continue
            m=read(source);assert inputs[source]==bindings[source]
            assert m['eval_seed'] in (config['curve_eval_seed'],config['eval_seed'])
            population='curve' if m['eval_seed']==config['curve_eval_seed'] else 'final'
            for c in m['rows']:
                add('records_'+population,source,experiment=experiment.upper(),seed=config['seed'],arm='records',condition=c['condition'],
                    presentations=int(f.stem.split('-')[1])*config['batch'],correct=round(c['task']*c['examples']),total=c['examples'])
    for seed in (10,11,12):
        source=base+f'returns/r05-confirmation/{seed}/decision-margin-groups.json'
        for c in read(source):
            for margin,counts in c['signed_value_minus_threshold'].items():
                add('return_margin',source,seed=seed,arm=c['arm'],key=c['key'],delay=c['delay'],
                    signed_value_minus_threshold=float(margin),**counts)
    source=review+'A12-main-audit.json'
    if (root/source).exists():
        audit=read(source)
        assert audit['zero_updates'] and audit['profile_and_main_frozen_binding_verified']
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'A12 audit binding mismatch: {path}'
            inputs[path]=digest
        for cell in audit['cells']:
            for metric in ('task','path','suffix'):
                add('a12_counts',source,policy=cell['policy'],condition=cell['condition'],metric=metric,
                    correct=cell[metric+'_correct'],total=cell['examples'])
        for cell in audit['paired_policy_counts']:
            add('a12_paired',source,**cell)
        for cell in audit['weighted_diagnostics']:
            for step in cell['by_reverse_execution_step']:
                add('a12_diagnostic',source,policy=cell['policy'],condition=cell['condition'],**step)
    source=review+'A13-main-audit.json'
    if (root/source).exists():
        audit=read(source)
        assert audit['zero_updates'] and audit['queried_mean_full_path_equals_existing_metric']
        assert audit['all_nonoracle_allnode_routes_equal']
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'A13 audit binding mismatch: {path}'
            inputs[path]=digest
        for cell in audit['cells']:
            for metric in ('task','path','suffix'):
                add('a13_counts',source,policy=cell['policy'],condition=cell['condition'],metric=metric,
                    correct=cell[metric+'_correct'],total=cell['examples'])
        for cell in audit['paired_queried_route_analysis']:
            add('a13_paired_query',source,**cell)
            joint=cell['task_route_joint_counts'];numerator=joint['route_correct_task_correct']
            denominator=numerator+joint['route_correct_task_wrong']
            assert abs(numerator/denominator-cell['task_given_route'])<1e-12
            add('a13_task_given_route',source,policy=cell['policy'],condition=cell['condition'],
                correct=numerator,total=denominator)
            for head,counts in cell['head_paths'].items():
                add('a13_query_heads',source,policy=cell['policy'],condition=cell['condition'],head=head,**counts)
        status.append(dict(experiment='A13',status='frozen audited checkpoint; nonoracle routes exactly equal; oracle privileged; no seed replication'))
    if c04_audit:
        # An explicit final audit index binds each approved summary byte-for-byte.
        # Shape: {"input_sha256": {"repository/relative/path": "sha256"}}.
        bindings={}
        for audit_path in c04_audit:
            receipt=read(str(audit_path))
            for path,digest in receipt['input_sha256'].items():
                if path.startswith(base+'composition/c04-confirmation/') and path.endswith('/summary.json') and path.split('/')[-2] in ('hybrid','n1_static','n1_roles','n2_rekey'):
                    assert path not in bindings or bindings[path]==digest
                    bindings[path]=digest
        assert bindings, 'empty C04 audit summary bindings'
        for source,digest in sorted(bindings.items()):
            assert source.startswith(base+'composition/c04-confirmation/') and source.endswith('/summary.json')
            summary=read(source)
            assert inputs[source]==digest, f'Unaudited C04 summary bytes: {source}'
            rows.extend(composition_rows(summary,source))
        completed={r['lineage'] for r in rows if r['panel']=='c04_hybrid'}
        for lineage in (560,561,562):
            if lineage not in completed: status.append(dict(experiment='C04',lineage=lineage,status='pending audited summary'))
    else:
        status.append(dict(experiment='C04',status='pending explicit audited-summary hash index'))
    status.append(dict(experiment='S13',status='cache/protocol only; no model outcome plotted'))
    conditions=[]
    for row in rows:
        if row['panel']=='attention_final' and row['condition'] not in conditions:
            conditions.append(row['condition'])
    return dict(schema_version=1,inputs=inputs,rows=rows,status=status,attention_condition_index=conditions,
                uncertainty='Seed traces are not confidence intervals. No pooled seed/event intervals are computed. Return Wilson intervals retained in data only; repeated delays are not independent.')


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'topoformer-campaign-01','savefig.dpi':180})
    rows=data['rows']
    def select(panel, **kw):
        return [r for r in rows if r['panel']==panel and all(r.get(k)==v for k,v in kw.items())]
    def finish(fig,name,caption):
        fig.text(.02,.015,caption,fontsize=8,va='bottom')
        fig.tight_layout(rect=(0,.1,1,1))
        for ext in ('png','svg','pdf'):
            metadata={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
            fig.savefig(output/f'{name}.{ext}',metadata=metadata)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,policy in zip(axes[0],('raw','calibrated')):
        rr=select('semantic_learning',policy=policy)
        for run in sorted({r['run'] for r in rr}):
            ss=sorted([r for r in rr if r['run']==run],key=lambda r:r['presentations'])
            ax.plot([r['presentations']/1000 for r in ss],[r['correct'] for r in ss],'.-',label=run.split('-')[0].upper() + (' '+next((part for part in run.split('-') if part.startswith('n') and part[1:].isdigit()), '') if run.startswith('s01') else ''))
        ax.set(title=f'Development: {policy}',xlabel='Optimizer presentations (thousands)',ylabel='Complete graphs / 512');ax.legend(fontsize=7)
    ax=axes[1,0]
    for seed in (701,702,703):
        for policy,style in [('raw','--'),('calibrated','-')]:
            ss=select('semantic_confirmation',seed=seed,policy=policy)
            if ss:
                ss.sort(key=lambda r:r['arm']=='decay');ax.plot([0,1],[r['correct'] for r in ss],style,marker='o',label=f'{seed} {policy}')
                ax.axhline(ss[0]['threshold'],color='.7',lw=.7)
        if not select('semantic_confirmation',seed=seed):
            ax.text(.02,.9-(seed-701)*.1,f'{seed}: pending',transform=ax.transAxes)
    ax.set(title=f"Fresh confirmation: {len({r['seed'] for r in select('semantic_confirmation')})}/3 audited seeds",xticks=[0,1],xticklabels=['Constant','Decay'],ylabel='Complete graphs / 1024');ax.legend(fontsize=7)
    ax=axes[1,1];rr=select('semantic_supplied',policy='calibrated')
    checkpoints=sorted({r['checkpoint'] for r in rr})
    for arm in ('baseline','bookkeeping','schema','combined'):
        ss=[next(r for r in rr if r['checkpoint']==c and r['arm']==arm) for c in checkpoints]
        ax.plot(range(len(ss)),[r['correct'] for r in ss],'o-',label=arm)
    ax.set(title='S10 supplied decoding: development/calibrated',xticks=range(len(checkpoints)),xticklabels=checkpoints,ylabel='Complete graphs / 512');ax.legend(fontsize=7)
    finish(fig,'semantic','Learned acquisition and supplied decoding are separate. S04 / S11 share exposure at 196,608 presentations.\nConfirmation threshold = 103/1024; pending seeds are absent, never zero. No event uncertainty is plotted.')
    fig,axes=plt.subplots(1,3,figsize=(13,4.7));colors=dict(none='C0',soft='C1',hard='C2',context='C3')
    for ax,group in zip(axes[:2],(0,3)):
        for arm in colors:
            for seed in (601,602,603):
                ss=sorted([r for r in select('attention_curve',arm=arm,seed=seed) if r['condition']['data_group']==group],key=lambda r:r['presentations'])
                ax.plot([r['presentations'] for r in ss],[100*r['correct']/r['total'] for r in ss],color=colors[arm],alpha=.65,label=arm if seed==601 else None)
        ax.set(title=f'A06 curve population: condition {group}',xlabel='Optimizer presentations',ylabel='Task correct (%)',ylim=(-2,102));ax.legend(fontsize=7)
    ax=axes[2];rr=select('attention_final');conditions=[]
    for r in rr:
        if r['condition'] not in conditions: conditions.append(r['condition'])
    for ai,arm in enumerate(colors):
        for seed in (601,602,603):
            ss=select('attention_final',arm=arm,seed=seed)
            ax.scatter([conditions.index(r['condition'])+(ai-1.5)*.14 for r in ss], [100*r['correct']/r['total'] for r in ss],s=9,color=colors[arm],alpha=.65)
    ax.set(title='A06 final population: original target',xlabel='Condition index (see plotted data)',ylabel='Task correct (%)',ylim=(-2,102),xticks=range(0,len(conditions),2))
    finish(fig,'attention','Each trace/dot is one fitted seed. Final events are separate from curve events; endpoints are not joined.\nHard/soft receive topology; context receives explicit edge context. Supplied-target counts are retained in plotted data.')
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    ax=axes[0]
    for group in (0,1):
        ss=sorted([r for r in select('records_curve') if r['condition']['data_group']==group],key=lambda r:r['presentations'])
        ax.plot([r['presentations'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',label=f"Group {group} (curve)")
        ss=[r for r in select('records_final') if r['condition']==dict(nodes=32 if group==0 else 64,depth=4 if group==0 else 8,data_group=group)]
        ax.scatter([r['presentations'] for r in ss],[100*r['correct']/r['total'] for r in ss],marker='x',s=55,color=f'C{group}',label=f'Group {group} (final)')
    ax.set(title='Graph-record learning · development',xlabel='Optimizer presentations',ylabel='Task correct (%)',ylim=(-2,102));ax.legend(fontsize=7)
    for ax,group in zip(axes[1:],(0,1)):
        for arm in ('soft','hard','context'):
            for target,style in [('original','-'),('supplied','--')]:
                for seed in (601,602,603):
                    ss=sorted([r for r in select('attention_corruption',arm=arm,target=target,seed=seed) if r['condition']['data_group']==group],key=lambda r:r['condition']['fraction'])
                    ax.plot([r['condition']['fraction'] for r in ss],[100*r['correct']/r['total'] for r in ss],style,color=colors[arm],alpha=.55,label=f'{arm} / {target}' if seed==601 else None)
        ax.set(title=f'A08 corruption · condition {group}',xlabel='Changed-edge fraction',ylabel='Task correct (%)',ylim=(-2,102));ax.legend(fontsize=6)
    finish(fig,'attention-controls','A09+audited continuations: one seed; final crosses are unconnected to curve events. Group 0 = 32 nodes / depth 4; group 1 = 64 / 8.\nA08: three seed traces per arm; solid = original target, dashed = supplied target. Conditions: 64 nodes / depth 8 and 128 / 32.')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,panel,title in [(axes[0,0],'return_recovery','R04 scalar recovery'),(axes[0,1],'return_use','R05 downstream decision')]:
        arms=('unchanged','ce_16384') if panel=='return_recovery' else (None,)
        for ai,arm in enumerate(arms):
            for seed in (10,11,12):
                ss=select(panel,seed=seed,split='test',distractors=8)
                if arm: ss=[r for r in ss if r['arm']==arm]
                ss.sort(key=lambda r:r['delay']);ax.plot([r['delay'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',color=f'C{ai}',alpha=.65,label=(arm or 'decision') if seed==10 else None)
        ax.set(title=title+' · test / 8 distractors',xlabel='Delay (32 unseen in fitting)',ylabel='Correct (%)');ax.legend(fontsize=7)
    ax=axes[1,0]
    for ai,arm in enumerate(('wrong','swap')):
        for seed in (10,11,12):
            ss=select('return_intervention',arm=arm,seed=seed);ax.plot([r['delay'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',color=f'C{ai}',alpha=.65,label=arm if seed==10 else None)
    ax.set(title='R05 changed-fact intervention gates',xlabel='Delay',ylabel='Correct conditional on changed fact (%)');ax.legend(fontsize=7)
    ax=axes[1,1]
    for metric,style in [('answer','-'),('value','--')]:
        for ai,arm in enumerate(('static','roles')):
            ss=[next(r for r in select('composition',arm=arm,metric=metric) if r['split']==split) for split in ('fresh_validation','fresh_validation-reversed')]
            ax.plot([0,1],[100*r['correct']/r['total'] for r in ss],style,marker='o',color=f'C{ai}',label=f'{arm} {metric}')
    ax.set(title='C03 fixed endpoint · development',xticks=[0,1],xticklabels=['Clean','Roles reversed'],ylabel='Correct (%)');ax.legend(fontsize=7)
    finish(fig,'returns-composition','Restricted original mixture / finite domain. Seed traces are separate; repeated delays share events.\nC03 is one development initialization with supplied scheduling. C04 appears separately when audited; incomplete lineages are never imputed.')

    if select('a12_counts'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        policies=('unchanged','record_hard','destination_hard','both_hard')
        shapes=sorted({tuple(r['condition'][k] for k in ('nodes','depth','groups')) for r in select('a12_counts')})
        labels=[f'{n} nodes / depth {d}' for n,d,g in shapes]
        for ax,metric in zip(axes[0],('task','path')):
            for i,policy in enumerate(policies):
                ss=select('a12_counts',policy=policy,metric=metric)
                ss.sort(key=lambda r:r['condition']['nodes'])
                ax.plot(range(len(shapes)),[r['correct'] for r in ss],'o-' if policy=='unchanged' else 'o--',color=f'C{i}',label=policy)
            ax.set(title=f'A12 event-level exact {metric}',xticks=range(len(shapes)),xticklabels=labels,ylabel='Correct events / 512',ylim=(-8,525));ax.legend(fontsize=7)
        ax=axes[1,0]
        for i,policy in enumerate(policies):
            ss=select('a12_paired',policy=policy);ss.sort(key=lambda r:r['condition']['nodes'])
            ax.plot(range(len(shapes)),[r['fixed'] for r in ss],'o-',color=f'C{i}',label=f'{policy}: fixed')
            ax.plot(range(len(shapes)),[-r['rebroken'] for r in ss],'x:',color=f'C{i}',label=f'{policy}: -rebroken')
        ax.axhline(0,color='.6',lw=.7);ax.set(title='Paired task changes relative to unchanged',xticks=range(len(shapes)),xticklabels=labels,ylabel='Events fixed (+) / rebroken (-)');ax.legend(fontsize=6)
        ax=axes[1,1]
        for i,policy in enumerate(policies):
            ss=[r for r in select('a12_diagnostic',policy=policy) if r['condition']['nodes']==128]
            ss.sort(key=lambda r:r['reverse_execution_step'])
            ax.plot([r['reverse_execution_step'] for r in ss],[100*r['destination_argmax_correct'] for r in ss],'-' if policy=='unchanged' else '--',color=f'C{i}',label=policy)
        ax.set(title='128 / 32: destination-head argmax diagnostic',xlabel='Reverse execution step',ylabel='Correct across instrumented node/head entries (%)');ax.legend(fontsize=7)
        finish(fig,'attention-read-localization','A12: one frozen model, zero optimizer updates. Dashed policies are engineered hardening, not newly learned interfaces.\nPath counts use events; head diagnostics use instrumented node/head entries. Their gap does not identify query-head failures. Paired events, not independent policies.')
    if select('a13_counts'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        policies=('unchanged','shared_soft','shared_hard','oracle_common')
        names=('Learned unchanged','Common soft (engineered)','Common hard (engineered)','Common oracle (privileged)')
        labels=['32 nodes / depth 4','64 nodes / depth 8','128 nodes / depth 32']
        for ax,panel,metric,title in [(axes[0,0],'a13_counts','task','Task correctness'),(axes[0,1],'a13_counts','path','Queried mean full-route correctness'),(axes[1,0],'a13_task_given_route',None,'Task conditional on correct full query route')]:
            for i,policy in enumerate(policies):
                ss=select(panel,policy=policy)
                if metric: ss=[r for r in ss if r['metric']==metric]
                ss.sort(key=lambda r:r['condition']['nodes'])
                ax.plot(range(3),[100*r['correct']/r['total'] for r in ss],['o-','s--','^--','D:'][i],color=f'C{i}',label=names[i])
                if panel=='a13_task_given_route':
                    r=ss[-1];ax.annotate(f"{r['correct']}/{r['total']}",(2,100*r['correct']/r['total']),xytext=(5,(i-1.5)*10),textcoords='offset points',fontsize=7,color=f'C{i}')
            ax.set(title=title,xticks=range(3),xticklabels=labels,ylabel='Correct (%)',ylim=(-3,108));ax.legend(fontsize=6)
        ax=axes[1,1]
        for offset,head,name in [(-.18,'original_destination','Original destination'),(.18,'used_destination','Used destination')]:
            ss=[next(r for r in select('a13_query_heads',policy=policy,head=head) if r['condition']['nodes']==128) for policy in policies]
            ax.bar([i+offset for i in range(4)],[r['correct_head_paths'] for r in ss],width=.34,label=name)
            assert len({r['head_path_denominator'] for r in ss})==1
            denominator=ss[0]['head_path_denominator']
        ax.set(title='Deep query: individual-head full-route correctness',xticks=range(4),xticklabels=['Unchanged','Common soft','Common hard','Oracle'],ylabel=f'Correct queried head paths / {denominator}');ax.legend(fontsize=7)
        finish(fig,'attention-common-route','A13: one frozen checkpoint, paired events, zero updates; no seed replication. Nonoracle mean routes are exactly equal across policies.\nCommon hard is engineered; oracle uses privileged routes. Queried heads share event support (8 heads/event); full all-node suffix counts remain separate in the data.')
    if select('c04_hybrid'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        audited_lineages=sorted({r['lineage'] for r in select('c04_hybrid')})
        pending_lineages=[n for n in (560,561,562) if n not in audited_lineages]
        for ax,view in zip(axes[0],('clean','reversed')):
            for ai,path in enumerate(('workspace','supplied_copy')):
                for lineage in (560,561,562):
                    ss=select('c04_hybrid',view=view,path=path,lineage=lineage,metric='joint',distractors=8)
                    ss.sort(key=lambda r:-1 if r['delay'] is None else r['delay'])
                    ax.plot([0 if r['delay'] is None else r['delay'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-' if path=='workspace' else 's',color=f'C{ai}',alpha=.65,label=path if lineage==560 else None)
            for ai,arm in enumerate(('n1_static','n1_roles','n2_rekey'),2):
                for endpoint,marker in [('endpoint','x'),('selected','+')]:
                    for lineage in (560,561,562):
                        ss=select('c04_neural',view=view,arm=arm,endpoint=endpoint,lineage=lineage,metric='answer',target='supplied' if view=='reversed' else 'original')
                        ax.scatter([18+ai for r in ss],[100*r['correct']/r['total'] for r in ss],marker=marker,color=f'C{ai}',label=f'{arm} {endpoint}' if lineage==560 else None)
            ax.set(title=f'C04 {view}: joint hybrid / neural answer',xlabel='Workspace delay; neural markers at right',ylabel='Correct (%)');ax.legend(fontsize=6)
        ax=axes[1,0]
        for ai,intervention in enumerate(('wrong','swap')):
            for lineage in (560,561,562):
                ss=select('c04_causal',view='clean',path='workspace',lineage=lineage,metric='changed_supplied',intervention=intervention)
                ss=[r for r in ss if r['total']];ss.sort(key=lambda r:r['delay'])
                ax.plot([r['delay'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',color=f'C{ai}',alpha=.65,label=intervention if lineage==560 else None)
        ax.set(title='C04 changed-fact gates · clean view',xlabel='Workspace delay',ylabel='Supplied-target correct on changed facts (%)');ax.legend(fontsize=7)
        ax=axes[1,1]
        for ai,metric in enumerate(('original','joint','refused')):
            for lineage in (560,561,562):
                ss=select('c04_hybrid',view='reversed',path='workspace',lineage=lineage,metric=metric,distractors=8)
                ss.sort(key=lambda r:r['delay'])
                ax.plot([r['delay'] for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',color=f'C{ai}',alpha=.65,label=metric if lineage==560 else None)
        ax.set(title='C04 reversed: answer / joint / refusal',xlabel='Workspace delay',ylabel='Count / total (%)');ax.legend(fontsize=7)
        finish(fig,'composition-confirmation',f"Audited lineages: {audited_lineages}; pending: {pending_lineages}. Each trace is a lineage; shared views/delays are not independent.\nNeural x = fixed 4000 primary; + = selected secondary. Supplied-copy is a sole-return reference. Finite numeric train/test overlap; supplied scheduling.")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path)
    p.add_argument('--extract-only',action='store_true');p.add_argument('--render-data',type=Path)
    p.add_argument('--c04-audit',type=Path,action='append',help='Repeatable repository-relative input_sha256 audit receipt; never infer audit from presence alone')
    a=p.parse_args()
    output=a.output or a.root/'research/campaigns/extended-01/figures';output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if a.render_data:
        with (gzip.open(a.render_data,'rt') if a.render_data.suffix=='.gz' else a.render_data.open()) as f:
            data=json.load(f)
    else:
        data=extract(a.root.resolve(),a.c04_audit)
    # Stable bytes: sorted compact JSON, mtime=0, no embedded source filename.
    encoded=(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    compressed=gzip.compress(encoded,mtime=0)
    (output/'plotted-data.json.gz').write_bytes(compressed)
    receipt=dict(schema_version=data['schema_version'],table='plotted-data.json.gz',
                 table_sha256=hashlib.sha256(compressed).hexdigest(),
                 uncompressed_sha256=hashlib.sha256(encoded).hexdigest(),
                 rows=len(data['rows']),panel_rows=dict(sorted(Counter(r['panel'] for r in data['rows']).items())),
                 inputs=data['inputs'],status=data['status'],uncertainty=data['uncertainty'])
    (output/'data-manifest.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if not a.extract_only: render(data,output)
    print(json.dumps(dict(rows=len(data['rows']),inputs=len(data['inputs']),wall_seconds=time.perf_counter()-start,rendered=not a.extract_only)))
if __name__=='__main__': main()
