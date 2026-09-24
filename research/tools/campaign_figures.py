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
    if 'components' in summary and 'neural_loads' in summary:
        for cell in summary['cells']:
            attempts={p['attempted'] for p in cell['passes']};assert len(attempts)==1
            rows.append(dict(common,panel='c04_timing',path=cell['path'],batch=cell['batch'],delay=cell['delay'],
                             median_seconds=cell['median_seconds'],attempted=next(iter(attempts)),passes=cell['passes']))
    elif summary.get('phase')=='hybrid':
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


def extract(root, c04_audit=None, c04_provisional=None):
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
    source=review+'S12-final-aggregate-audit.json'
    if (root/source).exists():
        audit=read(source)
        assert audit['status']=='independently_audited' and audit['all_six_actual_targets_and_order_exact']
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S12 aggregate binding mismatch: {path}'
            inputs[path]=digest
        for field,result in audit['results'].items():
            policy=field.removesuffix('_metrics')
            for pair in result['pairs']:
                for arm in ('constant','decay'):
                    match=[r for r in rows if r['panel']=='semantic_confirmation' and r['policy']==policy and r['seed']==pair['seed'] and r['arm']==arm]
                    assert len(match)==1 and match[0]['correct']==pair[arm] and match[0]['total']==pair['support']
            add('semantic_confirmation_uncertainty',source,policy=policy,**result)
        status.append(dict(experiment='S12',status='three paired seeds audited; directional advantage passes; all-seed competence fails'))
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
    source=review+'A14-final-aggregate-audit.json'
    if (root/source).exists():
        audit=read(source)
        assert audit['all_three_previously_audited_inputs_rebound'] and audit['primary_actual_joint_tables_verified']
        summary_path=base+'attention/a14/confirmation-analysis.json'
        assert summary_path in audit['input_sha256'], 'A14 requires explicit aggregate hash binding'
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'A14 audit binding mismatch: {path}'
            inputs[path]=digest
        summary=read(summary_path)
        assert summary['public_hashes_match'] and summary['record_order_hashes_match']
        assert summary['joint_primary']==audit['primary']
        for seed,bindings in summary['input_sha256'].items():
            for relative,digest in bindings.items():
                path=base+f'attention/a14/{seed}/results/'+relative
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'A14 seed binding mismatch: {path}'
                inputs[path]=digest
        for condition in summary['conditions']:
            for seed,policies in condition['seed_rows'].items():
                for policy,counts in policies.items():
                    add('a14_confirmation',summary_path,seed=int(seed),shape=condition['shape'],policy=policy,threshold=condition['threshold'],**counts)
        add('a14_joint_primary',summary_path,**summary['joint_primary'])
        # Aggregate audit separately verifies every reference cell and checkpoint binding.
        if audit.get('reference_cells_verified')==27 and audit.get('reference_checkpoint_bindings_to_A06'):
            reference_audit=audit
            for path,digest in reference_audit['input_sha256'].items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'A14 reference binding mismatch: {path}'
                inputs[path]=digest
            assert len(summary['engineering_references'])==9
            for model,result in summary['engineering_references'].items():
                for cell in result['rows']:
                    add('a14_reference',summary_path,model=model,condition=cell['condition'],total=cell['examples'],
                        task_correct=round(cell['task']*cell['examples']),route_correct=round(cell['exact_pointer_path']*cell['examples']),
                        suffix_correct=round(cell['suffix_value_trajectory']*cell['examples']))
        else:
            status.append(dict(experiment='A14 references',status='pending independent reference audit'))
    for experiment in ('R10','R11'):
        audit_path=review+experiment+'-development-summary-audit.json'
        if not (root/audit_path).exists(): continue
        audit=read(audit_path)
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'{experiment} binding mismatch: {path}'
            inputs[path]=digest
        source=base+f'returns/{experiment.lower()}-development/summary.json';summary=read(source)
        for group in ('cells','strata','paired'):
            for cell in summary[group]: add('return_tail_'+group,source,experiment=experiment,**cell)
        add('return_tail_gate',audit_path,experiment=experiment,gate=audit.get('development_gate',audit.get('gates')),selected=audit.get('selected'))
    audit_path=review+'S13-main-pair-audit.json'
    if (root/audit_path).exists():
        for name in ('S13-main-pair-audit.json','S13-english-main-audit.json','S13-mixed-main-audit.json'):
            receipt=read(review+name)
            for path,digest in receipt['input_sha256'].items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S13 binding mismatch: {path}'
                inputs[path]=digest
        source=base+'semantics/s13-paired-analysis.json';summary=read(source)
        for arm,result in summary['results'].items():
            for curve in result['curves']:
                for surface,policies in curve['surfaces'].items():
                    for policy,counts in policies.items():
                        add('s13_learning',source,arm=arm,surface=surface,policy=policy,added_update=curve['added_update'],
                            total=sum(v['examples'] for v in counts['by_node_count'].values()),**counts)
        add('s13_paired',source,paired_endpoint=summary['paired_endpoint'],promotion_criteria=summary['promotion_criteria'],promotion_pass=summary['promotion_pass'])
    audit_path=review+'S15-main-pair-audit.json'
    if (root/audit_path).exists():
        for name in ('S15-main-pair-audit.json','S15-control-main-audit.json','S15-mixed-main-audit.json'):
            receipt=read(review+name)
            for path,digest in receipt['input_sha256'].items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S15 binding mismatch: {path}'
                inputs[path]=digest
        pair_audit=read(audit_path)
        assert pair_audit['all_curve_components_verified'] and pair_audit['independent_event_multiplicity_intervals_exact']
        source=base+'semantics/s15-paired-analysis.json';summary=read(source)
        assert summary['paired']==pair_audit['paired'] and summary['criteria']==pair_audit['criteria']
        for arm,result in summary['results'].items():
            for curve in result['curves']:
                for shape,policies in curve['cells'].items():
                    for policy,counts in policies.items():
                        add('s15_learning',source,arm=arm,shape=shape,policy=policy,added_update=curve['added_update'],**counts)
        add('s15_paired',source,paired=summary['paired'],criteria=summary['criteria'],confirmation_eligible=summary['confirmation_eligible'],one_paired_extension_eligible=summary['one_paired_extension_eligible'])
    audit_path=review+'S16-main-audit.json'
    if (root/audit_path).exists():
        audit=read(audit_path)
        assert audit['all_paired_components_repairs_regressions_verified']
        for path,digest in audit['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S16 binding mismatch: {path}'
            inputs[path]=digest
        source=base+'semantics/s16-main/paired-summary.json';summary=read(source)
        for endpoint in summary['endpoints']:
            for policy,counts in endpoint['policies'].items():
                add('s16_normalization',source,seed=endpoint['seed'],arm=endpoint['arm'],policy=policy,**counts)
    audit_path=review+'S17-main-pair-audit.json'
    if (root/audit_path).exists():
        for name in ('S17-main-audit.json','S17-main-pair-audit.json'):
            audit=read(review+name)
            for path,digest in audit['input_sha256'].items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S17 binding mismatch: {path}'
                inputs[path]=digest
        source=base+'semantics/s17-paired-analysis.json';summary=read(source)
        for arm,cells in summary['results'].items():
            for shape,counts in cells.items():
                add('s17_calibration',source,arm=arm,shape=shape,**counts)
    audit_path=review+'S18-main-pair-audit.json'
    if (root/audit_path).exists():
        for name in ('S18-main-pair-audit.json','S18-main-archive-audit.json','S18-main-context-raw-audit.json','S18-main-control-raw-audit.json','S18-reference-v2-audit.json','S18-train-ranking-audit.json','S18-main-train-panel-audit.json','S18-main-components-audit.json'):
            receipt=read(review+name)
            for path,digest in receipt.get('input_sha256',{}).items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S18 binding mismatch: {path}'
                inputs[path]=digest
        source=base+'semantics/s18-paired-analysis.json';summary=read(source)
        for artifact in summary['artifact_inventory'].values():
            for kind in ('evaluation','calibration'):
                path=artifact[kind+'_path'];path=path[path.index('research/'):]
                digest=artifact[kind+'_sha256']
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S18 inventory mismatch: {path}'
                inputs[path]=digest
        for arm,updates in summary['results'].items():
            for update,policies in updates.items():
                for policy,cells in policies.items():
                    for shape,counts in cells.items():
                        add('s18_learning',source,arm=arm,added_update=int(update),policy=policy,shape=shape,**counts)
        for arm,updates in summary['calibration_overlap_train_panels'].items():
            for update,calibrations in updates.items():
                for calibration,panel in calibrations.items():
                    for shape,policies in panel['cells'].items():
                        for policy,counts in policies.items():
                            add('s18_train',source,arm=arm,added_update=int(update),calibration=calibration,policy=policy,shape=shape,**counts)
        add('s18_paired',source,paired=summary['paired'],decisions=summary['decisions'],endpoint_delta_percentage_points_ci95=summary['endpoint_delta_percentage_points_ci95'],baseline_inherited_seconds=summary['baseline_inherited_seconds'])
        source=review+'S18-train-ranking-audit.json';ranking=read(source)
        add('s18_rank',source,trained3x3=ranking['trained3x3'],scope=ranking['scope'])
        source=review+'S18-main-archive-audit.json';cost=read(source)
        add('s18_cost',source,total_new_S18_occupancy_seconds=cost['total_new_S18_occupancy_seconds'],historical_reused_training_seconds=cost['historical_reused_training_seconds'],scope=cost['scope'])
    source=base+'semantics/s19-analysis.json'
    s19_bindings={}
    s19_report_updates={}
    for update_path in sorted((root/review).glob('S19-*.json')):
        update=json.loads(update_path.read_text())
        if update.get('supersedes_receipt')==review+'S19-final-analysis-audit.json':
            prior=read(update['supersedes_receipt'])
            read(str(update_path.relative_to(root)))
            for path,old_digest in update['supersedes_input_sha256'].items():
                assert path=='research/campaigns/extended-01/semantics/S19-report.md', 'Only explicit S19 report wording supersession is allowed'
                assert prior['input_sha256'][path]==old_digest
                new_digest=update['input_sha256'][path]
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==new_digest
                s19_report_updates[path]=(old_digest,new_digest)
    for receipt_path in sorted((root/review).glob('S19-*-audit.json')):
        receipt=json.loads(receipt_path.read_text())
        if source in receipt.get('input_sha256',{}):
            read(str(receipt_path.relative_to(root)))
            for path,digest in receipt['input_sha256'].items():
                if path in s19_report_updates:
                    old_digest,new_digest=s19_report_updates[path]
                    assert digest==old_digest
                    digest=new_digest
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S19 binding mismatch: {path}'
                inputs[path]=digest
                s19_bindings[path]=digest
    if source in s19_bindings:
        summary=read(source)
        for artifact in summary['artifact_inventory'].values():
            path=artifact['path'];path=path[path.index('research/'):]
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==artifact['sha256'], f'S19 inventory mismatch: {path}'
            inputs[path]=artifact['sha256']
        for update,splits in summary['curves'].items():
            for split,counts in splits.items():
                add('s19_learning',source,added_update=int(update),split=split,**counts)
        add('s19_summary',source,scope=summary['scope'],paired=summary['endpoint_paired'],decisions=summary['decisions'],costs=summary['costs'])
    else:
        status.append(dict(experiment='S19',status='Explicit independent analysis hash binding pending; omitted'))
    # S20 is final-only: never inspect arm archives or an unbound interim aggregate.
    audit_path=review+'S20-final-confirmation-audit.json'
    source=base+'semantics/s20-analysis.json'
    if (root/audit_path).exists():
        receipt=read(audit_path)
        assert source in receipt.get('input_sha256',{}), 'S20 requires final independent aggregate hash binding'
        for path,digest in receipt['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S20 binding mismatch: {path}'
            inputs[path]=digest
        summary=read(source)
        assert set(summary['counts'])=={'701','702','703'}, 'S20 requires every registered seed'
        assert set(summary['paired'])=={'701','702','703'}, 'S20 requires complete paired matrix'
        for seed,comparisons in summary['paired'].items():
            assert set(comparisons)=={'original/raw','original/matched','context/raw','context/matched'}
            for comparison,paired in comparisons.items():
                assert set(paired['cells'])=={'3x3','3x4','4x3','4x4'}
                for shape,cell in paired['cells'].items():
                    assert cell['examples']==512
                    assert cell['record_complete']==summary['counts'][seed][shape]
                add('s20_paired',source,seed=int(seed),comparison=comparison,**paired)
            add('s20_endpoint',source,seed=int(seed),counts=summary['counts'][seed])
        add('s20_summary',source,scope=summary['scope'],decisions=summary['decisions'],bootstrap=summary['bootstrap'],mean_ci95=summary['three_fixed_lineage_mean_conditional_ci95_percentage_points'],arms=summary['arms'],process_seconds=summary['process_seconds'])
    else:
        status.append(dict(experiment='S20',status='Final closed independently audited aggregate pending; no interim outcomes read'))
    audit_path=review+'S21-final-development-audit.json'
    source=base+'semantics/s21-analysis.json'
    if (root/audit_path).exists():
        receipt=read(audit_path)
        assert source in receipt.get('input_sha256',{}), 'S21 requires final independent aggregate binding'
        for path,digest in receipt['input_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest, f'S21 binding mismatch: {path}'
            inputs[path]=digest
        summary=read(source)
        assert set(summary['curves'])=={'original','broad'}
        for arm,updates in summary['curves'].items():
            assert set(updates)=={'0','1024','2048','4096'}
            for update,splits in updates.items():
                for split,counts in splits.items():
                    add('s21_learning',source,arm=arm,added_update=int(update),split=split,**counts)
        add('s21_summary',source,scope=summary['scope'],paired=summary['paired'],ci95=summary['endpoint_delta_pp_conditional_ci95'],endpoint_counts=summary['endpoint_counts'],decisions=summary['decisions'],exposures=summary['exposures'],training_seconds=summary['training_seconds'],process_seconds=summary['process_seconds'],training_loss_curves=summary['training_loss_curves'])
    else:
        status.append(dict(experiment='S21',status='Final independent aggregate audit pending; omitted'))
    if c04_audit:
        # An explicit final audit index binds each approved summary byte-for-byte.
        # Shape: {"input_sha256": {"repository/relative/path": "sha256"}}.
        bindings={}
        for audit_path in c04_audit:
            receipt=read(str(audit_path))
            for path,digest in receipt['input_sha256'].items():
                if path.startswith(base+'composition/c04-confirmation/') and path.endswith('/summary.json') and path.split('/')[-2] in ('hybrid','n1_static','n1_roles','n2_rekey','timing'):
                    assert path not in bindings or bindings[path]==digest
                    bindings[path]=digest
        assert bindings, 'empty C04 audit summary bindings'
        for source,digest in sorted(bindings.items()):
            assert source.startswith(base+'composition/c04-confirmation/') and source.endswith('/summary.json')
            summary=read(source)
            assert inputs[source]==digest, f'Unaudited C04 summary bytes: {source}'
            rows.extend(dict(row,audit_status='independently_audited') for row in composition_rows(summary,source))
        completed={r['lineage'] for r in rows if r['panel']=='c04_hybrid'}
        for lineage in (560,561,562):
            if lineage not in completed: status.append(dict(experiment='C04',lineage=lineage,status='pending audited summary'))
    else:
        status.append(dict(experiment='C04',status='pending explicit audited-summary hash index'))
    for replicate in c04_provisional or []:
        lineage=560+replicate
        assert not any(r.get('lineage')==lineage and r['panel'].startswith('c04_') for r in rows), 'duplicate audited/provisional lineage'
        for arm in ('hybrid','n1_static','n1_roles','n2_rekey','timing'):
            source=base+f'composition/c04-confirmation/{replicate}/{arm}/summary.json'
            rows.extend(dict(row,audit_status='independent_final_audit_pending') for row in composition_rows(read(source),source))
        status[:]=[entry for entry in status if not (entry.get('experiment')=='C04' and entry.get('lineage')==lineage)]
        status.append(dict(experiment='C04',lineage=lineage,status='completed worker results; independent final audit pending'))
    status.append(dict(experiment='S13',status='one audited development pair; all criteria fail' if any(r['panel']=='s13_learning' for r in rows) else 'cache/protocol only; no model outcome plotted'))
    conditions=[]
    for row in rows:
        if row['panel']=='attention_final' and row['condition'] not in conditions:
            conditions.append(row['condition'])
    return dict(schema_version=1,inputs=inputs,rows=rows,status=status,attention_condition_index=conditions,
                uncertainty='Seed traces are not confidence intervals. S12 audited shared-event bootstrap intervals condition on fixed fitted seeds and are not seed-population intervals. Return Wilson intervals retained in data only; repeated delays are not independent.')


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
    if select('semantic_confirmation_uncertainty'):
        fig,axes=plt.subplots(2,2,figsize=(11,7.5))
        for col,policy in enumerate(('raw','calibrated')):
            row=select('semantic_confirmation_uncertainty',policy=policy)[0]
            ax=axes[0,col]
            for i,pair in enumerate(row['pairs']):
                ax.plot([0,1],[pair['constant'],pair['decay']],'o-',color=f'C{i}',label=str(pair['seed']))
                ax.annotate(str(pair['decay']),(1,pair['decay']),xytext=(5,0),textcoords='offset points',fontsize=8,color=f'C{i}')
            threshold=select('semantic_confirmation',policy=policy)[0]['threshold']
            if policy=='calibrated': ax.axhline(threshold,color='.5',ls=':',label=f'Competence = {threshold}/1024')
            ax.set(title=f'S12 {policy}: paired complete graphs',xticks=[0,1],xticklabels=['Constant','Decay'],ylabel='Correct / 1024');ax.legend(fontsize=7)
            ax=axes[1,col];interval=row['interval']
            for i,(pair,mean,ci) in enumerate(zip(row['pairs'],interval['per_seed_mean'],interval['per_seed_interval'])):
                ax.errorbar(100*mean,i,xerr=[[100*(mean-ci[0])],[100*(ci[1]-mean)]],fmt='o',color=f'C{i}',capsize=3)
            mean=interval['mean_paired_gain'];ci=interval['shared_event_mean_interval']
            ax.errorbar(100*mean,3,xerr=[[100*(mean-ci[0])],[100*(ci[1]-mean)]],fmt='D',color='black',capsize=4)
            ax.set(title=f'{policy}: decay − constant gain',yticks=[0,1,2,3],yticklabels=[str(p['seed']) for p in row['pairs']]+['Fixed-seed mean'],xlabel='Paired gain (percentage points)');ax.invert_yaxis();ax.axvline(0,color='.8',lw=.7)
        finish(fig,'semantic-confirmation','Directional advantage passes across all three pairs; all-seed calibrated competence fails (seed 701 below 103/1024).\n95% intervals jointly resample shared events, conditional on these fixed fitted seeds; they are not seed-population intervals. Raw is secondary; calibrated is primary.')
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
    if select('a14_confirmation'):
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        policies=('unchanged','both_hard','shared_soft','shared_hard','oracle_common')
        names=('Unchanged','Both hard (engineered)','Common soft (engineered)','Common hard (engineered)','Oracle (privileged)')
        shapes=list(dict.fromkeys(r['shape'] for r in select('a14_confirmation')))
        for ax,metric,title in zip(axes[0],('task_correct','route_correct','complete_all_node_suffix'),('Joint task answer','Complete queried route','Complete all-node suffix')):
            for i,policy in enumerate(policies):
                for seed in (1401,1402,1403):
                    ss=select('a14_confirmation',policy=policy,seed=seed);ss.sort(key=lambda r:shapes.index(r['shape']))
                    ax.plot(range(len(shapes)),[100*r[metric]/r['events'] for r in ss],'.-' if policy=='unchanged' else '.--',color=f'C{i}',alpha=.65,label=names[i] if seed==1401 else None)
            ax.set(title=title,xticks=range(len(shapes)),xticklabels=shapes,ylabel='Correct events (%)',ylim=(-3,104));ax.legend(fontsize=6)
        joint=select('a14_joint_primary')[0];ax=axes[1,0]
        for i,value in enumerate(joint['per_seed_gain']):ax.scatter(100*value,i,color=f'C{i}',s=25)
        mean=joint['mean_gain'];ci=joint['gain_percentile95']
        ax.errorbar(100*mean,3,xerr=[[100*(mean-ci[0])],[100*(ci[1]-mean)]],fmt='D',color='black',capsize=4)
        ax.set(title='Joint: common hard − unchanged',yticks=[0,1,2,3],yticklabels=['1401','1402','1403','Fixed-seed mean'],xlabel='Paired task gain (percentage points)');ax.invert_yaxis()
        ax=axes[1,1]
        for i,policy in enumerate(policies):
            ss=select('a14_confirmation',policy=policy,shape='joint')
            ax.scatter([i+(r['seed']-1402)*.1 for r in ss],[100*r['task_given_correct_queried_route'] for r in ss],color=f'C{i}',s=25)
        ax.set(title='Joint: task given correct queried route',xticks=range(5),xticklabels=['Original','Both hard','Soft','Hard','Oracle'],ylabel='Conditional task correctness (%)',ylim=(-3,104))
        ax=axes[1,2]
        references=select('a14_reference')
        for i,arm in enumerate(('soft','hard','context')):
            ss=[r for r in references if r['model'].endswith('-'+arm)]
            ax.scatter([i+(int(r['model'].split('-')[0])-602)*.1 for r in ss],[100*r['task_correct']/r['total'] for r in ss],s=25,label=arm)
        ax.set(title='Engineering references · all three shapes',xticks=range(3),xticklabels=['Soft topology','Hard topology','Graph context'],ylabel='Task correct (%)',ylim=(95,101));ax.tick_params(axis='x',labelrotation=12)
        if not references: ax.text(.05,.5,'Independent reference audit pending',transform=ax.transAxes)
        finish(fig,'attention-confirmation','A14: three fitted seeds; engineered common hard is distinct from learned unchanged and privileged oracle. Top-row metrics have different requirements.\nCI jointly resamples shared events conditional on fixed seeds. References use different interfaces/training; no parameter/FLOP match or unique attention win is claimed.')
    if select('return_tail_strata'):
        fig,axes=plt.subplots(2,2,figsize=(12,7.5))
        for ax,experiment,arms in [(axes[0,0],'R10',('unchanged','balanced_16384','balanced_65536')),(axes[0,1],'R11',('reference','constant','decay'))]:
            for i,arm in enumerate(arms):
                ss=[select('return_tail_strata',experiment=experiment,head=arm,split=split) for split in ('calibration_grid','validation_grid')]
                minima=[min(r['correct']['value'] for r in group) for group in ss]
                ax.plot([0,1],minima,'o-',label=arm)
                for j,value in enumerate(minima):ax.annotate(str(value),(j,value),xytext=(4,(i-1)*8),textcoords='offset points',fontsize=7)
            ax.axhline(122,color='.5',ls=':',label='Frozen 95% gate: 122/128')
            ax.set(title=f'{experiment}: worst type/value/delay stratum',xticks=[0,1],xticklabels=['Calibration (gate)','Validation (reused)'],ylabel='Worst scalar count / 128');ax.legend(fontsize=7)
        ax=axes[1,0]
        for experiment,arms in [('R10',('balanced_16384','balanced_65536')),('R11',('reference','constant','decay'))]:
            for arm in arms:
                ss=select('return_tail_cells',experiment=experiment,head=arm,split='validation')
                ax.scatter(experiment+' / '+arm,100*min(r['correct']['value']/r['total'] for r in ss))
        ax.set(title='Original-mixture reused-validation minimum',ylabel='Scalar correct (%)');ax.tick_params(axis='x',labelrotation=25)
        ax=axes[1,1]
        for arm in ('reference','constant','decay'):
            ss=select('return_tail_cells',experiment='R11',head=arm,split='train',target_delay=16,distractors=2)
            assert len(ss)==1
            ax.bar(arm,100*ss[0]['correct']['value']/ss[0]['total'])
        ax.set(title='R11 training fit · delay 16 / 2 distractors',ylabel='Scalar correct (%)',ylim=(98,100))
        finish(fig,'return-tail-development','R10/R11 fixed development screens fail: calibration threshold governs; reused validation cannot override failure. One development backbone; no promotion.\nR11 decay versus constant identifies a relative learning-rate effect; reference is the inherited endpoint, not a same-initialization additional-exposure control.')
    if select('s13_learning'):
        fig,axes=plt.subplots(2,2,figsize=(12,7.5))
        for ax,surface in zip(axes[0],('english','spanish')):
            for i,arm in enumerate(('english','mixed')):
                for policy,style in [('raw','--'),('calibrated','-')]:
                    ss=select('s13_learning',arm=arm,surface=surface,policy=policy);ss.sort(key=lambda r:r['added_update'])
                    ax.plot([r['added_update'] for r in ss],[r['exact'] for r in ss],style,marker='o',color=f'C{i}',label=f'{arm}-training / {policy}')
            ax.set(title=f'S13 {surface} complete graphs',xlabel='Additional optimizer updates from shared parent',ylabel='Complete graphs / 512',ylim=(-3,150));ax.legend(fontsize=7)
        for ax,metric,title in [(axes[1,0],'mean_copy','Spanish mean copy accuracy'),(axes[1,1],'ordered_edge','Spanish ordered-edge F1')]:
            for i,arm in enumerate(('english','mixed')):
                for policy,style in [('raw','--'),('calibrated','-')]:
                    ss=select('s13_learning',arm=arm,surface='spanish',policy=policy);ss.sort(key=lambda r:r['added_update'])
                    values=[r[metric] if metric=='mean_copy' else r['edges'][metric]['f1'] for r in ss]
                    ax.plot([r['added_update'] for r in ss],values,style,marker='o',color=f'C{i}',label=f'{arm}-training / {policy}')
            ax.set(title=title,xlabel='Additional optimizer updates',ylabel='Component score',ylim=(-.02,1.02));ax.legend(fontsize=7)
        english_end=max(select('s13_learning',arm='english',surface='english',policy='calibrated'),key=lambda r:r['added_update'])
        mixed_end=max(select('s13_learning',arm='mixed',surface='english',policy='calibrated'),key=lambda r:r['added_update'])
        finish(fig,'semantic-bilingual-development',f'S13: one paired development parent; all promotion criteria fail. Spanish complete graphs remain zero despite partial copy/order acquisition.\nEnglish calibrated endpoint falls from English-only {english_end["exact"]:g} to mixed {mixed_end["exact"]:g} /{mixed_end["total"]}. Token/compute exposure differs; these are not confirmation curves or broad multilingual competence.')
    if select('s15_learning'):
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        titles={'3x3':'3×3 · new trained motif','3x4':'3×4 · held-out combination','4x3':'4×3 · known motif retention','4x4':'4×4 · known motif retention'}
        for ax,shape in zip((axes[0,0],axes[0,1],axes[1,0],axes[1,1]),titles):
            for i,arm in enumerate(('control','mixed')):
                for policy,style in [('raw','--'),('calibrated','-')]:
                    ss=select('s15_learning',shape=shape,arm=arm,policy=policy);ss.sort(key=lambda r:r['added_update'])
                    ax.plot([r['added_update'] for r in ss],[r['exact'] for r in ss],style,marker='o',color=f'C{i}',label=f'{arm} / {policy}')
                    if policy=='calibrated':
                        ax.annotate(f"{ss[-1]['exact']:g}",(ss[-1]['added_update'],ss[-1]['exact']),xytext=(5,6+i*10 if shape in ('3x3','3x4') else (i-.5)*10),textcoords='offset points',fontsize=8,color=f'C{i}')
            if shape in ('3x3','3x4'):ax.axhline(52,color='.6',ls=':',lw=.8)
            ax.set(title=titles[shape],xlabel='Added optimizer updates',ylabel='Complete graphs / 512',ylim=(-3,250));ax.legend(fontsize=6)
        for ax,shape in zip(axes[:,2],('3x3','3x4')):
            for i,arm in enumerate(('control','mixed')):
                ss=select('s15_learning',shape=shape,arm=arm,policy='calibrated');ss.sort(key=lambda r:r['added_update'])
                for metric,style in [('copy','-'),('ordered F1','--')]:
                    values=[r['mean_copy'] if metric=='copy' else r['edges']['ordered_edge']['f1'] for r in ss]
                    ax.plot([r['added_update'] for r in ss],values,style,marker='o',color=f'C{i}',label=f'{arm} / {metric}')
            ax.set(title=titles[shape]+' · components',xlabel='Added optimizer updates',ylabel='Component score',ylim=(-.02,1.02));ax.legend(fontsize=6)
        criteria=select('s15_paired')[0]['criteria']
        failed=', '.join(k for k in ('acquisition','retention','recombination') if not criteria[k])
        finish(fig,'semantic-shape-development',f'S15: one S14-informed development pair; failed gates: {failed}. No confirmation or symmetric extension eligible.\nFixed historical TRAIN128 calibration. Construction exposure matched; tokens/graph sizes differ. Component learning does not establish complete acquisition or isolate recombination.')
    if select('s16_normalization'):
        fig,axes=plt.subplots(1,2,figsize=(10,4.6))
        for ax,policy in zip(axes,('raw','calibrated')):
            for row in select('s16_normalization',policy=policy):
                arm=row['arm'];seed=row['seed']
                ax.plot([0,1],[row['original']['exact'],row['normalized']['exact']],'o-' if arm=='decay' else 'x--',color=f'C{seed-701}',label=f'{seed} / {arm}')
            ax.set(title=f'S16 frozen models · {policy}',xticks=[0,1],xticklabels=['Original public input','Programmed normalization'],ylabel='Complete graphs / 1024');ax.legend(fontsize=7)
        finish(fig,'semantic-input-diagnostic','Six frozen models share the same 1024 semantic instances; model×event cells are not independent replication.\nNormalization is a programmed input transformation, not learned invariance. Historical correctness gates remain unchanged.')
    if select('s17_calibration'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        shapes=('3x3','3x4','4x3','4x4')
        for ax,arm in zip(axes,('control','mixed')):
            for i,(policy,label) in enumerate([('raw','Raw'),('historical','S15 historical TRAIN128'),('matched','Actual endpoint TRAIN128')]):
                ss=[next(r for r in select('s17_calibration',arm=arm) if r['shape']==shape) for shape in shapes]
                ax.plot(range(4),[r['policies'][policy]['complete'] for r in ss],'.--' if policy=='raw' else '.-',label=label)
                if policy!='raw':
                    for j,row in enumerate(ss): ax.annotate(str(row['policies'][policy]['complete']),(j,row['policies'][policy]['complete']),xytext=(3,(i-1)*10),textcoords='offset points',fontsize=7)
            ax.set(title=f'S17 {arm}: frozen S15 endpoint',xticks=range(4),xticklabels=['3×3 new','3×4 held out','4×3 known','4×4 known'],ylabel='Complete graphs / 512',ylim=(-5,275));ax.legend(fontsize=7)
        finish(fig,'semantic-calibration-diagnostic','Exploratory TRAIN-only recalibration: same raw predictions and DEV targets; one threshold policy per endpoint across all four cells.\nNo new training, DEV fitting, learned invariance, checkpoint selection or retroactive S15 gate promotion. Component F1 retained in data is macro across graphs.')
    if select('s18_learning'):
        arms=('original','context','workspace_control');labels=('Reused original','Context','10-pass control')
        shapes=('3x3','3x4','4x3','4x4')
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,shape in zip(axes.flat,shapes):
            for i,arm in enumerate(arms):
                for policy,style in [('raw','--'),('matched','-')]:
                    ss=select('s18_learning',arm=arm,shape=shape,policy=policy);ss.sort(key=lambda r:r['added_update'])
                    ax.plot([r['added_update'] for r in ss],[r['complete'] for r in ss],style,marker='.',color=f'C{i}',label=f'{labels[i]} / {policy}')
            ax.set(title=f'S18 {shape} · '+('new trained' if shape=='3x3' else 'held out' if shape=='3x4' else 'known'),xlabel='Added optimizer updates',ylabel='Complete graphs / 512',ylim=(-3,230));ax.legend(fontsize=6)
        finish(fig,'semantic-context-development','All fixed acquisition, retention and recombination gates fail. Matched TRAIN calibration is primary; raw is separate. One inspected development parent.\nAll arms share the S11 parent, 57,853,781 parameters and incremental stream; arithmetic/runtime differ. Original training is reused; curves retrospectively calibrated.')
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for ax,policy in zip(axes,('raw','matched')):
            for i,arm in enumerate(arms):
                train=select('s18_train',arm=arm,shape='3x3',added_update=4096,calibration='matched',policy='raw' if policy=='raw' else 'calibrated')[0]
                dev=select('s18_learning',arm=arm,shape='3x3',added_update=4096,policy=policy)[0]
                ax.plot([0,1],[100*r['complete']/r['examples'] for r in (train,dev)],'o-',color=f'C{i}',label=labels[i])
                for j,row in enumerate((train,dev)):ax.annotate(f"{row['complete']}/{row['examples']}",(j,100*row['complete']/row['examples']),xytext=(4,(i-1)*9),textcoords='offset points',fontsize=7,color=f'C{i}')
            ax.set(title=f'S18 3×3 acquisition · {policy}',xticks=[0,1],xticklabels=['Actual TRAIN (calibration overlap)','DEV (inspected)'],ylabel='Complete graphs (%)',ylim=(-.8,13));ax.legend(fontsize=7)
        totals=[]
        for arm in arms:
            ss=select('s18_train',arm=arm,added_update=4096,calibration='matched',policy='calibrated')
            totals.append(f"{sum(r['complete'] for r in ss)}/{sum(r['examples'] for r in ss)}")
        finish(fig,'semantic-context-train-dev','Matched full TRAIN128 totals (original/context/10-pass): '+', '.join(totals)+'. Calibration overlap makes these optimistic in-sample fit measures.\nShared S11 parent, 57,853,781 parameters and incremental stream; arithmetic/runtime differ. Reused original training.\nLow complete counts coexist with strong but imperfect relation ranking. One lineage; no confirmation or automatic extension.')
    if select('s19_learning'):
        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        ss=sorted(select('s19_learning',split='development'),key=lambda r:r['added_update'])
        shapes=('3x3','3x4','4x3','4x4')
        for i,shape in enumerate(shapes):
            axes[0].plot([r['added_update'] for r in ss],[100*r['cells'][shape]['complete']/r['cells'][shape]['examples'] for r in ss], 's--' if shape=='3x4' else 'o-',color=f'C{i}',label=f'DEV {shape}'+(' held out' if shape=='3x4' else ' trained'))
        train=sorted(select('s19_learning',split='train'),key=lambda r:r['added_update'])
        axes[0].plot([r['added_update'] for r in train],[100*r['complete']/r['examples'] for r in train],'k:',label='TRAIN128 (exposed)')
        axes[0].set(title='S19 categorical greedy learning',xlabel='Optimizer updates',ylabel='Complete graphs (%)',ylim=(-3,103));axes[0].legend(fontsize=7)
        axes[1].plot(range(4),[ss[-1]['cells'][shape]['complete'] for shape in shapes],'o-',label='S19 greedy / scratch')
        for arm,label in [('original','S18 original'),('context','S18 contextual'),('workspace_control','S18 ten-pass')]:
            for policy,style in [('raw',':'),('matched','--')]:
                values=[select('s18_learning',arm=arm,shape=shape,added_update=4096,policy=policy)[0]['complete'] for shape in shapes]
                axes[1].plot(range(4),values,style,marker='.',label=f'{label} / {policy}')
        for i,shape in enumerate(shapes):axes[1].annotate(str(ss[-1]['cells'][shape]['complete']),(i,ss[-1]['cells'][shape]['complete']),xytext=(4,4),textcoords='offset points',fontsize=8)
        axes[1].set(title='Fixed endpoint · same DEV events',xticks=range(4),xticklabels=['3×3 trained','3×4 held out','4×3 trained','4×4 trained'],ylabel='Complete graphs / 512',ylim=(-10,555));axes[1].legend(fontsize=6)
        finish(fig,'semantic-sequential-development','Known-motif acquisition passes; held-out combination fails. One inspected development seed; no confirmation.\nS19: 62,677,315 parameters, scratch; S18: 57,853,781 parameters, inherited training. Same incremental construction stream, different objectives/interfaces.\nNo parameter/FLOP/history match or isolated architecture claim. Gold-prefix diagnostics are not generated-graph accuracy.')
    if select('s20_endpoint'):
        shapes=('3x3','3x4','4x3','4x4')
        fig,axes=plt.subplots(1,3,figsize=(13,5.1),sharey=True)
        for ax,seed in zip(axes,(701,702,703)):
            record=select('s20_endpoint',seed=seed)[0]['counts']
            ax.plot(range(4),[record[s] for s in shapes],'ko-',label='Record / categorical')
            for arm,color in [('original','C0'),('context','C1')]:
                for policy,style in [('raw',':'),('matched','--')]:
                    row=select('s20_paired',seed=seed,comparison=f'{arm}/{policy}')[0]
                    ax.plot(range(4),[row['cells'][s]['baseline_complete'] for s in shapes],style,marker='.',color=color,label=f'{arm} / {policy}')
            ax.set(title=f'S20 lineage {seed}',xticks=range(4),xticklabels=['3×3','3×4\nheld out','4×3','4×4'],ylim=(-10,530),ylabel='Complete confirmation graphs / 512');ax.legend(fontsize=6)
        gates=select('s20_summary')[0]['decisions']
        outcome='Known-motif claim: '+('PASS' if gates['replicated_known_claim'] else 'FAIL')+'; held-out competence: '+('PASS' if gates['heldout_claim'] else 'FAIL')+'. '
        finish(fig,'semantic-confirmation-shapes',outcome+'Record is categorical, without edge calibration.\nRecord: scratch, 62,677,315 parameters; workspace: inherited parents, 57,853,781 parameters. Same incremental stream; different objectives/history/arithmetic.\nNo isolated architecture or matched-FLOP claim. TRAIN fitting/calibration overlap is separate from fresh confirmation.')
        fig,axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True)
        summary=select('s20_summary')[0]
        for ax,arm in zip(axes,('original','context')):
            for policy,color,offset in [('raw','C0',-.10),('matched','C1',.10)]:
                ss=[select('s20_paired',seed=seed,comparison=f'{arm}/{policy}')[0] for seed in (701,702,703)]
                centers=[r['known_macro_delta_percentage_points'] for r in ss]
                intervals=[r['conditional_event_ci95_percentage_points'] for r in ss]
                centers.append(sum(centers)/3)
                intervals.append(summary['mean_ci95'][f'{arm}/{policy}'])
                x=[i+offset for i in range(4)]
                ax.vlines(x,[c[0] for c in intervals],[c[1] for c in intervals],color=color)
                ax.plot(x,centers,'o',color=color,label=f'Record minus {arm} / {policy}')
            ax.set(title=f'Known-motif difference versus {arm}',xticks=range(4),xticklabels=['701','702','703','Fixed-model mean'],ylabel='Paired macro difference (percentage points)');ax.legend(fontsize=7)
        endpoints=[v for row in select('s20_paired') for v in row['conditional_event_ci95_percentage_points']]
        for ax in axes: ax.set_ylim(min(endpoints)-1,max(endpoints)+1)
        finish(fig,'semantic-confirmation-differences','Vertical axis zoomed to effects. 95% paired event-bootstrap intervals use shared draws within the three known cells across all fixed models and policies.\nIntervals condition on these trained models; the fixed-model mean is not seed-population uncertainty. Both baselines and all seeds retained.\nHeld-out competence is separate; positive direction alone does not establish the registered all-seed acquisition claim.')
    if select('s21_learning'):
        groups=[('Old known',('3x3','4x3','4x4')),('Newly exposed in broad',('2x4','5x3','5x4')),('Omitted combination',('3x4',))]
        fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
        for ax,(label,shapes) in zip(axes,groups):
            for i,shape in enumerate(shapes):
                for arm,style in [('original','--'),('broad','-')]:
                    ss=sorted(select('s21_learning',arm=arm,split='development'),key=lambda r:r['added_update'])
                    ax.plot([r['added_update'] for r in ss],[r['cells'][shape]['complete'] for r in ss],style,marker='.',color=f'C{i}',label=f'{shape} / {arm}')
            ax.set(title=label,xlabel='Optimizer updates',ylabel='Complete DEV graphs / 512',ylim=(-10,530));ax.legend(fontsize=6)
        finish(fig,'semantic-broad-motif-learning','S21 paired development: same scratch initialization and learner; original three-motif versus broad six-motif corpus.\nEqual 32,768 presentations; token/node/edge/record exposure and runtime differ. Curves are inspected DEV, not sealed confirmation.\nAll curves retained; only the fixed 4096 endpoint determines the registered joint gate.')
        summary=select('s21_summary')[0];shapes=('3x3','4x3','4x4','2x4','5x3','5x4','3x4')
        fig,axes=plt.subplots(1,2,figsize=(12,5.2))
        labels=['3×3\nold','4×3\nold','4×4\nold','2×4\nnew','5×3\nnew','5×4\nnew','3×4\nheld out']
        for arm,color,offset in [('original','C0',-.1),('broad','C1',.1)]:
            values=[summary['endpoint_counts'][arm][s] for s in shapes]
            axes[0].plot([i+offset for i in range(7)],values,'o',color=color,label=arm)
            for i,v in enumerate(values):axes[0].annotate(str(v),(i+offset,v),xytext=(0,5 if arm=='broad' else -12),textcoords='offset points',ha='center',fontsize=6,color=color)
        axes[0].set(title='Fixed endpoint counts',xticks=range(7),xticklabels=labels,ylabel='Complete DEV graphs / 512',ylim=(-25,545));axes[0].legend(fontsize=7)
        delta=[100*summary['paired'][s+'/4096']['delta_complete']/512 for s in shapes]
        intervals=[summary['ci95'][s] for s in shapes]
        axes[1].vlines(range(7),[v[0] for v in intervals],[v[1] for v in intervals],color='C2')
        axes[1].plot(range(7),delta,'o',color='C2');axes[1].axhline(0,color='gray',linewidth=.8)
        axes[1].set(title='Broad minus original · paired effects',xticks=range(7),xticklabels=labels,ylabel='Difference (percentage points)')
        d=summary['decisions'];gate=lambda value:'PASS' if value else 'FAIL'
        caption=f"Held-out gain: {gate(d['heldout_gain'])}; old-known retention: {gate(d['old_known_retention'])} (loss {d['old_known_loss_complete']}); new-motif acquisition: {gate(d['new_motif_acquisition'])}. Joint: {gate(d['advance'])}."
        finish(fig,'semantic-broad-motif-endpoints',caption+' Conditional confirmation: '+('eligible' if d['advance'] else 'INELIGIBLE')+'.\n95% event intervals condition on one paired initialization and inspected DEV; they are not seed uncertainty or gate criteria.\nCanonical exactness is distinct from graph isomorphism; a node-position error alone does not rule out all permutations.')
    if select('c04_hybrid'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        audited_lineages=sorted({r['lineage'] for r in select('c04_hybrid') if r.get('audit_status')=='independently_audited'})
        provisional_lineages=sorted({r['lineage'] for r in select('c04_hybrid') if r.get('audit_status')!='independently_audited'})
        pending_lineages=[n for n in (560,561,562) if n not in audited_lineages and n not in provisional_lineages]
        matrix_label='All three lineages independently audited.' if len(audited_lineages)==3 else f'Audited: {audited_lineages}; independent audit pending: {provisional_lineages}; missing: {pending_lineages}.'
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
        finish(fig,'composition-confirmation',f"{matrix_label} Each trace is a lineage; shared views/delays are not independent.\nNeural x = fixed 4000 primary; + = selected secondary. Supplied-copy is a sole-return reference. Finite numeric train/test overlap; supplied scheduling.")

        if select('c04_timing'):
            fig,axes=plt.subplots(1,3,figsize=(13,4.8))
            for ai,arm in enumerate(('n1_roles','workspace','supplied_copy')):
                for lineage in (560,561,562):
                    ss=select('c04_neural',arm=arm,endpoint='endpoint',lineage=lineage,metric='answer') if arm=='n1_roles' else select('c04_hybrid',path=arm,lineage=lineage,metric='joint',distractors=8)
                    if arm!='n1_roles': ss=[r for r in ss if r['delay'] in (None,16)]
                    ss.sort(key=lambda r:r['view'])
                    axes[0].plot([0 if r['view']=='clean' else 1 for r in ss],[100*r['correct']/r['total'] for r in ss],'.-',color=f'C{ai}',alpha=.65,label=arm if lineage==560 else None)
            axes[0].set(title='Primary fixed roles vs interfaces',xticks=[0,1],xticklabels=['Clean','Reversed'],ylabel='Correct (%)');axes[0].legend(fontsize=7)
            for ax,batch in zip(axes[1:],(1,64)):
                for ai,arm in enumerate(('n1_roles','workspace','supplied_copy')):
                    ss=[r for r in select('c04_timing',path=arm,batch=batch) if r['delay'] in (None,16)]
                    ax.scatter([ai+(r['lineage']-561)*.07 for r in ss],[1000*r['median_seconds']/r['attempted'] for r in ss],s=25,label=arm)
                ax.set(title=f'Measured execution · batch {batch}',xticks=[0,1,2],xticklabels=['Fixed roles','Workspace d16','Supplied copy'],ylabel='Median ms / attempted example',yscale='log');ax.tick_params(axis='x',labelrotation=15)
            finish(fig,'composition-accuracy-timing',f'{matrix_label} Accuracy: neural answer; hybrid joint lowering + answer.\nEach dot/trace is one lineage; timing uses median of recorded repeats, excludes setup, and does not measure training or prove a deployment speedup.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path)
    p.add_argument('--extract-only',action='store_true');p.add_argument('--render-data',type=Path)
    p.add_argument('--c04-audit',type=Path,action='append',help='Repeatable repository-relative input_sha256 audit receipt; never infer audit from presence alone')
    p.add_argument('--c04-provisional',type=int,choices=(0,1,2),action='append',help='Explicit completed lineage pending independent audit; visibly labeled provisional')
    a=p.parse_args()
    output=a.output or a.root/'research/campaigns/extended-01/figures';output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if a.render_data:
        with (gzip.open(a.render_data,'rt') if a.render_data.suffix=='.gz' else a.render_data.open()) as f:
            data=json.load(f)
    else:
        data=extract(a.root.resolve(),a.c04_audit,a.c04_provisional)
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
