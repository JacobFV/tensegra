"""Artifact-only Stage 6 summaries; execute this file directly to avoid Torch import."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def ratio(n, d):
    return n / d if d else None


def describe(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('nonfinite metric')
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def verify_files(expected, root):
    records=[]
    for name,wanted in sorted(expected.items()):
        path=Path(root)/name
        actual=file_hash(path) if path.is_file() else None
        records.append(dict(path=name,expected=wanted,actual=actual,match=actual==wanted))
    return dict(ok=all(r['match'] for r in records),files=records)


def calibration(rows):
    bins=[dict(n=0,probability_sum=0.,target_sum=0.) for _ in range(10)]
    thresholds=[dict(threshold=i/10,accepted=0,target_sum=0.) for i in range(11)]
    n=0;square=0.
    for row in rows:
        p,t=row['probability'],row['target']
        if not (0<=p<=1 and 0<=t<=1):raise ValueError('invalid readiness calibration')
        n+=1;square+=(p-t)**2
        bucket=bins[min(9,int(p*10))];bucket['n']+=1;bucket['probability_sum']+=p;bucket['target_sum']+=t
        for point in thresholds:
            if p>=point['threshold']:point['accepted']+=1;point['target_sum']+=t
    return dict(n=n,squared_error_sum=square,bins=bins,risk_coverage=thresholds)


def merge_calibration(parts):
    n=sum(p['n'] for p in parts)
    bins=[];risks=[]
    for i in range(10):
        b={k:sum(p['bins'][i][k] for p in parts) for k in ('n','probability_sum','target_sum')}
        bins.append(dict(lower=i/10,upper=(i+1)/10,**b,mean_probability=ratio(b['probability_sum'],b['n']),mean_target=ratio(b['target_sum'],b['n'])))
    for i in range(11):
        accepted=sum(p['risk_coverage'][i]['accepted'] for p in parts)
        target=sum(p['risk_coverage'][i]['target_sum'] for p in parts)
        risks.append(dict(threshold=i/10,accepted=accepted,denominator=n,coverage=ratio(accepted,n),risk=ratio(accepted-target,accepted)))
    return dict(n=n,brier=ratio(sum(p['squared_error_sum'] for p in parts),n),bins=bins,risk_coverage=risks)


FLAGS={'task_accuracy':'task_correct','exact_semantic_accuracy':'exact_semantic_correct',
       'trajectory_exact_accuracy':'trajectory_exact','numeric_accuracy':'numeric_correct','decision_accuracy':'decision_correct',
       'semantic_decision_accuracy':'semantic_decision_correct','transition_set_exact_accuracy':'transition_set_exact',
       'halt_rate':'halted','max_forced_rate':'max_forced','post_event_pass_rate':'post_event_pass'}


def controlled_cell(row):
    counts=Counter(examples=len(row['examples']));values=defaultdict(list);histogram=Counter()
    for e in row['examples']:
        for metric,flag in FLAGS.items():
            if flag in e:counts[metric+'_correct']+=int(e[flag]);counts[metric+'_denominator']+=1
        counts['task_and_exact']+=int(e['task_correct'] and e['exact_semantic_correct'])
        counts['exact']+=int(e['exact_semantic_correct'])
        counts['early_halt_incomplete']+=int(e.get('halted',False) and not e['trajectory_exact'])
        values['microsteps'].append(e['microsteps']);histogram[str(e['microsteps'])]+=1
        if 'injected_events' in e:values['injected_events'].append(e['injected_events'])
        for step in e.get('trace',[]):
            counts['microstep_observations']+=1
            counts['proposals']+=len(step.get('proposals',[]))
            for event in step.get('events',[]):
                counts['runtime_'+event['status']]+=1;counts['runtime_events']+=1
                if event.get('reason'):counts['reason:'+event['reason']]+=1
            for metric in ('projection_overlap','grounding_entropy','null_binding_mass','emit_probability'):
                if metric in step:values[metric].append(step[metric])
            for event in step.get('event_roundtrips',[]):
                counts['roundtrip_events']+=1
                for metric,value in event.items():
                    if isinstance(value,(int,float)):
                        values['roundtrip_'+event.get('stage','unspecified')+'_'+metric].append(value)
    if row.get('n',counts['examples'])!=counts['examples']:raise ValueError('evaluation population mismatch')
    metrics={metric:ratio(counts[metric+'_correct'],counts[metric+'_denominator']) for metric in FLAGS}
    metrics.update({k:describe(v)['mean'] for k,v in values.items()})
    metrics['task_given_exact']=ratio(counts['task_and_exact'],counts['exact'])
    metrics['early_halt_incomplete_rate']=ratio(counts['early_halt_incomplete'],counts['examples'])
    for status in ('executed','duplicate','conflict','deferred','refused'):
        metrics['runtime_'+status+'_rate']=ratio(counts['runtime_'+status],counts['runtime_events'])
    if 'evaluation_seconds' in row:metrics['evaluation_seconds']=row['evaluation_seconds']
    for key,value in metrics.items():
        if key in row and row[key] is not None and value is not None and not math.isclose(value,row[key],abs_tol=1e-7):raise ValueError('inconsistent stored metric '+key)
    return dict(seed=row['seed'],metrics=metrics,counts=dict(counts),calibration=calibration(row.get('readiness_calibration',[])),
        data_hash=digest([(e.get('input_hash'),e.get('data_hash')) for e in row['examples']]),initial_hash=row.get('initial_hash'),
        compute_cap=row.get('compute_cap'),microstep_histogram=dict(histogram),observation_protocol=row.get('observation_protocol'),execution_backend=row.get('execution_backend'))


def aggregate(items):
    metrics=set().union(*(x['metrics'] for x in items))
    counts=Counter();histogram=Counter()
    for item in items:
        counts.update(item.get('counts',{}));histogram.update(item.get('microstep_histogram',{}))
    return dict(execution_backends=sorted({x['execution_backend'] for x in items if x.get('execution_backend')}),observation_protocols=sorted({x['observation_protocol'] for x in items if x.get('observation_protocol')}),seeds=sorted(x['seed'] for x in items),metrics={k:describe([x['metrics'].get(k) for x in items]) for k in sorted(metrics)},counts=dict(counts),microstep_histogram=dict(histogram))


def contrasts(groups, reference, fields):
    output=[]
    for key,items in sorted(groups.items()):
        if key[0]==reference or (reference,*key[1:]) not in groups:continue
        right={x['seed']:x for x in groups[(reference,*key[1:])]};left={x['seed']:x for x in items}
        matched=sorted(left.keys()&right.keys())
        metrics=set().union(*(left[s]['metrics'].keys()&right[s]['metrics'].keys() for s in matched)) if matched else set()
        differences={m:describe([left[s]['metrics'][m]-right[s]['metrics'][m] for s in matched if left[s]['metrics'][m] is not None and right[s]['metrics'][m] is not None]) for m in sorted(metrics)}
        output.append(dict(zip(fields,key),reference=reference,seeds=matched,unmatched_seeds=sorted(left.keys()^right.keys()),differences=differences,matched_observation_protocol=(all(left[s].get('observation_protocol')==right[s].get('observation_protocol') for s in matched) if matched and all(left[s].get('observation_protocol') and right[s].get('observation_protocol') for s in matched) else None)))
    return output


def summarize_controlled(rows):
    groups=defaultdict(list);training=defaultdict(list);seen=set();sources=set();initials={};data={};schedules={};training_seen=set();objectives=set()
    for row in rows:
        if row['kind']=='training':
            train_id=(row['variant'],row['seed'],row['step'])
            if train_id in training_seen:raise ValueError('duplicate training row')
            training_seen.add(train_id)
            if row.get('objective'):objectives.add(row['objective'])
            key=row['variant'],row['step'];identity=(row['seed'],row['step'])
            if identity in schedules and schedules[identity]!=row.get('data_hash'):raise ValueError('unpaired training data')
            schedules[identity]=row.get('data_hash')
            training[key].append(dict(seed=row['seed'],metrics={**{'loss_'+k:v for k,v in row['losses'].items()},**{'weight_'+k:v for k,v in row.get('auxiliary_weights',{}).items()},**({'train_seconds':row['train_seconds']} if 'train_seconds' in row else {}),**({'auxiliary_teacher_forcing_rate':float(row['auxiliary_teacher_forcing'])} if 'auxiliary_teacher_forcing' in row else {})}));continue
        if row['kind']!='evaluation':raise ValueError('unknown row kind')
        key=(row['variant'],row.get('condition','depth'),row['step'],row['depth'],row.get('intervention','none'))
        identity=(*key,row['seed'])
        if identity in seen:raise ValueError('duplicate evaluation')
        seen.add(identity);sources.add(row.get('source_hash'))
        cell=controlled_cell(row);pair=(row['seed'],row['step'],row['depth'], 'depth' if key[1] in ('frozen_intervention','oracle_trace','oracle_minimal') else key[1])
        if pair in data and data[pair]!=cell['data_hash']:raise ValueError('unpaired evaluation data')
        data[pair]=cell['data_hash']
        if row['seed'] in initials and initials[row['seed']]!=cell['initial_hash']:raise ValueError('unpaired initialization')
        initials[row['seed']]=cell['initial_hash'];groups[key].append(cell)
    if len(sources)>1:raise ValueError('mixed source hashes')
    fields=('variant','condition','step','depth','intervention')
    cells=[dict(zip(fields,key),**aggregate(items),calibration=merge_calibration([i['calibration'] for i in items]),compute_caps=sorted({i['compute_cap'] for i in items if i['compute_cap'] is not None})) for key,items in sorted(groups.items())]
    interventions=[]
    for key,items in groups.items():
        if key[4] in ('none','oracle_trace','oracle_minimal'):continue
        reference=groups.get((key[0],'depth',key[2],key[3],'none'),[])
        paired={('control',*key[1:]):reference,('intervention',*key[1:]):items}
        interventions.extend(contrasts(paired,'control',fields))
    return dict(track='controlled_execution',aggregates=[c for c in cells if c['condition'] not in ('oracle_trace','oracle_minimal')],privileged_oracles=[c for c in cells if c['condition'] in ('oracle_trace','oracle_minimal')],paired_contrasts=contrasts(groups,'local',fields),frozen_intervention_contrasts=interventions,
        curves=[dict(variant=v,step=s,**aggregate(items)) for (v,s),items in sorted(training.items())],
        provenance=dict(source_hashes=sorted(s for s in sources if s),initial_hashes=initials,evaluation_cells=len(seen),training_schedule_steps=len(schedules),training_objectives=sorted(objectives)),
        caveats=['Supplied progressively disclosed interface is not TCN language induction.',
                 'Exact runtime semantics are supplied; task outputs are learned.',
                 'Readiness risk is 1 minus posterior target among accepted proposals, not task error.',
                 'Runtime status rates use returned events; proposal count is separately reported.',
                 'Early halt incomplete means learned halt with incomplete exact trajectory.',
                 'Fixed and recurrent arms have different compute; consult compute_caps and microsteps.',
                 'Missing metrics are unsupported; undefined denominators remain null.',
                 'Oracle trace evaluations use privileged targets and must not be ranked as autonomous actors.',
                 'Unavailable or nonnumeric argument refusal is combined by the runtime; pure type validity cannot be recovered.'])


def summarize_language(rows, losses=()):
    groups=defaultdict(list);seen=set();initials={};data_hashes=set();config_hashes=set();provenance_rows=0;total_rows=0
    def record_identity(row):
        if row.get('initial_hash'):
            seed=row['seed']
            if seed in initials and initials[seed]!=row['initial_hash']:raise ValueError('unpaired language initialization')
            initials[seed]=row['initial_hash']
        if row.get('data_hash'):data_hashes.add(row['data_hash'])
        if row.get('config_hash'):config_hashes.add(row['config_hash'])
        if len(data_hashes)>1 or len(config_hashes)>1:raise ValueError('mixed language data/config identity')
    for row in rows:
        record_identity(row);total_rows+=1
        provenance_rows+=int(all(row.get(k) for k in ('initial_hash','data_hash','config_hash')))
        for renderer,cell in row['evaluation'].items():
            key=(row['arm'],renderer,row['step']);identity=(*key,row['seed'])
            if identity in seen:raise ValueError('duplicate language evaluation')
            seen.add(identity)
            metrics={k:v for k,v in cell.items() if (v is None or isinstance(v,(int,float))) and not k.startswith('oracle_')}
            metrics.update({'task_'+k:v for k,v in cell.get('per_task_accuracy',{}).items()})
            counts={'examples':sum(cell.get('task_counts',{}).values()),**{'task_'+k:v for k,v in cell.get('task_counts',{}).items()}}
            for kind,populations in cell.get('graph_counts',{}).items():
                if any(not isinstance(v,int) or v<0 for v in populations.values()):raise ValueError('invalid graph counts')
                if populations['true_positive']>min(populations['predicted_count'],populations['gold_count']):raise ValueError('invalid graph true positives')
                counts.update({kind+'_'+k:v for k,v in populations.items()})
            for kind,defined in cell.get('graph_defined_examples',{}).items():
                counts.update({kind+'_'+metric+'_defined_examples':v for metric,v in defined.items()})
            groups[key].append(dict(seed=row['seed'],metrics=metrics,counts=counts,oracles={k:v for k,v in cell.items() if k.startswith('oracle_')}))
    curves=defaultdict(list);loss_seen=set()
    for row in losses:
        record_identity(row);key=(row['arm'],row['seed'],row['step'])
        if key in loss_seen:raise ValueError('duplicate language loss row')
        loss_seen.add(key)
        curves[(row['arm'],row['step'])].append(dict(seed=row['seed'],metrics={'loss':row['loss'],**row.get('parts',{})}))
    fields=('variant','renderer','step');cells=[]
    for key,items in sorted(groups.items()):
        oracles=set().union(*(x['oracles'] for x in items));cell=aggregate(items);counts=cell['counts'];pooled={}
        for kind in ('node','typed_edge'):
            if all(kind+'_true_positive' in item['counts'] for item in items):
                pooled[kind+'_precision']=ratio(counts[kind+'_true_positive'],counts[kind+'_predicted_count'])
                pooled[kind+'_recall']=ratio(counts[kind+'_true_positive'],counts[kind+'_gold_count'])
        cells.append(dict(zip(fields,key),**cell,pooled_graph_rates=pooled,privileged_baselines={k:describe([x['oracles'].get(k) for x in items]) for k in sorted(oracles)}))
    return dict(track='tcn_semantic_decoder',aggregates=cells,paired_contrasts=contrasts(groups,'single_pass',fields),
        provenance=dict(initial_hashes=initials,data_hashes=sorted(data_hashes),config_hashes=sorted(config_hashes),evaluation_rows=total_rows,rows_with_complete_identity=provenance_rows),
        curves=[dict(variant=v,step=s,**aggregate(items)) for (v,s),items in sorted(curves.items())],
        caveats=['Graph decoder scores are separate from controlled runtime execution.',
                 'Canonical node alignment and lexical hash fidelity are supplied representation choices.',
                 'Spanish is exposed in the consistency arm; symbols are withheld.',
                 'No heldout lexical or motif claim without an explicitly evaluated split.',
                 'Decoder metrics are per-example macro averages; pooled_graph_rates are separately computed only when raw populations exist.',
                 'Old pilots with no graph_counts use producer zero-denominator convention, which cannot be reconstructed.',
                 'Oracle position majority uses heldout labels and is privileged.'])


def audit_language(data, manifest, root):
    modern='config_hash' in data
    expected_data=data.get('examples',[])+(data.get('renamed_examples',[]) if modern else [])
    checkpoint_expected={r['checkpoint']:r['checkpoint_sha256'] for r in manifest.get('runs',[]) if r.get('checkpoint_sha256')}
    return dict(data_hash_matches=data.get('data_hash')==digest(expected_data),
        config_hash_matches=data.get('config_hash')==digest(data.get('config',{})) if modern else None,
        manifest_data_hash_matches=manifest.get('data_hash')==data.get('data_hash') if manifest else None,
        manifest_config_hash_matches=manifest.get('config_hash')==data.get('config_hash') if manifest else None,
        semantic_split_disjoint=not(set(data.get('semantic_train',[]))&set(data.get('semantic_eval',[]))),
        checkpoint_hashes=verify_files(checkpoint_expected,root) if checkpoint_expected else None,
        producer_run_manifest_present=bool(manifest.get('runs')),
        denominator_semantics='undefined_is_null_with_raw_counts' if modern else 'legacy_zero_undefined_no_raw_counts')


def expected_controlled_evaluations(config):
    """Mirror the declared runner schedule, retaining old full-grid support."""
    expected=set();depths=sorted(set(config.get('eval_depths',[])))
    if not depths:return expected
    schedule=config.get('evaluation_schedule','full_grid')
    if schedule not in ('full_grid','economy_v1'):raise ValueError('unknown evaluation schedule')
    final=config.get('steps',0);steps=set(config.get('checkpoints',[]))|{0,final}
    for variant in config.get('variants',[]):
        for seed in config.get('seeds',[]):
            for step in steps:
                selected=depths if schedule=='full_grid' or step==final else sorted({depths[0],depths[-1]}) if step==0 else depths[:1]
                expected.update((variant,seed,step,depth,'depth','none') for depth in selected)
            if schedule!='economy_v1':continue
            if config.get('extra_evaluations',True):
                for condition in ('heldout_surface','cross_motif','heldout_composition','heldout_wordorder','distractors16','distractors64','ambiguity_delay'):
                    expected.add((variant,seed,final,max(2,config.get('train_depth',2)),condition,'none'))
            if variant=='local':
                for depth in {depths[0],depths[-1]}:
                    for oracle in ('oracle_trace','oracle_minimal'):expected.add((variant,seed,final,depth,oracle,oracle))
                    for intervention in ('event_drop','event_shuffle','event_wrong_value','runtime_off','graph_permuted','graph_drop50','readiness_0.5','readiness_0.95'):
                        expected.add((variant,seed,final,depth,'frozen_intervention',intervention))
    return expected


def coverage_audit(summary, config):
    """Report missing declared cells rather than mislabel partial runs as complete."""
    actual=set()
    if summary['track']=='controlled_execution':
        for cell in summary['aggregates']+summary.get('privileged_oracles',[]):
            actual.update((cell['variant'],seed,cell['step'],cell['depth'],cell['condition'],cell.get('intervention','none')) for seed in cell['seeds'])
        expected=expected_controlled_evaluations(config)
    else:
        for cell in summary['aggregates']:actual.update((cell['variant'],seed,cell['step'],cell['renderer']) for seed in cell['seeds'])
        variants=config.get('arms',['single_pass','recurrent','semantic_supervision','multisurface_consistency'])
        steps=set(config.get('eval_steps',[]))|{config.get('updates',0)}
        expected={(v,seed,step,axis) for v in variants for seed in config.get('seeds',[]) for step in steps for axis in ('english','spanish','symbols')}
    return dict(complete=bool(expected) and expected<=actual,expected_cells=len(expected),observed_cells=len(actual),missing=[list(x) for x in sorted(expected-actual)],
                scope='Declared economy_v1 includes OOD/interventions/oracles; legacy controlled and language coverage check core grid.')


def merge_shards(canonical_config, shard_dirs, output):
    """Merge completed disjoint controlled seed shards without loading model state.

    Resolved shard configs must differ only in seeds. Canonical config can omit
    default-valued fields, but every supplied value must match the resolved config.
    Output is committed by directory rename only after all validation succeeds.
    """
    import shutil
    import tempfile
    output=Path(output)
    if output.exists():raise ValueError('merge output must not exist')
    if not shard_dirs:raise ValueError('no shards')
    expected_seeds=canonical_config.get('seeds',[])
    if not expected_seeds or len(set(expected_seeds))!=len(expected_seeds):raise ValueError('canonical seeds must be unique and nonempty')
    entries=[];all_seeds=set();shared=None;common_manifest=None
    for directory in shard_dirs:
        directory=Path(directory);manifest_path=directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text());config=manifest['config'];seeds=config['seeds']
        if not seeds or len(set(seeds))!=len(seeds) or all_seeds.intersection(seeds):raise ValueError('overlapping/duplicate shard seeds')
        all_seeds.update(seeds)
        comparable={k:v for k,v in config.items() if k!='seeds'}
        varying={'config','config_hash','initializations','checkpoints','elapsed_seconds','started_utc'}
        common={k:v for k,v in manifest.items() if k not in varying}
        if manifest.get('config_hash') and manifest['config_hash']!=digest(config):raise ValueError('invalid shard config hash')
        if manifest.get('source_hash')!=digest(manifest.get('sources',{})):raise ValueError('invalid shard source manifest hash')
        if shared is None:shared=comparable;common_manifest=common
        elif comparable!=shared:raise ValueError('shard configs differ beyond seeds')
        elif common!=common_manifest:raise ValueError('shard source/manifest identities differ')
        for key,value in canonical_config.items():
            if key!='seeds' and (key not in comparable or comparable[key]!=value):raise ValueError('canonical config mismatch: '+key)
        metrics=directory/'metrics.jsonl'
        if not metrics.exists():metrics=directory/'metrics.jsonl.gz'
        if not metrics.is_file():raise ValueError('missing shard metrics')
        entries.append((directory,manifest_path,manifest,metrics))
    if all_seeds!=set(expected_seeds):raise ValueError('missing or unexpected shard seeds')
    merged_config={**shared,'seeds':list(expected_seeds)}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage6-merge-',dir=output.parent) as temporary:
        staging=Path(temporary)/'combined';staging.mkdir();provenance=[];checkpoint_names=set();merged_initializations={};merged_checkpoints={};elapsed_sum=0.;started=[]
        merged_path=staging/'metrics.jsonl.gz'
        with merged_path.open('wb') as merged_raw,gzip.GzipFile(filename='',mode='wb',fileobj=merged_raw,mtime=0,compresslevel=1) as merged_stream:
            for index,(directory,manifest_path,manifest,metrics) in enumerate(entries):
                config=manifest['config'];seeds=set(config['seeds']);variants=set(config['variants'])
                merged_initializations.update(manifest.get('initializations',{}));merged_checkpoints.update(manifest.get('checkpoints',{}))
                elapsed_sum+=manifest.get('elapsed_seconds',0.)
                if manifest.get('started_utc'):started.append(manifest['started_utc'])
                shard_archive=staging/'shards'/f'{index:03d}';shard_archive.mkdir(parents=True)
                shutil.copyfile(manifest_path,shard_archive/'manifest.json')
                (shard_archive/'config.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
                pairs=set();seen=set();training=Counter();grid=set();row_count=0;uncompressed=hashlib.sha256()
                archived_metrics=shard_archive/'metrics.jsonl.gz'
                opener=gzip.open if metrics.suffix=='.gz' else open
                with opener(metrics,'rb') as source,archived_metrics.open('wb') as archive_raw,gzip.GzipFile(filename='',mode='wb',fileobj=archive_raw,mtime=0,compresslevel=1) as archive_stream:
                    for line in source:
                        uncompressed.update(line);archive_stream.write(line)
                        if not line.strip():continue
                        row=json.loads(line);variant=row['variant'];seed=row['seed'];kind=row['kind'];step=row['step']
                        if variant not in variants or seed not in seeds:raise ValueError('row outside declared shard seed/variant')
                        pairs.add((variant,seed));row_count+=1
                        identity=(variant,seed,kind,step,row.get('condition'),row.get('depth'),row.get('intervention'))
                        if identity in seen:raise ValueError('duplicate shard row')
                        seen.add(identity)
                        if kind=='evaluation':
                            if row.get('source_hash')!=manifest['source_hash']:raise ValueError('row source hash differs from shard manifest')
                            grid.add((variant,seed,step,row['depth'],row.get('condition','depth'),row.get('intervention','none')))
                            init=manifest.get('initializations',{}).get(f'{variant}:seed{seed}')
                            if init is not None and row.get('initial_hash')!=init:raise ValueError('row initial state differs from manifest')
                            checkpoint=manifest.get('checkpoints',{}).get(f'{variant}-seed{seed}-step{step}.pt')
                            if checkpoint is not None and row.get('checkpoint_hash')!=checkpoint['state_hash']:raise ValueError('row checkpoint state differs from manifest')
                        elif kind=='training':
                            if not 1<=step<=config['steps']:raise ValueError('training step outside configured range')
                            training[(variant,seed)]+=1
                        else:raise ValueError('unknown shard row kind')
                        merged_stream.write(line if line.endswith(b'\n') else line+b'\n')
                expected_pairs={(v,seed) for v in variants for seed in seeds}
                if pairs!=expected_pairs:raise ValueError('missing variant/seed run')
                if any(training[pair]!=config['steps'] for pair in expected_pairs):raise ValueError('missing training updates')
                steps=set(config.get('checkpoints',[]))|{0,config['steps']}
                expected_grid=expected_controlled_evaluations(config)
                if not expected_grid<=grid:raise ValueError('missing declared evaluation cells')
                checkpoints={}
                for checkpoint in sorted(directory.glob('*.pt')):
                    name=checkpoint.name
                    expected_names={f'{v}-seed{seed}-step{step}.pt' for v,seed in expected_pairs for step in steps}
                    if name not in expected_names or name in checkpoint_names:raise ValueError('checkpoint outside shard identity or duplicate name')
                    checkpoint_names.add(name);wanted=file_hash(checkpoint)
                    producer=manifest.get('checkpoints',{}).get(name)
                    if producer is not None and producer['file_sha256']!=wanted:raise ValueError('checkpoint differs from producer hash')
                    shutil.copyfile(checkpoint,staging/name)
                    if file_hash(staging/name)!=wanted:raise ValueError('checkpoint changed while copying')
                    checkpoints[name]=dict(sha256=wanted,bytes=checkpoint.stat().st_size)
                if not manifest.get('checkpoints',{}).keys()<=checkpoints.keys():raise ValueError('missing producer checkpoint file')
                required_final={f'{v}-seed{seed}-step{config["steps"]}.pt' for v,seed in expected_pairs}
                if config.get('save_checkpoints',False) and not required_final<=checkpoints.keys():raise ValueError('missing final checkpoint')
                provenance.append(dict(shard_index=index,input_directory=str(directory),manifest_sha256=file_hash(manifest_path),
                    config_hash=digest(config),source_hash=manifest['source_hash'],seeds=config['seeds'],variants=config['variants'],
                    metrics_input_sha256=file_hash(metrics),metrics_uncompressed_sha256=uncompressed.hexdigest(),
                    archived_metrics_sha256=file_hash(archived_metrics),metrics_rows=row_count,checkpoints=checkpoints,
                    archived_manifest=f'shards/{index:03d}/manifest.json',archived_metrics=f'shards/{index:03d}/metrics.jsonl.gz'))
        combined={**common_manifest,'config':merged_config,'config_hash':digest(merged_config),
            'initializations':merged_initializations,'checkpoints':merged_checkpoints,'elapsed_seconds':elapsed_sum,
            'started_utc':min(started) if started else None,'combined_from':provenance,'merge':dict(
            elapsed_seconds_semantics='sum_shard_elapsed_seconds_not_parallel_wall',
            kind='disjoint_seed_shards_of_one_study',canonical_input_config_hash=digest(canonical_config),resolved_config_hash=digest(merged_config),
            metrics_sha256=file_hash(merged_path),analysis_source_sha256=file_hash(__file__),
            checkpoint_note='Copied bytes checked against measured shard file hashes; no model state loaded.')}
        (staging/'config.json').write_text(json.dumps(merged_config,indent=2,allow_nan=False)+'\n')
        (staging/'canonical-input.json').write_text(json.dumps(canonical_config,indent=2,allow_nan=False)+'\n')
        (staging/'manifest.json').write_text(json.dumps(combined,indent=2,allow_nan=False)+'\n')
        staging.rename(output)
    return combined


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cells=summary['aggregates'];variants=sorted({c['variant'] for c in cells})
    if not cells:return
    final=max(c['step'] for c in cells)
    metrics=('task_accuracy','exact_semantic_accuracy','trajectory_exact_accuracy') if summary['track']=='controlled_execution' else ('task_choice_accuracy','node_precision','typed_edge_precision','exact_canonical_graph')
    for metric in metrics:
        selected=[c for c in cells if c['step']==final and c.get('intervention','none')=='none']
        labels=sorted({str(c.get('depth',c.get('renderer'))) + ':' + c.get('condition','') for c in selected})
        lookup={(c['variant'],str(c.get('depth',c.get('renderer'))) + ':' + c.get('condition','')):c['metrics'].get(metric,{}).get('mean') for c in selected}
        matrix=[[lookup.get((v,l)) if lookup.get((v,l)) is not None else math.nan for l in labels] for v in variants]
        fig,ax=plt.subplots(figsize=(max(6,len(labels)),max(3,len(variants)*.35)),constrained_layout=True)
        im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto');ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(len(variants)),yticklabels=variants,title=metric)
        ax.tick_params(axis='x',rotation=40);fig.colorbar(im,ax=ax);fig.savefig(output/(metric+'-heatmap.png'),dpi=150);plt.close(fig)
    for metric in sorted(set().union(*(c['metrics'] for c in summary['curves']))):
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for variant in variants:
            points=[c for c in summary['curves'] if c['variant']==variant and c['metrics'].get(metric,{}).get('mean') is not None]
            if points:ax.plot([p['step'] for p in points],[p['metrics'][metric]['mean'] for p in points],label=variant)
        ax.set(xlabel='Optimizer updates',ylabel=metric);ax.legend(fontsize=6);fig.savefig(output/(metric+'-curve.png'),dpi=150);plt.close(fig)
    if summary['track']=='controlled_execution':
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for c in cells:
            if c['step']!=final or c['condition']!='depth':continue
            label=f"{c['variant']} d{c['depth']}"
            bins=[b for b in c['calibration']['bins'] if b['n']]
            axes[0].plot([b['mean_probability'] for b in bins],[b['mean_target'] for b in bins],label=label)
            points=[p for p in c['calibration']['risk_coverage'] if p['risk'] is not None]
            axes[1].plot([p['coverage'] for p in points],[p['risk'] for p in points],label=label)
        axes[0].plot([0,1],[0,1],'k--');axes[0].set(xlabel='Predicted readiness',ylabel='Posterior readiness')
        axes[1].set(xlabel='Proposal coverage',ylabel='Posterior unready risk');axes[1].legend(fontsize=5)
        fig.savefig(output/'readiness-risk-calibration.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--merge-shards',nargs='+',type=Path,help='Merge completed controlled seed shard directories; input is canonical config JSON')
    parser.add_argument('--track',choices=['controlled','language'],default='controlled');parser.add_argument('--source-root',type=Path);parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.merge_shards:
        merge_shards(json.loads(args.input.read_text()),args.merge_shards,args.output)
        return
    root=args.input if args.input.is_dir() else args.input.parent
    path=(root/('metrics.jsonl' if args.track=='controlled' else 'curves.jsonl')) if args.input.is_dir() else args.input
    if not path.exists() and path.suffix=='.jsonl':path=path.with_suffix('.jsonl.gz')
    rows=read_rows(path)
    if args.track=='controlled':summary=summarize_controlled(rows);manifest_path=root/'manifest.json'
    else:
        loss_path=root/'losses.jsonl';summary=summarize_language(rows,read_rows(loss_path) if loss_path.exists() else ());manifest_path=root/'data-audit.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    audit=dict(coverage=coverage_audit(summary,manifest.get('config',{})),manifest_present=bool(manifest),config_hash=digest(manifest.get('config',{})),manifest_sha256=file_hash(manifest_path) if manifest_path.exists() else None)
    if args.track=='controlled':
        audit['manifest_source_hash_matches']=manifest.get('source_hash')==digest(manifest.get('sources',{}))
        audit['row_source_hash_matches']=summary['provenance']['source_hashes']==[manifest.get('source_hash')]
        if manifest.get('combined_from'):
            audit['combined_from']=manifest['combined_from']
            audit['merged_metrics_hash_matches']=manifest.get('merge',{}).get('metrics_sha256')==file_hash(path)
    else:
        run_manifest_path=root/'manifest.json'
        run_manifest=json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {}
        audit.update(audit_language(manifest,run_manifest,root))
        audit['row_data_hash_matches']=summary['provenance']['data_hashes']==[manifest.get('data_hash')] if summary['provenance']['data_hashes'] else None
        audit['row_config_hash_matches']=summary['provenance']['config_hashes']==[manifest.get('config_hash')] if summary['provenance']['config_hashes'] else None
        summary['caveats'].extend(manifest.get('limitations',[]))
    if args.source_root:
        source_root=args.source_root if args.track=='controlled' else args.source_root/'src'/'topoformer'
        audit['sources']=verify_files(manifest.get('sources',manifest.get('source_hashes',{})),source_root)
    audit['checkpoints']={p.name:dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(root.glob('*.pt'))}
    audit['checkpoint_note']='Recorded hashes identify artifacts; independent producer checkpoint digests are checked when the run manifest supplies them.'
    summary.update(schema_version=6,audit=audit,artifact=dict(path=str(path),sha256=file_hash(path),bytes=path.stat().st_size),analysis_sha256=file_hash(__file__))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not args.no_plots:plots(summary,args.output/'figures')


if __name__=='__main__':main()
