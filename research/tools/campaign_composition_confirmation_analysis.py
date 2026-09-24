"""CPU-only C04 raw-outcome aggregation; imports no model or training code."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

ARMS = ('n1_static', 'n1_roles', 'n2_rekey')
DELAYS = (0, 1, 2, 4, 8, 16)
VIEWS = ('clean', 'reversed')


def read_json(path):
    return json.loads(path.read_text())


def raw(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def array(value):
    return value.detach().cpu().numpy()


def counts(prediction, target):
    return {'correct':int((prediction == target).sum()), 'attempted':len(target),
            'refused':int((prediction < 0).sum())}


def causal_counts(clean, drop, query, copy, original, controls):
    n = len(original)
    if any(len(x) != n for x in (clean, drop, query, copy)):
        raise ValueError('causal attempted denominator mismatch')
    rates = [float((x == original).mean()) for x in (clean, drop, query, copy)]
    gain = (rates[0]-rates[2])/(rates[3]-rates[2]) if rates[3] > rates[2] else None
    changed = {}
    for kind, (prediction, supplied) in controls.items():
        valid = supplied >= 0
        mask = valid & (supplied != original)
        unchanged = valid & (supplied == original)
        changed[kind] = dict(attempted=n, supplied_absent=int((~valid).sum()),
            prediction_refused=int((prediction < 0).sum()),
            original_correct=int((prediction == original).sum()),
            supplied_correct=int(((prediction == supplied) & valid).sum()),
            supplied_present=int(valid.sum()), changed_support=int(mask.sum()),
            changed_supplied_correct=int(((prediction == supplied) & mask).sum()),
            changed_original_correct=int(((prediction == original) & mask).sum()),
            unchanged_support=int(unchanged.sum()),
            unchanged_errors=int(((prediction != original) & unchanged).sum()))
    passed = (rates[0]-rates[1] >= .15 and gain is not None and gain >= .8
        and set(changed) == {'wrong','swap'} and all(
            x['changed_support'] >= 256 and x['changed_supplied_correct']/x['changed_support'] >= .9
            for x in changed.values()))
    return dict(attempted=n, clean=counts(clean,original), drop=counts(drop,original),
        query=counts(query,original), supplied_copy=counts(copy,original),
        clean_minus_drop=rates[0]-rates[1], normalized_gain=gain,
        changed_controls=changed, passed=passed)


def paired(left, right):
    return dict(left_only=int((left & ~right).sum()), right_only=int((right & ~left).sum()),
        both_correct=int((left & right).sum()), both_wrong=int((~left & ~right).sum()),
        attempted=len(left), difference=float(left.mean()-right.mean()))


def bootstrap(differences, repeats=2000, seed=6042026):
    """Fixed-lineage conditional event bootstrap; common draws for every cell."""
    import numpy as np
    sample_means = np.zeros(repeats)
    means = []
    for replicate, values in sorted(differences.items()):
        # Reset per cell to reuse draws across arms/views/delays in each lineage.
        rng = np.random.default_rng(seed + replicate)
        means.append(float(values.mean()))
        for start in range(0, repeats, 100):
            indices = rng.integers(0, len(values), (min(100,repeats-start), len(values)))
            sample_means[start:start+len(indices)] += values[indices].mean(axis=1)/len(differences)
    return dict(equal_lineage_mean=float(np.mean(means)),
        event_bootstrap_percentile95=np.quantile(sample_means,[.025,.975]).tolist(),
        seed_means=means, seed_min=min(means), seed_max=max(means),
        seed_sample_sd=float(np.std(means,ddof=1)) if len(means)>1 else None,
        repeats=repeats, bootstrap_seed=seed,
        scope='independent event resampling within each fixed lineage; identical draws across cells; no seed population confidence interval')


def required_files(base):
    paths = [Path(str(base)+'-'+phase)/'summary.json' for phase in ('hybrid',*ARMS,'timing')]
    for view in VIEWS:
        paths.extend(Path(str(base)+'-hybrid')/f'{view}-{d}.pt' for d in (2,8))
        paths.extend(Path(str(base)+'-hybrid')/f'{view}-causal-{kind}.pt' for kind in ('correct','drop','wrong','swap'))
    for arm in ARMS:
        root = Path(str(base)+'-'+arm)
        paths.append(root/'numeric-overlap.json.gz')
        for endpoint in ('endpoint','selected'):
            paths.extend(root/f'{endpoint}-validation{suffix}.pt' for suffix in ('','-reversed'))
    return paths


def validate_main_config(config, replicate, primary_endpoint_step=None):
    expected_base = 560000000 + replicate*1000000
    if config['seed'] != 1601+replicate or config['replicate'] != replicate or config['updates'] != 4000:
        raise ValueError('not the predeclared main model lineage')
    for split,offset,count in (('train',1,16384),('calibration',2,2048),('validation',3,4096)):
        if config['data'][split] != {'seed':expected_base+offset,'count':count}:
            raise ValueError('not the predeclared main population')
    if primary_endpoint_step is not None and primary_endpoint_step != 4000:
        raise ValueError('primary endpoint must be fixed4000')


def analyze_lineage(base, replicate, main=True):
    import numpy as np
    hybrid_root = Path(str(base)+'-hybrid')
    summary = read_json(hybrid_root/'summary.json')
    if main: validate_main_config(summary['config'],replicate)
    n = summary['config']['data']['validation']['count']
    result = dict(status='complete', replicate=replicate, model_seed=summary['config']['seed'],
        event_seed=summary['config']['data']['validation']['seed'], hybrid_cells=[], causal=[],
        neural={}, pairings={}, public_controls=summary['controls'],
        source_declared_hybrid_gates=summary['accuracy_gates'])
    correct = {}; labels_by_view = {}; overlap = {}
    for view in VIEWS:
        for distractors in (2,8):
            cell = raw(hybrid_root/f'{view}-{distractors}.pt')
            truth = array(cell['original']); full = array(cell['full_proposal']).astype(bool)
            if len(truth) != n: raise ValueError('hybrid count mismatch')
            if view in labels_by_view and not np.array_equal(truth,labels_by_view[view]):
                raise ValueError('distractor populations are not paired')
            labels_by_view[view] = truth
            for delay in DELAYS:
                pred = array(cell['result']['cells'][delay]['predictions'])
                joint = int(((pred == truth) & full).sum())
                result['hybrid_cells'].append(dict(view=view,distractors=distractors,delay=delay,
                    answer=counts(pred,truth), joint_correct=joint, ordered_lowering_correct=int(full.sum()),
                    correct_lowering_answer_correct=joint, correct_lowering_support=int(full.sum()),
                    required_joint_correct=(98*n+99)//100, joint_pass=joint >= (98*n+99)//100))
                correct[f'workspace/{view}/d{distractors}/t{delay}'] = pred == truth
            copied = array(cell['exact_copy'])
            correct[f'supplied_copy/{view}/d{distractors}'] = copied == truth
            result['hybrid_cells'].append(dict(path='supplied_copy',view=view,distractors=distractors,
                answer=counts(copied,truth), answer_pass=int((copied==truth).sum()) >= (98*n+99)//100))
        controls = {kind:raw(hybrid_root/f'{view}-causal-{kind}.pt') for kind in ('correct','drop','wrong','swap')}
        original = array(controls['correct']['original'])
        if len(original) != min(summary['config']['causal_size'],n):
            raise ValueError('causal configured count mismatch')
        if any(not np.array_equal(array(c['original']),original) for c in controls.values()):
            raise ValueError('causal original event ordering mismatch')
        query = array(controls['correct']['query_only']); copy = array(controls['correct']['exact_copy'])
        for path, delays in (('workspace',(0,1,16)),('supplied_copy',(None,))):
            for delay in delays:
                def pred(kind):
                    c = controls[kind]
                    return array(c['exact_copy'] if path=='supplied_copy' else c['result']['cells'][delay]['predictions'])
                detail = causal_counts(pred('correct'),pred('drop'),query,copy,original,
                    {kind:(pred(kind),array(controls[kind]['result']['supplied'])) for kind in ('wrong','swap')})
                result['causal'].append(dict(path=path,view=view,delay=delay,**detail))
    neural_summaries = {}
    for arm in ARMS:
        root = Path(str(base)+'-'+arm); report = read_json(root/'summary.json'); neural_summaries[arm]=report
        if main:
            validate_main_config(report['config'],replicate,report['primary_endpoint_step'])
            if report['primary_endpoint_step'] != 4000:
                raise ValueError('missing fixed4000 endpoint')
        with gzip.open(root/'numeric-overlap.json.gz','rt') as stream: overlap[arm]=json.load(stream)
        result['neural'][arm] = dict(selected_step=report['selected_step'],primary_endpoint_step=report['primary_endpoint_step'],
            summary_results=report['results'],raw_counts={})
        for endpoint in ('endpoint','selected'):
            for view in VIEWS:
                cell=raw(root/(f'{endpoint}-validation'+ ('-reversed.pt' if view=='reversed' else '.pt')))
                truth=array(cell['labels']['task']); pred=array(cell['logits']['answer'].argmax(-1))
                if not np.array_equal(truth,labels_by_view[view]): raise ValueError('arm target ordering mismatch')
                key=f'{arm}/{endpoint}/{view}'
                correct[key]=pred==truth
                result['neural'][arm]['raw_counts'][key]=counts(pred,truth)
    result['pairing_checks'] = dict(
        n1_initial_equal=neural_summaries['n1_static']['initial_state_sha256']==neural_summaries['n1_roles']['initial_state_sha256'],
        all_sample_streams_equal=len({d['sample_index_stream_sha256'] for d in neural_summaries.values()})==1)
    if not all(result['pairing_checks'].values()): raise ValueError('paired training contract mismatch')
    differences={}
    for view in VIEWS:
        # Neural public numeric evidence is identical under2/8 memory distractors.
        # Keep all memory cells; never treat delays or distractors as extra events.
        for distractors in (2,8):
            competitors=[f'supplied_copy/{view}/d{distractors}']+[f'{a}/{e}/{view}' for a in ARMS for e in ('endpoint','selected')]
            copy_key=competitors[0]
            for right in competitors[1:]:
                key=copy_key+' minus '+right
                result['pairings'][key]=paired(correct[copy_key],correct[right])
                differences[key]=correct[copy_key].astype(float)-correct[right].astype(float)
            for delay in DELAYS:
                left=f'workspace/{view}/d{distractors}/t{delay}'
                for right in competitors:
                    key=left+' minus '+right
                    result['pairings'][key]=paired(correct[left],correct[right])
                    differences[key]=correct[left].astype(float)-correct[right].astype(float)
        for left_arm,right_arm in (('n1_static','n1_roles'),('n1_static','n2_rekey'),('n1_roles','n2_rekey')):
            key=f'{left_arm}/endpoint/{view} minus {right_arm}/endpoint/{view}'
            left=correct[f'{left_arm}/endpoint/{view}'];right=correct[f'{right_arm}/endpoint/{view}']
            result['pairings'][key]=paired(left,right);differences[key]=left.astype(float)-right.astype(float)
    result['gates'] = dict(workspace_joint=all(c['joint_pass'] for c in result['hybrid_cells'] if 'joint_pass' in c),
        supplied_copy_answer=all(c['answer_pass'] for c in result['hybrid_cells'] if 'answer_pass' in c),
        causal_workspace=all(c['passed'] for c in result['causal'] if c['path']=='workspace'),
        causal_supplied_copy=all(c['passed'] for c in result['causal'] if c['path']=='supplied_copy'),
        neural_fixed_endpoint={a:all(v['correct'] >= (98*v['attempted']+99)//100 for k,v in result['neural'][a]['raw_counts'].items() if '/endpoint/' in k) for a in ARMS})
    result['timing']=read_json(Path(str(base)+'-timing')/'summary.json')
    result['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in required_files(base)}
    return result,differences,overlap


def overlap_report(audits):
    result={'per_lineage':{},'cross_lineage':[], 'inherited_overlap':'unmeasured; not zero',
        'scope':'typed ordered finite-domain operation/operand signatures, optionally exact query; no OOD claim'}
    for replicate,arms in audits.items():
        result['per_lineage'][replicate]={}
        for arm,record in arms.items():
            result['per_lineage'][replicate][arm]={}
            for kind,audit in record['audits'].items():
                populations={}
                for name,cell in audit['populations'].items():
                    populations[name]={k:v for k,v in cell.items() if k!='signature_counts'}
                    populations[name]['visited_training_event_overlap_fraction']=cell['training_overlap_events']/cell['events']
                result['per_lineage'][replicate][arm][kind]=dict(
                    actual_training_presentations=audit['actual_training_presentations'],
                    actual_training_distinct_signatures=audit['actual_training_distinct_signatures'],populations=populations)
                for other,other_arms in audits.items():
                    if other==replicate:continue
                    train=set(other_arms[arm]['audits'][kind]['actual_training_signature_counts'])
                    for view in VIEWS:
                        test=audit['populations']['validation/'+view]['signature_counts'];shared=set(test)&train
                        events=sum(test.values());overlap=sum(test[k] for k in shared)
                        result['cross_lineage'].append(dict(test_lineage=replicate,train_lineage=other,arm=arm,
                            signature=kind,view=view,test_events=events,test_distinct=len(test),
                            shared_distinct=len(shared),overlap_events=overlap,overlap_fraction=overlap/events))
    return result


def run(directory, output, repeats):
    if repeats <= 0: raise ValueError('positive bootstrap repeat count required')
    started=time.monotonic();reports={};differences={};overlaps={}
    for replicate in range(3):
        base=directory/f'c04-confirmation-{replicate}'
        missing=[str(p) for p in required_files(base) if not p.exists()]
        if missing:
            reports[replicate]=dict(status='pending',missing=missing,event_seed=560000003+replicate*1000000)
            batch_path=Path(str(base)+'.batch.json')
            if batch_path.exists(): reports[replicate]['process_receipt']=read_json(batch_path)
            continue
        reports[replicate],differences[replicate],overlaps[replicate]=analyze_lineage(base,replicate)
        if reports[replicate]['event_seed'] != 560000003+replicate*1000000 or reports[replicate]['model_seed'] != 1601+replicate:
            raise ValueError('not the predeclared confirmation lineage')
    complete=len(differences)==3
    result=dict(status='complete' if complete else 'pending',lineages=reports,
        aggregate_gates='pending' if not complete else {str(r):d['gates'] for r,d in reports.items()},
        comparisons={},semantic_overlap=overlap_report(overlaps),
        policy='fixed4000 primary; CLEAN-calibration selected secondary retained; all3 independent event populations; no averaging away gate failures')
    if complete:
        for key in differences[0]:
            result['comparisons'][key]=bootstrap({r:d[key] for r,d in differences.items()},repeats=repeats)
    else:
        result['aggregate_uncertainty']='pending until all three fixed lineages are available'
    result['cpu_seconds']=time.monotonic()-started
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args();run(args.directory,args.output,args.bootstrap_repeats)
