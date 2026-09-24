"""Prospective fixed three-seed paired confirmation analysis, CPU arrays only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS=(1401,1402,1403)
POLICIES=('unchanged','both_hard','shared_soft','shared_hard','oracle_common')
CONDITIONS=[dict(nodes=32,depth=4,groups=4),dict(nodes=64,depth=8,groups=4),dict(nodes=128,depth=32,groups=8)]
PUBLIC_KEYS={f'c{ci}_{name}' for ci in range(3) for name in ['gold','successor','start','relation','values','input_keys','input_attributes','input_instructions','input_adjacency']}


def require(condition,message):
    if not condition:raise ValueError(message)


def hash_map(value,keys):
    return set(value)==set(keys) and all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef') for v in value.values())


def validate_run_contract(cfg,manifest,arrays):
    require(cfg['seed'] in SEEDS and cfg['steps']==6000 and cfg['examples']==1024 and cfg['batch']==16,'Fixed seed/endpoint/batch/population mismatch')
    require(cfg['conditions']==CONDITIONS and cfg['data_seed']==271000000 and cfg['order_seed']==272000000 and cfg['monitor_steps']==[1000,3000,6000],'Frozen condition/population/monitor mismatch')
    require(manifest['seed']==cfg['seed'] and manifest['steps']==6000 and manifest['presentations']==96000,'Manifest training exposure mismatch')
    frozen=manifest['confirmation']
    require(frozen['initial_tensor_sha256']==frozen['final_tensor_sha256']==manifest['final_tensor_sha256'],'Inference tensor mutation/binding mismatch')
    require(frozen['policies']==list(POLICIES) and set(arrays)==set(POLICIES),'Frozen policy set/order mismatch')
    require(hash_map(frozen['public_sha256'],PUBLIC_KEYS),'Missing/invalid full public hash bindings')
    require(hash_map(frozen['record_order_sha256'],{'c0','c1','c2'}),'Missing/invalid record-order hash bindings')
    for policy,a in arrays.items():
        for ci,c in enumerate(CONDITIONS):
            for name in ['task','suffix_value_trajectory','exact_pointer_path','all_node']:
                value=a[f'c{ci}_{name}'];require(value.shape==(1024,),f'{policy}: truncated/malformed event metric {name}')
                require(np.isfinite(value).all() and ((value>=0)&(value<=1)).all(),f'{policy}: invalid event metric {name}')
                if name!='all_node':require(((value==0)|(value==1)).all(),f'{policy}: nonbinary event label {name}')
            for name in ['pred','route']:require(a[f'c{ci}_{name}'].shape==(1024,c['depth'],c['nodes']),f'{policy}: malformed {name} shape')
            required_query=['query_mean_route_correct']+[f'query_{kind}_{metric}' for kind in ['record','original_destination','used_destination'] for metric in ['head_correct','heads_agree']]
            for name in required_query:
                value=a[f'c{ci}_diagnostic_{name}'];require(value.shape==(1024,c['depth'],8),f'{policy}: malformed query/head shape {name}')
                require(((value==0)|(value==1)).all(),f'{policy}: nonbinary query/head label {name}')
            for name in a:
                if name.startswith(f'c{ci}_diagnostic_'):require(a[name].shape==(1024,c['depth'],8),f'{policy}: malformed diagnostic shape {name}')


def validate_references(cfg,manifest,public):
    expected=json.loads((Path(__file__).resolve().parents[2]/'configs/campaign-a14-engineering-references.json').read_text())
    require(cfg==expected,'Frozen reference configuration mismatch')
    require(manifest['optimizer_updates']==0 and len(manifest['references'])==9,'Reference count/update mismatch')
    identities={(r['seed'],r['mode']):r for r in expected['checkpoints']};seen=set()
    for ref in manifest['references']:
        identity=(ref['seed'],ref['mode']);require(identity in identities and identity not in seen,'Missing/duplicate/unknown reference identity');seen.add(identity)
        original=identities[identity]
        require(ref['sha256']==original['sha256'] and ref['path']==original['path'],'Reference checkpoint provenance mismatch')
        require(ref['policies']==['a06_successful'] and ref['reference_mode']==ref['mode'],'Reference policy mismatch')
        require(ref['initial_tensor_sha256']==ref['final_tensor_sha256'],'Reference inference mutation')
        require(hash_map(ref['public_sha256'],PUBLIC_KEYS) and ref['public_sha256']==public,'Reference public graph mismatch')
    require(seen==set(identities),'Incomplete reference identities')


def joint_statistics(original,shared,route,replicates=10000,seed=291000000):
    """Inputs are seed×paired-event booleans, preserving shared graph support."""
    original=np.asarray(original,dtype=bool);shared=np.asarray(shared,dtype=bool);route=np.asarray(route,dtype=bool)
    if original.shape!=shared.shape or shared.shape!=route.shape or original.ndim!=2:raise ValueError('Paired shapes differ')
    gain=shared.astype(np.int8)-original.astype(np.int8)
    per_seed_gain=gain.mean(1);support=route.sum(1);intersection=(shared&route).sum(1)
    conditional=[float(x/n) if n else None for x,n in zip(intersection,support)]
    rng=np.random.default_rng(seed);n=gain.shape[1];gain_reps=[];conditional_reps=[];undefined=0
    graph_gain=gain.mean(0);graph_num=(shared&route).sum(0);graph_den=route.sum(0)
    for offset in range(0,replicates,256):
        indices=rng.integers(0,n,size=(min(256,replicates-offset),n))
        gain_reps.extend(graph_gain[indices].mean(1).tolist())
        numerator=graph_num[indices].sum(1);denominator=graph_den[indices].sum(1)
        undefined+=int((denominator==0).sum())
        conditional_reps.extend((numerator[denominator>0]/denominator[denominator>0]).tolist())
    gain_ci=np.quantile(gain_reps,[.025,.975]).tolist()
    return dict(per_seed_gain=per_seed_gain.tolist(),mean_gain=float(gain.mean()),gain_percentile95=gain_ci,
        replicated_effect_pass=bool((per_seed_gain>0).all() and gain_ci[0]>0),
        per_seed_route_support=support.tolist(),per_seed_task_and_route_correct=intersection.tolist(),
        per_seed_conditional_task=conditional,pooled_conditional_task=float(intersection.sum()/support.sum()) if support.sum() else None,
        conditional_percentile95=np.quantile(conditional_reps,[.025,.975]).tolist() if undefined==0 else None,
        undefined_conditional_bootstrap_replicates=undefined,
        conditional_mechanistic_pass=bool(all(value is not None and value>=.98 for value in conditional)),
        bootstrap=dict(replicates=replicates,seed=seed,unit='same graph indices shared across all seeds/policies; seeds not independently resampled'))


def analyze(runs,references=None):
    payload={};public=None;order=None;provenance={}
    for run in runs:
        cfg=json.loads((run/'config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
        seed=cfg['seed']
        if seed in payload:raise ValueError('Duplicate seed')
        arrays={policy:np.load(run/'confirmation'/f'{policy}.npz') for policy in POLICIES}
        validate_run_contract(cfg,manifest,arrays)
        if public is None:public=manifest['confirmation']['public_sha256'];order=manifest['confirmation']['record_order_sha256']
        if public!=manifest['confirmation']['public_sha256'] or order!=manifest['confirmation']['record_order_sha256']:raise ValueError('Public graphs/record order differ across seeds')
        payload[seed]=arrays
        provenance[str(seed)]={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz')}
    if set(payload)!=set(SEEDS):raise ValueError('All three fixed seeds required')
    result=dict(seeds=list(SEEDS),input_sha256=provenance,public_hashes_match=True,record_order_hashes_match=True,conditions=[])
    gates={policy:[] for policy in POLICIES[:-1]}
    for ci,(name,threshold) in enumerate([('iid',.98),('moderate',.95),('joint',.98)]):
        rows={}
        for seed in SEEDS:
            base=payload[seed]['unchanged'][f'c{ci}_task'].astype(bool);rows[str(seed)]={}
            for policy in POLICIES:
                a=payload[seed][policy];task=a[f'c{ci}_task'].astype(bool)
                route=a[f'c{ci}_diagnostic_query_mean_route_correct'][:,:,0].astype(bool).all(1)
                if not np.array_equal(route,a[f'c{ci}_exact_pointer_path'].astype(bool)):raise ValueError('Queried route trace/path metric mismatch')
                row=dict(task_correct=int(task.sum()),events=len(task),complete_all_node_suffix=int(a[f'c{ci}_suffix_value_trajectory'].sum()),
                    all_node_value_accuracy=float(a[f'c{ci}_all_node'].mean()),
                    fixed=int((task&~base).sum()),broken=int((~task&base).sum()),route_correct=int(route.sum()),
                    joint_counts=dict(route_correct_task_correct=int((route&task).sum()),route_correct_task_wrong=int((route&~task).sum()),route_wrong_task_correct=int((~route&task).sum()),route_wrong_task_wrong=int((~route&~task).sum())),
                    task_given_correct_queried_route=float((route&task).sum()/route.sum()) if route.any() else None,
                    all_node_routes_equal_original=bool(np.array_equal(a[f'c{ci}_route'],payload[seed]['unchanged'][f'c{ci}_route'])))
                row['queried_head_fullpaths']={}
                for head_kind in ['record','original_destination','used_destination']:
                    head=a[f'c{ci}_diagnostic_query_{head_kind}_head_correct'].astype(bool).all(1)
                    row['queried_head_fullpaths'][head_kind]=dict(correct=int(head.sum()),head_event_denominator=int(head.size),
                        correct_per_head=head.sum(0).tolist(),events_all_heads_correct=int(head.all(1).sum()),event_denominator=len(head))
                rows[str(seed)][policy]=row
                if policy!='oracle_common':gates[policy].append(bool(task.mean()>=threshold))
        result['conditions'].append(dict(shape=name,threshold=threshold,seed_rows=rows))
    originals=np.stack([payload[s]['unchanged']['c2_task'] for s in SEEDS])
    shared=np.stack([payload[s]['shared_hard']['c2_task'] for s in SEEDS])
    route=np.stack([payload[s]['shared_hard']['c2_diagnostic_query_mean_route_correct'][:,:,0].all(1) for s in SEEDS])
    result['joint_primary']=joint_statistics(originals,shared,route)
    result['competence_all_three_core_cells_all_seeds']={policy:all(values) for policy,values in gates.items()}
    result['gate_labels']=dict(unchanged='acquired-soft only',other_nonoracle='supplied inference-policy competence only',oracle_common='privileged routing reference; no competence promotion')
    result['engineering_references']='pending' if references is None else {}
    if references is not None:
        m=json.loads((references/'manifest.json').read_text())
        validate_references(json.loads((references/'config.json').read_text()),m,public)
        for ref in m['references']:
            if ref['public_sha256']!=public:raise ValueError('Reference public graph mismatch')
            tag=f"{ref['seed']}-{ref['mode']}"
            result['engineering_references'][tag]=json.loads((references/tag/'a06_successful.json').read_text())
        result['engineering_reference_note']='Different A06 training histories and supplied-neighborhood interfaces; not a paired architecture-training comparison'
        result['engineering_reference_input_sha256']={str(p.relative_to(references)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(references.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz')}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',type=Path,nargs=3);parser.add_argument('--references',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(analyze(args.runs,args.references),indent=2)+'\n');print(args.output)
