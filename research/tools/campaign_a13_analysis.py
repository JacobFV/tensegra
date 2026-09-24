"""Paired A13 raw-array analysis; no model inference or policy selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


POLICIES=('unchanged','shared_soft','shared_hard','oracle_common')


def analyze(run):
    source=run/'results'
    data={policy:np.load(source/f'{policy}.npz') for policy in POLICIES}
    cfg=json.loads((source/'config.json').read_text())
    report=dict(input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.iterdir()) if p.is_file()},
                policies=list(POLICIES),oracle_note='oracle_common is privileged routing, not learned competence',
                conditions=[])
    for ci,condition in enumerate(cfg['conditions']):
        base=data['unchanged'];base_task=base[f'c{ci}_task'].astype(bool)
        row=dict(condition=condition,examples=len(base_task),policy={})
        for policy in POLICIES:
            a=data[policy];task=a[f'c{ci}_task'].astype(bool)
            suffix=a[f'c{ci}_suffix_value_trajectory'].astype(bool)
            d=lambda name:a[f'c{ci}_diagnostic_{name}']
            mean_steps=d('query_mean_route_correct').astype(bool)
            route=mean_steps[:,:,0].all(1)
            joint=dict(route_correct_task_correct=int((route&task).sum()),
                       route_correct_task_wrong=int((route&~task).sum()),
                       route_wrong_task_correct=int((~route&task).sum()),
                       route_wrong_task_wrong=int((~route&~task).sum()))
            details=dict(task_correct=int(task.sum()),complete_suffix=int(suffix.sum()),
                         fixed=int((~base_task&task).sum()),broken=int((base_task&~task).sum()),
                         queried_mean_route_fullpath_correct=int(route.sum()),
                         queried_mean_route_denominator=len(route),task_route_joint_counts=joint,
                         task_correct_given_queried_mean_route_correct=float((route&task).sum()/route.sum()) if route.any() else None,
                         suffix_correct_given_queried_mean_route_correct=float((route&suffix).sum()/route.sum()) if route.any() else None,
                         query_mean_trace_equals_existing_path_metric=bool(np.array_equal(route,a[f'c{ci}_exact_pointer_path'].astype(bool))),
                         mean_trace_is_repeated_across_heads=bool(np.equal(mean_steps,mean_steps[:,:,:1]).all()),
                         all_node_used_routes_equal_unchanged=bool(np.array_equal(a[f'c{ci}_route'],base[f'c{ci}_route'])),
                         head_paths={},query_step_means={})
            for name in ['record','original_destination','used_destination']:
                correct=d(f'query_{name}_head_correct').astype(bool)
                head_path=correct.all(1)
                route_head=np.broadcast_to(route[:,None],head_path.shape)
                details['head_paths'][name]=dict(correct_head_paths=int(head_path.sum()),head_path_denominator=int(head_path.size),
                    correct_per_head=head_path.sum(0).tolist(),event_denominator=head_path.shape[0],
                    events_all_heads_fullpath_correct=int(head_path.all(1).sum()),
                    headpath_correct_and_meanpath_correct=int((head_path&route_head).sum()),
                    headpath_denominator_given_meanpath_correct=int(route_head.sum()))
                details['query_step_means'][f'{name}_head_correct']=correct.mean((0,2)).tolist()
            for name in ['record','original_destination','used_destination']:
                agreement=d(f'query_{name}_heads_agree').astype(bool)[:,:,0]
                details['query_step_means'][f'{name}_heads_agree']=agreement.mean(0).tolist()
                details['head_paths'][name]['events_unanimous_at_every_query_step']=int(agreement.all(1).sum())
            details['query_step_means']['mean_route_correct']=mean_steps[:,:,0].mean(0).tolist()
            mse=d('payload_mse_to_destination_argmax').astype(np.float64)
            energy=d('payload_argmax_energy').astype(np.float64)
            err=d('payload_mse_correct_destination_sum').astype(np.float64)
            count=d('correct_destination_count').astype(np.float64)
            details['all_node_local_payload_metrics']=dict(relative_mse=float(mse.sum()/energy.sum()) if energy.sum() else None,
                conditional_mse_given_correct_used_destination=float(err.sum()/count.sum()) if count.sum() else None,
                zero_energy=bool(energy.sum()==0),zero_correct_support=bool(count.sum()==0))
            row['policy'][policy]=details
        report['conditions'].append(row)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(analyze(args.run),indent=2)+'\n');print(args.output)
