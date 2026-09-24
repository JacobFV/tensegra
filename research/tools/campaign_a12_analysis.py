"""Summarize paired A12 archived arrays; no inference or model selection."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'results/campaign-01/attention/a12-main/results'
policies=['unchanged','record_hard','destination_hard','both_hard']
data={policy:np.load(root/f'{policy}.npz') for policy in policies}
result={}
for ci,label in enumerate(['iid','moderate','joint']):
    base=data['unchanged'][f'c{ci}_task'].astype(bool)
    result[label]={}
    for policy in policies:
        arrays=data[policy];task=arrays[f'c{ci}_task'].astype(bool)
        diag=lambda name:arrays[f'c{ci}_diagnostic_{name}'].astype(np.float64)
        mse=diag('payload_mse_to_destination_argmax');energy=diag('payload_argmax_energy')
        error=diag('payload_mse_correct_destination_sum');count=diag('correct_destination_count')
        result[label][policy]=dict(
            correct=int(task.sum()),fixed=int((~base&task).sum()),broken=int((base&~task).sum()),
            complete_suffix=int(arrays[f'c{ci}_suffix_value_trajectory'].sum()),
            relative_payload_mse=float(mse.sum()/energy.sum()) if energy.sum() else None,
            payload_mse_given_correct_destination=float(error.sum()/count.sum()) if count.sum() else None,
            record_argmax_accuracy_per_head=diag('record_argmax_correct').mean((0,1)).tolist(),
            destination_argmax_accuracy_per_head=diag('destination_argmax_correct').mean((0,1)).tolist(),
            reverse_step_metrics={name:diag(name).mean((0,2)).tolist() for name in [
                'record_soft_target_mass','destination_used_target_mass','record_argmax_correct',
                'destination_argmax_correct','payload_mse_to_destination_argmax','payload_argmax_energy',
                'payload_mse_to_oracle','record_all_heads_same_destination','destination_all_heads_same_node']},
        )
out=root.parent/'paired-analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(out)
