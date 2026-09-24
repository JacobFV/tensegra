import torch,json,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);root=a.root;x=[torch.load(root/f's13-{a}-main/model-u28672.pt',map_location='cpu',weights_only=True) for a in ('english','mixed')]
for k in ('generator','schedule','order','position','visits','added_visits','renderer_phases'):
 if torch.is_tensor(x[0][k]):assert torch.equal(x[0][k],x[1][k])
 else:assert x[0][k]==x[1][k]
out=dict(final_negative_pair_rng_equal=True,final_schedule_order_position_visits_phases_equal=True,cpu_audit_wall_seconds=time.monotonic()-t);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
