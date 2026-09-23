"""Independently link five frozen failures to prior raw graph records (stdlib)."""
import argparse,gzip,json,math
from pathlib import Path

def audit(root):
 new=json.loads((root/'research/results/stage10/semantic-edge/frozen-seed10.json').read_text())
 with gzip.open(root/'research/results/stage9/semantic-contracts/calibrated-confirmation.json.gz','rt')as f:old=json.load(f)
 run=next(r for r in old['runs']if r['seed']==10 and r['head']=='additive'and r['objective']=='edge_conditional')
 errors=[];expected=set()
 for row in run['calibrated']['rows']:
  p={tuple(e)for e in row['predicted_edges']};g={tuple(e)for e in row['gold_edges']}
  expected.update((row['seed'],i,j,r,False)for i,j,r in p-g)
  expected.update((row['seed'],i,j,r,True)for i,j,r in g-p)
 found=set()
 for failure in new['failures']:
  key=(failure['graph_seed'],failure['source_index'],failure['target_index'],3,failure['gold']);found.add(key)
  if failure['relation']!='argument':errors.append(['unexpected_relation'])
  row=next(r for r in run['calibrated']['rows']if r['seed']==failure['graph_seed']);positive=len(row['gold_edges']);negative=row['nodes']**2*13-positive
  weight=.5/(positive if failure['gold']else negative)/len(run['calibrated']['rows'])
  gradient=(1/(1+math.exp(-failure['logit']))-int(failure['gold']))*weight
  if abs(gradient-failure['weighted_loss_logit_gradient'])>1e-10:errors.append([key,'local_gradient'])
  if abs(failure['signed_threshold_margin']-(failure['logit']-failure['threshold']))>1e-6:errors.append([key,'margin'])
  for point,checkpoint in zip(failure['raw_prediction_trajectory'],run['curve']):
   oldrow=next(r for r in checkpoint['rows']if r['seed']==failure['graph_seed'])
   pred=[failure['source_index'],failure['target_index'],3]in oldrow['predicted_edges']
   if point!={'update':checkpoint['update'],'raw_prediction':pred}:errors.append([key,'raw_trajectory'])
 if found!=expected:errors.append(['failure_set'])
 if new['checkpoint_sha256']!=run['checkpoint_sha256']:errors.append(['checkpoint_sha'])
 return dict(failures=len(found),exact_failure_set=found==expected,errors=errors,passed=len(found)==5 and not errors,scope='raw label/trajectory/margin/local-gradient reconstruction; global parameter-gradient direction separately requires CPU replay')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['passed']else 1)
