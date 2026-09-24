"""CPU archived first/individual semantic confirmation pair, without inference."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();load=lambda p:json.load(gzip.open(p));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roots=[a.root/f's12-{arm}-{a.seed}' for arm in ('constant','decay')];ms=[load(r/'manifest.json.gz') for r in roots]
for k in ('initial_state_sha256','parent_checkpoint_sha256','inherited_optimizer_steps','start_update','parent_manifest_sha256','confirmation_sha256','source_sha256','first_batch_replay','presentation_counts','optimizer_presentations','actual_unique_visited'):assert ms[0][k]==ms[1][k],k
assert {k for k in ms[0]['config'] if ms[0]['config'][k]!=ms[1]['config'][k]}=={'learning_rate','output_dir'}
assert ms[0]['config']['learning_rate']==1e-4 and ms[1]['config']['learning_rate']==1e-5
assert all(m['initial_prediction_replay_exact'] for m in ms)
assert all([(c['update'],c['evaluation_split']) for c in m['curves']]==[(16384,'development_replay'),(24576,'reserved_confirmation')] for m in ms)
initial=[load(r/'evaluation-u16384.json.gz') for r in roots]
for k in ('thresholds','calibration_records','train_rows','rows'):assert initial[0][k]==initial[1][k],k
final=[]
for root,m in zip(roots,ms):
 path=root/'evaluation-u24576.json.gz';assert sha(path)==m['curves'][-1]['sha256'];final.append(load(path)['rows'])
assert len(final[0])==len(final[1])==1024
assert [(r['seed'],r['semantic_sha256'],r['graph_sha256'],r['target']) for r in final[0]]==[(r['seed'],r['semantic_sha256'],r['graph_sha256'],r['target']) for r in final[1]]
results={}
for policy in ('raw','calibrated'):
 x=[bool(r[policy+'_metrics']['semantic_equivalence']) for r in final[0]];y=[bool(r[policy+'_metrics']['semantic_equivalence']) for r in final[1]]
 results[policy]=dict(constant=sum(x),decay=sum(y),shared_correct=sum(i and j for i,j in zip(x,y)),constant_only=sum(i and not j for i,j in zip(x,y)),decay_only=sum(not i and j for i,j in zip(x,y)),both_wrong=sum(not i and not j for i,j in zip(x,y)))
out=dict(seed=a.seed,examples=1024,manifest_sha256=[sha(r/'manifest.json.gz') for r in roots],results=results,decay_competence_threshold=103,decay_competence_pass=results['calibrated']['decay']>=103,paired_population_exact=True,parent_prediction_exact=True,paired_initial_state_and_first_batch_receipts_exact=True,final_only_reserved_artifacts=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='Independent archived pairing and counts; reserved target correctness inherits previously completed constant-arm population audit. Manifest state bindings verified here; checkpoint tensors require separate byte/state audit. No three-seed claim.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
