"""Independent R11 raw metric, historical-reference and fork exposure checks."""
import argparse,gzip,hashlib,json,subprocess,time
from pathlib import Path
from campaign_return_balanced_diversity_audit import counts
from campaign_returns_audit import groups,FIELDS
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--lane',choices=('profile','development'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/f'research/results/campaign-01/returns/r11-{a.lane}';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));m=load(root/'manifest.json.gz');cfg=m['config'];rows=load(root/'predictions.json.gz');oldroot=a.repo/f'research/results/campaign-01/returns/r10-{a.lane}';old=load(oldroot/'manifest.json.gz');historical=load(oldroot/'predictions.json.gz');ref={(r['split'],r['target_delay'],r['distractors']):r for r in historical if r['head']==cfg['reference_arm']};assert sha(root/'predictions.json.gz')==m['predictions_sha256'];assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['config_sha256']
for n,h in m['source'].items():assert hashlib.sha256(subprocess.check_output(['git','show','a7f99f05:src/topoformer/'+n])).hexdigest()==h
index={}
for r in rows:
 k=r['split'],r['target_delay'],r['distractors'];assert (k,r['head']) not in index;index[k,r['head']]=r;assert counts(r)==r['counts'] and groups(r)==r['groups'];base=ref[k];assert r['targets']==base['targets'] and r['event_sha256']==base['event_sha256'];assert all(r['predictions'][f]==base['predictions'][f] for f in FIELDS[1:]);
 if r['head']=='reference':assert r['predictions']==base['predictions']
assert set(index)=={(k,arm) for k in ref for arm in ('reference','constant','decay')};assert len(rows)==108
for fit in m['fits']:
 assert fit['optimizer_presentations']==cfg['extra_steps']*cfg['batch_size'] and len(fit['losses'])==cfg['extra_steps'];bits=[(b>>i)&1 for b in bytes.fromhex(fit['visited_row_bits_little_endian']) for i in range(8)][:m['fit_rows']];assert sum(bits)==fit['unique_rows_sampled'];assert len({i%cfg['fit_events'] for i,b in enumerate(bits) if b})==fit['unique_events_sampled'];assert fit['lr']==cfg['forks'][fit['arm']]
for key in ('visited_row_bits_little_endian','sampled_index_sha256','final_rng_sha256'):assert m['fits'][0][key]==m['fits'][1][key]
assert all(m['replay'][k] for k in ('exact_reference_predictions','exact_parameters','exact_normalization','exact_loss_sequence'));assert not m['replay']['historical_index_order_hash_available'];assert m['replay']['visited_row_bits_little_endian']==next(f for f in old['fits'] if f['arm']==cfg['reference_arm'])['visited_row_bits_little_endian'];outer=json.loads((root/'outer-process.json').read_text());assert outer['exit_code']==0
out=dict(rows_verified=108,reference_predictions_targets_event_hashes_exact=True,nonvalue_outputs_invariant=True,fork_membership_order_and_rng_equal=True,source_ref='a7f99f05',full_outer_seconds=outer['outer_seconds'],phase_seconds=m['phase_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in root.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Mechanical profile only; no advancement interpretation.' if a.lane=='profile' else 'Reused development, not confirmation; frozen endpoint-only rule.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='input_sha256'})
