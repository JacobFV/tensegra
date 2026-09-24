import json,hashlib,time,sys
from pathlib import Path
t=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=Path('/home/brandonin/topoformer-campaign01-semantics/s18-profile-outer-command-v1.json');d=json.loads(p.read_text());cwd=Path(d['cwd']);c=json.loads(Path(d['source_manifest_path']).read_text());assert sha(d['source_manifest_path'])==d['config_sha256']=='418864b31d66d299b3dd5f6e7470d1dec4aa5304b76856ea7d1098e4b7ebdd08';assert hashlib.sha256(json.dumps(d['argv'],separators=(',',':')).encode()).hexdigest()==d['argv_json_sha256'];assert d['argv'][0]=='/usr/bin/time' and d['argv'][5:8]==['/usr/bin/timeout','--signal=KILL','180s'];assert d['argv'][9:13]==['src/topoformer/campaign_semantics_s18_launch.py','configs/campaign-s18-profile-frozen-v2.json','--cap','180'];assert d['cap_seconds']==c['proposed_cap_seconds']==180
for path,r in d['utilities'].items():assert sha(path)==r['sha256']
source=cwd/'src/topoformer';sys.path.insert(0,str(source));from campaign_semantics_s18_freeze import verify
verify(c,source,180)
for n,h in c['source_sha256'].items():assert sha(source/n)==h and not (source/n).stat().st_mode&0o222
assert not Path(c['output_dir']).exists()
for suffix in ('.occupancy.json','.started.json','.log','.outer.txt'):assert not Path(d['argv'][-1]+suffix).exists()
out=dict(clearance='pass for root-controlled timing-only profile180',config_sha256=d['config_sha256'],outer_command_sha256=sha(p),argv_json_sha256=d['argv_json_sha256'],source_files=len(c['source_sha256']),immutable_source_and_all_reference_guards=True,whole_process_group_KILL180_GNUtime_outside=True,source_and_recipe_unchanged_except_resolved_child_PYTHONPATH=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='No model construction/forward/GPU call. Existing23 CPU tests reused and five changed runner fixtures independently passed1.24s. Main/reference remain unallocated.')
Path('/tmp/s18-profile-freeze-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
