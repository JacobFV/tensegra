import os,sys,time,json,hashlib
from pathlib import Path
t=time.monotonic();root=Path('/home/brandonin/topoformer-campaign01-semantics');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();out={}
for job,cap,expected in [('main',4000,'4ffecfea1a6aa3b968a9568b3545c8fba0fe43044e248e821fd6209eead44fdf'),('reference',240,'944b342c8a4ae58c3ebce6ba9a4b143f02d9d033eac17875ce0d4557c768a6d6')]:
 path=root/f's18-{job}-outer-command-v1.json';d=json.loads(path.read_text());cwd=Path(d['cwd']);os.chdir(cwd);cp=Path(d['source_manifest_path']);c=json.loads(cp.read_text());assert sha(cp)==d['config_sha256']==expected;assert hashlib.sha256(json.dumps(d['argv'],separators=(',',':')).encode()).hexdigest()==d['argv_json_sha256'];assert d['argv'][0]=='/usr/bin/time' and d['argv'][5:8]==['/usr/bin/timeout','--signal=KILL',f'{cap}s'];assert d['argv'][9:13]==['src/topoformer/campaign_semantics_s18_launch.py',f'configs/campaign-s18-{job}-frozen-v1.json','--cap',str(cap)];assert d['cap_seconds']==c['proposed_cap_seconds']==cap
 for p,r in d['utilities'].items():assert sha(p)==r['sha256']
 source=cwd/'src/topoformer';sys.path.insert(0,str(source));from campaign_semantics_s18_freeze import verify
 verify(c,source,cap)
 profile=json.loads((root/'s18-source-35e94fce/configs/campaign-s18-profile-frozen-v2.json').read_text());assert c['source_sha256']==profile['source_sha256'] and c['bindings']==profile['bindings']
 for n,h in c['source_sha256'].items():assert sha(source/n)==h and not (source/n).stat().st_mode&0o222
 for p,h in c['profile_budget_evidence'].items():assert sha(cwd/p)==h
 assert not Path(c['output_dir']).exists()
 for suffix in ('.occupancy.json','.started.json','.log','.outer.txt'):assert not Path(d['argv'][-1]+suffix).exists()
 out[job]=dict(config_sha256=expected,outer_command_sha256=sha(path),argv_json_sha256=d['argv_json_sha256'],cap_seconds=cap,all_source_reference_evidence_guards=True,readonly_snapshot=True,outputs_unused=True)
out['cpu_audit_wall_seconds']=time.monotonic()-t;out['scope']='Prospective unchanged science; verified full GNUtime/KILL caps4000/240. No actor construction/forward/GPU call. Root separately releases each job.';Path('/tmp/s18-main-freeze-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
