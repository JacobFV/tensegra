import hashlib,json,subprocess,sys,tempfile,time,types
from pathlib import Path
t=time.monotonic();ref='c34effb';src=subprocess.check_output(['git','show',ref+':src/topoformer/campaign_composition_confirm_batch.py']);ns={'__name__':'review_module','__file__':'/tmp/review-c04-batch-source.py'};Path(ns['__file__']).write_bytes(src);exec(compile(src,ns['__file__'],'exec'),ns);hashes=[]
for i in range(3):
 p=f'configs/campaign-c04-confirmation-{i}.json';raw=subprocess.check_output(['git','show',ref+':'+p]);hashes.append(hashlib.sha256(raw).hexdigest());new=json.loads(raw);old=json.loads(subprocess.check_output(['git','show','8a92869:'+p]));assert {k for k in new if new[k]!=old.get(k)}=={'status','requested_phase_caps_seconds'};assert sum(new['requested_phase_caps_seconds'].values())==1980
results=[]
for failure in (None,'n1_roles'):
 with tempfile.TemporaryDirectory() as td:
  r=Path(td);p=r/'config.json';p.write_bytes(subprocess.check_output(['git','show',ref+':configs/campaign-c04-confirmation-0.json']));prefix=r/'run';called=[]
  def run(cmd):
   phase=cmd[cmd.index('--phase')+1];called.append(phase);out=Path(cmd[cmd.index('--output')+1]);out.mkdir();(out/'endpoint.pt').write_bytes(phase.encode());Path(str(out)+'.occupancy.json').write_text(json.dumps(dict(process_occupancy_seconds=1.)));return types.SimpleNamespace(returncode=1 if phase==failure else 0)
  ns['subprocess']=types.SimpleNamespace(run=run,check_output=lambda *a,**k:'');sys.argv=['batch','--config',str(p),'--prefix',str(prefix),'--released-replicate','0']
  try:ns['main']()
  except SystemExit as e:assert e.code==(1 if failure else 0)
  d=json.loads(Path(str(prefix)+'.batch.json').read_text());assert d['total_process_occupancy_seconds']==len(called);assert len(called)==(3 if failure else 5);assert not list(r.glob('*.tmp'))
  if failure is None:
   config=json.loads(Path(str(prefix)+'-timing-config.json').read_text());assert all(v['sha256']==hashlib.sha256(k.encode()).hexdigest() for k,v in config['neural_checkpoints'].items())
  results.append(dict(failed_phase=failure,phases_called=called,durable_receipts=True))
out=dict(source_ref=ref,config_sha256=hashes,scientific_runner_changes_from_profile=[],config_only_status_caps_changed=True,cap_per_lineage=1980,total_cap=5940,mocked_control_flow=results,cpu_audit_wall_seconds=time.monotonic()-t,scope='Independent local no-model control-flow verification and immutable config diff. Freeze clears separately released serial lineages; no GPU permission conveyed by this receipt.')
Path('research/campaigns/extended-01/review/C04-freeze-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
