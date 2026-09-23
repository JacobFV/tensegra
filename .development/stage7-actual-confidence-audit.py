import json,hashlib,subprocess,gzip
from pathlib import Path
p=Path('.development/results/stage7-ab/actual-confidence');s=json.loads((p/'actual-confidence-summary.json').read_text());print('config',s['config'])
for name,h in s['source_hashes'].items():assert hashlib.sha256(subprocess.check_output(['git','show','7e0f322:src/topoformer/'+name])).hexdigest()==h
assert hashlib.sha256(json.dumps(s['config'],sort_keys=True).encode()).hexdigest()==s['config_hash']
def metrics(scores,truth,threshold):
 chosen=[i for i,x in enumerate(scores) if x>=threshold];tp=sum(truth[i] for i in chosen);return dict(count=len(scores),selected=len(chosen),executable=sum(truth),true_positive=tp,precision=tp/len(chosen) if chosen else 0,recall=tp/sum(truth) if sum(truth) else 0,coverage=len(chosen)/len(scores) if scores else 0,brier=sum((x-y)**2 for x,y in zip(scores,truth))/len(scores) if scores else 0)
def threshold(scores,truth):
 groups={}
 for x,y in zip(scores,truth):groups.setdefault(x,[]).append(y)
 n=tp=0;eligible=[]
 for x,ys in sorted(groups.items(),reverse=True):
  n+=len(ys);tp+=sum(ys)
  if tp/n>.99:eligible.append((tp,-x,x))
 return max(eligible)[2] if eligible else 1.000001
count=0
for seed in s['config']['seeds']:
 r=json.loads(gzip.decompress((p/f'seed{seed}-actual-confidence.json.gz').read_bytes()));assert hashlib.sha256((p.parent/'ab-main'/f'seed{seed}-proposal.pt').read_bytes()).hexdigest()==r['proposal_checkpoint_sha256']
 for split in ('validation','test'):
  rows=r[split]['records'];scores=[x['calibrated'] for x in rows];truth=[x['correct'] for x in rows];assert len(rows)==s['config']['examples']*6
  public=[{k:v for k,v in x.items() if k!='calibrated'} for x in rows];assert hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()==r['data_hashes'][split]
  groups={}
  for i,x in enumerate(rows):groups.setdefault(x['group'],[]).append(i)
  assert all(len(v)==4 for v in groups.values());gs=[sum(scores[j] for j in groups[x['group']])/4 for x in rows]
  mixed=[i for i,x in enumerate(rows) if 0<sum(truth[j] for j in groups[x['group']])<4]
  for x in rows:
   full=x['primitive']==x['target_primitive'] and x['pointers'][:2]==x['target_pointers'][:2] and (x['target_primitive']==3 or x['pointers'][2]==x['target_pointers'][2]);assert full==x['full_correct'];assert x['correct']==bool(full and x['schema']);assert x['factors'][4]==float(x['schema'])
  if split=='validation':assert threshold(scores,truth)==r['threshold'];assert threshold(gs,truth)==r['global_threshold']
  for key,sc,tr,th in [('local',scores,truth,r['threshold']),('global',gs,truth,r['global_threshold']),('mixed_local',[scores[i] for i in mixed],[truth[i] for i in mixed],r['threshold']),('mixed_global',[gs[i] for i in mixed],[truth[i] for i in mixed],r['global_threshold'])]:
   calc=metrics(sc,tr,th)
   for k,v in calc.items():assert abs(v-r[split][key][k])<1e-8,(seed,split,key,k)
   for b in r[split][key]['reliability']:
    lo=b['lower']; ix=[i for i,v in enumerate(sc) if lo<=v and (v<lo+.1 or lo==.9 and v<=1)];assert len(ix)==b['count']
    if ix:assert abs(sum(sc[i] for i in ix)/len(ix)-b['confidence'])<1e-8;assert abs(sum(tr[i] for i in ix)/len(ix)-b['accuracy'])<1e-8
  for key,m in r[split]['per_condition'].items():
   indices=[i for i,x in enumerate(rows) if x['split']+'_'+x['condition']==key];assert len(indices)==s['config']['examples'];calc=metrics([scores[i] for i in indices],[truth[i] for i in indices],r['threshold'])
   for k,v in calc.items():assert abs(v-m[k])<1e-8
  local=r[split]['local'];assert abs(local['local_advantage']-(r[split]['mixed_local']['recall']-r[split]['mixed_global']['recall']))<1e-8
  gate=local['precision']>.99 and local['recall']>=.5 and local['local_advantage']>0;assert gate==r[split]['gate_b'];print(seed,split,{k:local[k] for k in ('selected','true_positive','executable','precision','recall','local_advantage')},'gate',gate)
  count+=len(rows)
print('PASS raw records',count,'thresholds groups conditions bins hashes')
