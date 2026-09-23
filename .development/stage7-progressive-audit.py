"""Independent finite joint posterior artifact audit; standard library only."""
import gzip,hashlib,json,math
from pathlib import Path
p=Path('.development/results/stage7-ab/a2-main');m=json.loads((p/'manifest.json').read_text());summary=json.loads((p/'summary.json').read_text());assert hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()==m['config_hash']
for name,h in m['source_hashes'].items():assert hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()==h
assert set(m['gate_matrix'])=={'0','1','2'}
for seed in m['config']['seeds']:
 prior=json.loads(gzip.decompress((p.parent/'ab-main'/f'seed{seed}-raw.json.gz').read_bytes())) if (p.parent/'ab-main'/f'seed{seed}-raw.json.gz').exists() else json.loads((p.parent/'ab-main'/f'seed{seed}-raw.json').read_text())
 for split,metrics in m['gate_matrix'][str(seed)].items():assert metrics==prior['evaluations'][split]['complete']['metrics']
 r=json.loads(gzip.decompress((p/f'seed{seed}-progressive.json.gz').read_bytes()));assert r['optimizer_examples']==16000;assert r['train_cardinality']==256
 assert set(r['final'])=={'iid_validation','ood_validation','iid_test','ood_test'}
 for split,raw in r['final'].items():
  beliefs,target=raw['beliefs'],raw['private_posterior'];n=len(target);assert n==512;valid=[i for i,t in enumerate(target) if t[-1][-1]==0];invalid=[i for i in range(n) if i not in valid];argmax=lambda xs:max(range(len(xs)),key=xs.__getitem__);correct=[argmax(b[-1])==argmax(t[-1]) for b,t in zip(beliefs,target)]
  calc=dict(count=n,final_joint_accuracy=sum(correct)/n,executable_count=len(valid),no_executable_count=len(invalid),executable_joint_accuracy=sum(correct[i] for i in valid)/len(valid),reject_accuracy=sum(correct[i] for i in invalid)/len(invalid),mean_absolute_posterior_error=sum(abs(x-y) for b,t in zip(beliefs,target) for bf,tf in zip(b,t) for x,y in zip(bf,tf))/(n*5*11))
  for k,v in calc.items():assert abs(v-raw['metrics'][k])<1e-7,(seed,split,k)
  mass=[];entropy=[]
  for frame in range(5):
   mass.append(sum(sum(x for x,y in zip(b[frame],t[frame]) if y==0) for b,t in zip(beliefs,target))/n)
   entropy.append(sum(-sum(x*math.log(max(x,1e-8)) for x in b[frame]) for b in beliefs)/n)
  for k,vs in [('impossible_mass_by_frame',mass),('posterior_entropy_by_frame',entropy)]:
   for a,b in zip(vs,raw['metrics'][k]):assert abs(a-b)<4e-7,(seed,split,k,a,b)
  for episode in beliefs:
   for distribution in episode:assert all(0<=x<=1 for x in distribution) and abs(sum(distribution)-1)<5e-7
  print(seed,split,'joint',calc['final_joint_accuracy'],'final impossible',mass[-1])
assert not summary['runtime_composition_authorized']
print('PASS 6144 episodes / 30720 posterior frames, source/config/prerequisite matrix and all final metrics')
