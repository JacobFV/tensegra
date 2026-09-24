"""Independently regenerate historical corpus identities; never consume predictions."""
import collections,gzip,hashlib,json,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
package=types.ModuleType('topoformer');package.__path__=[str(ROOT/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,LESSONS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alpha(nodes,edges):
 mapping={};result=[]
 for kind,value in nodes:
  if kind in ('ident','entity'):mapping.setdefault(value,len(mapping));value=['identity',mapping[value]]
  result.append((kind,value))
 return hashlib.sha256(json.dumps([result,sorted(edges)],sort_keys=True,separators=(',',':')).encode()).hexdigest()
def graphparts(g):
 ids={n.id:i for i,n in enumerate(g.nodes)}
 return [[n.kind,n.value] for n in g.nodes],[(ids[e.source],ids[e.target],e.role,e.slot) for e in g.edges],ids

def historicalkey(g):
 nodes,edges,ids=graphparts(g);mapping={};result=[]
 for kind,value in nodes:
  if kind in ('ident','entity'):mapping.setdefault(value,len(mapping));value=['identity',mapping[value]]
  result.append((kind,value))
 return json.dumps([result,sorted(edges),[ids[r] for r in g.roots]],sort_keys=True,separators=(',',':'))

def main():
 start=time.monotonic();assert verify_vendor_manifest();base=ROOT/'research/results';campaign=base/'campaign-01/semantics';keys=set();sources=[];regenerations=[]
 # Explicit known caches; do not glob future datasets or model output files.
 names=['s01-data/train','s01-data/development','s01-data/reserved_confirmation','s13-multisurface-cache/train','s13-multisurface-cache/development','s13-cache-profile/train-prefix','s12-rename-case-audit/renamed_confirmation','s14-arity3-cache/diagnostic']+[f'{v}/{s}' for v in ('s15-shape-cache','s15-shape-cache-v2') for s in ('train_mixed','train_control','development','confirmation')]
 for name in names:
  p=campaign/(name+'.jsonl.gz');local=set();count=0
  for line in gzip.open(p,'rt'):
   row=json.loads(line);local.add(alpha(row['nodes'],row['edges']));count+=1
  sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=count,unique_alpha=len(local),new_union=len(local-keys),use='metadata_only_split_exclusion'));keys.update(local)
 # Reconstruct complete published corpus digests, including non-unification rows,
 # then add only unification identities. This also checks pinned source compatibility.
 for folder in ('stage7-semantic/scaling-corrected-10000','stage8/semantic-main-n10000'):
  mpath=base/folder/'manifest.json';cpath=base/folder/'corpus.json';m=json.loads(mpath.read_text());audit=json.loads(cpath.read_text());c=m['config'];pinned=m.get('pinned_generator',m.get('pinned_audit'))
  for filename in ('tcn_data.py','semantic_graph.py'):assert sha(ROOT/'src/topoformer'/filename)==m['source_sha256'][filename]
  if pinned and isinstance(pinned,dict) and 'semantic_digest' in pinned:
   e=build_tcn_example(pinned['lesson'],pinned['seed'],difficulty=pinned.get('difficulty'));assert e.privileged.graph.digest()==pinned['semantic_digest']
  seen=set();digest=hashlib.sha256();local=set();tick=time.monotonic()
  for attempt in range(audit['attempts']):
   lesson=LESSONS[attempt%3];seed=c['data_seed']+attempt;e=build_tcn_example(lesson,seed,difficulty=audit['difficulty'],languages=('english',));key=historicalkey(e.privileged.graph)
   if key not in seen:
    seen.add(key);digest.update(key.encode());digest.update(f'{lesson}:{seed}'.encode())
   if lesson=='unification':local.add(alpha(*graphparts(e.privileged.graph)[:2]))
  assert len(seen)==audit['actual_unique_graphs'] and digest.hexdigest()==audit['dataset_sha256']
  regenerations.append(dict(path=folder,manifest_sha256=sha(mpath),corpus_sha256=sha(cpath),attempts=audit['attempts'],exact_dataset_digest_verified=True,unique_unification_alpha=len(local),new_union=len(local-keys),cpu_wall_seconds=time.monotonic()-tick));keys.update(local)
 # Conservative bounded configured-candidate exclusions for tiny stages.
 for label,startseed,count,stride,difficulty in [('stage6_configured_max_candidates',610001,1920,3,.5),('stage9_fixture_superset',9000001,34,3,.5),('stage10_fixture_superset',9001001,34,3,.5),('stage11_fixed_superset',11000000,16,1,.5)]:
  local=set()
  for i in range(count):local.add(alpha(*graphparts(build_tcn_example('unification',startseed+i*stride,difficulty=difficulty,languages=('english',)).privileged.graph)[:2]))
  regenerations.append(dict(path=label,first_seed=startseed,count=count,stride=stride,difficulty=difficulty,conservative_overexclusion=True,unique_alpha=len(local),new_union=len(local-keys)));keys.update(local)
 p=base/'stage11/semantic-text/fresh-observability.json';fresh=json.loads(p.read_text());local=set()
 for row in fresh['rows']:
  e=build_tcn_example('unification',row['seed'],difficulty=.5,languages=('english',));assert e.privileged.graph.digest()==row['graph_sha256'];local.add(alpha(*graphparts(e.privileged.graph)[:2]))
 regenerations.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=len(fresh['rows']),all_graph_hashes_verified=True,unique_alpha=len(local),new_union=len(local-keys)));keys.update(local)
 dest=campaign/'s21-prior-alpha-exclusions.json.gz'
 with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as f:f.write(json.dumps(sorted(keys),separators=(',',':')).encode())
 result=dict(scope='Conservative exclusion inventory, no scientific predictions consumed. Complete Stage7/8 corpus digests verified; some tiny configured ranges overexcluded. Reused S15TRAIN/DEV require explicit reuse exceptions.',source_sha256=sha(Path(__file__)),sources=sources,regenerations=regenerations,unique_alpha=len(keys),alpha_artifact=str(dest.relative_to(ROOT)),alpha_sha256=sha(dest),gaps=['Unarchived or undocumented ad-hoc constructions cannot be certified absent. Generator-only feasibility probes are not learned/confirmation populations and are not reserved datasets.'],cpu_wall_seconds=time.monotonic()-start)
 (campaign/'s21-exclusion-inventory.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
