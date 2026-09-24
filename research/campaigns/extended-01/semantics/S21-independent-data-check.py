"""CPU independent S21 compact data/answer/split audit, no model import."""
import argparse,collections,gzip,hashlib,importlib.util,json,random,re,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
package=types.ModuleType('topoformer');package.__path__=[str(ROOT/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest
from topoformer.campaign_semantics_s19_codec import decode_records,KINDS,ROLES,NODE,EDGE,EOS,EMPTY
spec=importlib.util.spec_from_file_location('inventory',Path(__file__).with_name('S21-exclusion-inventory.py'));helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def load(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 start=time.monotonic();assert verify_vendor_manifest();base=ROOT/'research/results/campaign-01/semantics';oldtrain=load(base/'s15-shape-cache-v2/train_mixed.jsonl.gz');olddev=load(base/'s15-shape-cache-v2/development.jsonl.gz');excluded=set(json.load(gzip.open(base/'s21-prior-alpha-exclusions.json.gz','rt')))
 oldtrainix={helpers.alpha(r['nodes'],r['edges']):r for r in oldtrain};olddevix={helpers.alpha(r['nodes'],r['edges']):r for r in olddev};globalkeys=set();seenpublic={};reports={};inputs={};allcells=['2x4','3x3','3x4','4x3','4x4','5x3','5x4'];expectedtrain={'3x3':1024,'4x3':512,'4x4':512,'2x4':512,'5x3':768,'5x4':768};vocab=['<unknown>','"parent"','"unify"','null'];checks=0
 expected_shared=set()
 for cell,quota in (('3x3',1024),('4x3',512),('4x4',512)):
  positions=[i for i,r in enumerate(oldtrain) if f"{r['arity']}x{r['facts']}"==cell];random.Random(210021+allcells.index(cell)).shuffle(positions);expected_shared.update(helpers.alpha(oldtrain[i]['nodes'],oldtrain[i]['edges']) for i in positions[:quota])
 for split in ('train_broad','development','confirmation'):
  path=a.data/(split+'.jsonl.gz');rows=load(path);inputs[str(path)]=sha(path);counts=collections.Counter();keys=set();reused=set();fresh=0;maxnodes=maxrecords=0
  for r in rows:
   arity=r['arity'];cell=f"{arity}x{r['facts']}";counts[cell]+=1;key=helpers.alpha(r['nodes'],r['edges']);assert key==r['alpha_sha256'] and key not in keys and key not in globalkeys;keys.add(key)
   old=(oldtrainix if split=='train_broad' else olddevix if split=='development' else {}).get(key)
   if old is not None:
    for field in ('seed','text','nodes','edges','graph_sha256','semantic_sha256','public_sha256'):assert r[field]==old[field],(split,key,field)
    reused.add(key)
   else:assert key not in excluded;fresh+=1
   e=build_tcn_example('unification',r['seed'],difficulty={2:0.,3:1/3,4:.5,5:1.}[arity],languages=('english',));g=e.privileged.graph;nodes,edges,ids=helpers.graphparts(g)
   assert nodes==r['nodes'] and [list(x) for x in edges]==r['edges'] and g.digest()==r['graph_sha256'] and e.public[0].text==r['text']
   assert hashlib.sha256(helpers.historicalkey(g).encode()).hexdigest()==r['semantic_sha256']
   assert hashlib.sha256(r['text'].encode()).hexdigest()==r['public_sha256'];tokens=re.findall(r'\w+|[^\w\s]',r['text']);assert len(tokens)==r['tokens']
   outgoing=collections.defaultdict(list)
   for i,j,role,slot in edges:outgoing[i].append((role,slot,j))
   def children(i,role):return [j for rr,s,j in sorted(outgoing[i],key=lambda x:-1 if x[1] is None else x[1]) if rr==role]
   fields={role:j for role,s,j in outgoing[ids[g.roots[0]]]};facts=children(fields['field:facts'],'item');pattern=children(fields['field:pattern'],'argument');var=nodes[children(fields['field:query'],'argument')[0]][1];answers=[]
   assert len(facts)==r['facts'] and len(pattern)==arity
   for f in facts:
    arg=children(f,'argument');assert len(arg)==arity
    if all(nodes[p][1]==var or nodes[p][1]==nodes[q][1] for p,q in zip(pattern,arg)):answers.append(next(nodes[q][1] for p,q in zip(pattern,arg) if nodes[p][1]==var))
   assert answers==[e.privileged.answer] and e.public[0].options[e.privileged.answer_index]==answers[0]
   records=[]
   for kind,value in nodes:
    if kind in ('ident','entity'):assert value in tokens;records.append((NODE,KINDS.index(kind),-1,tokens.index(value),-1))
    else:records.append((NODE,KINDS.index(kind),vocab.index(json.dumps(value)),-1,-1))
   records.extend((EDGE,i,j,ROLES.index(role),-1 if slot is None else slot) for i,j,role,slot in sorted(edges));records.append((EOS,*EMPTY));decode_records(records,token_count=len(tokens),vocab_size=len(vocab));maxnodes=max(maxnodes,len(nodes));maxrecords=max(maxrecords,len(records))
   signature=json.dumps([nodes,edges],separators=(',',':'));assert r['text'] not in seenpublic or seenpublic[r['text']]==signature;seenpublic[r['text']]=signature;checks+=1
  if split=='train_broad':assert dict(counts)==expectedtrain and reused==expected_shared and counts['3x4']==0
  else:
   assert dict(counts)==dict.fromkeys(allcells,512)
   if split=='development':assert reused==set(olddevix)
   else:assert not reused
  globalkeys.update(keys);reports[split]=dict(rows=len(rows),cells=dict(counts),unique_alpha=len(keys),reused=len(reused),fresh_prior_disjoint=fresh,max_nodes=maxnodes,max_records=maxrecords)
 out=dict(scope='Independent seed/fullgraph/publiccopy/answer and alpha-exclusion checks; no predictions. Prior confirmation metadata used only for exclusion.',source_sha256=sha(Path(__file__)),inputs=inputs,exclusion_sha256=sha(base/'s21-prior-alpha-exclusions.json.gz'),reports=reports,independent_answer_and_fullgraph_checks=checks,total_alpha_disjoint=len(globalkeys),cpu_wall_seconds=time.monotonic()-start)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
