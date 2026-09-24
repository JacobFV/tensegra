"""Bounded existing-generator-only support probe. No dataset/model/confirmation access."""
import collections,hashlib,json,re,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
package=types.ModuleType('topoformer');package.__path__=[str(ROOT/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT
from topoformer.campaign_semantics_s19_codec import decode_records,KINDS,ROLES,NODE,EDGE,EOS,EMPTY

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main(arities=(2,3,4), output_name="s21-generator-feasibility.json"):
 start=time.monotonic();assert verify_vendor_manifest();out={};texts={};checks=0;total=0
 for arity in arities:
  out[f'{arity}x2']=dict(status='unavailable_under_existing_generator',proof='Hardened: one matching fact plus randint(2,3) near misses. Unhardened: exactly one fact.')
  for wanted in (3,4):
   seen=set();lexical=set();hist=collections.defaultdict(collections.Counter);accepted=0;duplicates=0;curve=[];vocab=['<unknown>','"parent"','"unify"','null']
   for attempt in range(256):
    seed=921000000+arity*100000+wanted*1000+attempt;difficulty={2:0.,3:1/3,4:.5,5:1.}[arity];e=build_tcn_example('unification',seed,difficulty=difficulty,languages=('english',));g=e.privileged.graph;nodes=list(g.nodes);ids={n.id:i for i,n in enumerate(nodes)};edges=[(ids[x.source],ids[x.target],x.role,x.slot) for x in g.edges]
    byid={n.id:n for n in nodes};outgoing=collections.defaultdict(list)
    for edge in g.edges:outgoing[edge.source].append(edge)
    def children(n,role):return [byid[x.target] for x in sorted(outgoing[n.id],key=lambda x:-1 if x.slot is None else x.slot) if x.role==role]
    root=byid[g.roots[0]];fields={x.role:byid[x.target] for x in outgoing[root.id]};facts=children(fields['field:facts'],'item');pattern=children(fields['field:pattern'],'argument');query=children(fields['field:query'],'argument');var=query[0].value
    assert len(pattern)==arity;matches=[]
    for fact in facts:
     args=children(fact,'argument');assert len(args)==arity
     if all(p.value==var or p.value==a.value for p,a in zip(pattern,args)):matches.append(next(a.value for p,a in zip(pattern,args) if p.value==var))
    assert len(matches)==1 and matches[0]==e.privileged.answer
    assert e.public[0].options[e.privileged.answer_index]==e.privileged.answer
    checks+=1;total+=1
    if len(facts)!=wanted:continue
    accepted+=1;names={};alpha=[]
    for n in nodes:
     v=n.value
     if n.kind in ('ident','entity'):names.setdefault(v,len(names));v=['identity',names[v]]
     alpha.append((n.kind,v))
    key=digest([alpha,sorted(edges)]);duplicates+=key in seen;seen.add(key);lexical.add(g.digest());text=e.public[0].text;tok=re.findall(r'\w+|[^\w\s]',text)
    # Decode fully predicted-form records with only public copy addresses: mechanical fit, not model inference.
    rec=[]
    for n in nodes:
     if n.kind in ('ident','entity'):
      assert n.value in tok;rec.append((NODE,KINDS.index(n.kind),-1,tok.index(n.value),-1))
     else:rec.append((NODE,KINDS.index(n.kind),vocab.index(json.dumps(n.value)),-1,-1))
    rec.extend((EDGE,i,j,ROLES.index(r),-1 if s is None else s) for i,j,r,s in sorted(edges));rec.append((EOS,*EMPTY));decode_records(rec,token_count=len(tok),vocab_size=len(vocab),require_canonical=False)
    assert text not in texts or texts[text]==digest([[n.kind,n.value] for n in nodes]+[edges]);texts[text]=digest([[n.kind,n.value] for n in nodes]+[edges])
    hist['public_tokens'][len(tok)]+=1;hist['nodes'][len(nodes)]+=1;hist['records'][len(rec)]+=1
    if (attempt+1)%64==0:curve.append(dict(attempts=attempt+1,accepted=accepted,unique_alpha=len(seen)))
   out[f'{arity}x{wanted}']=dict(status='bounded_probe_not_512_support_proof',first_seed=921000000+arity*100000+wanted*1000,attempts=256,selected_cell_draws=accepted,unique_alpha=len(seen),unique_lexical=len(lexical),duplicate_alpha=duplicates,curve=curve,histograms={k:dict(v) for k,v in hist.items()},tree_motifs=1)
 sources=[Path(__file__),ROOT/'src/topoformer/_vendor/tcn_language/lessons/unification.py',ROOT/'src/topoformer/_vendor/tcn_language/context.py',ROOT/'src/topoformer/campaign_semantics_s19_codec.py']
 result=dict(scope='Generator-only fresh bounded probes; no main dataset, model, historical/confirmation lookup, or semantics extension.',generator_commit=SOURCE_COMMIT,attempts=total,independent_answer_checks=checks,visible_copy_and_codec_checks=sum(x.get('selected_cell_draws',0) for x in out.values()),public_target_collisions=0,cells=out,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},cpu_wall_seconds=time.monotonic()-start)
 (ROOT/'research/results/campaign-01/semantics'/output_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--arity5-only',action='store_true');a=p.parse_args()
 if a.arity5_only:main((5,), 's21-generator-arity5-feasibility.json')
 else:main()
