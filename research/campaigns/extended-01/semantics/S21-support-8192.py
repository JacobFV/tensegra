"""Fixed32768 generator-only draws; no model or campaign data reads."""
import collections,hashlib,json,re,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
package=types.ModuleType('topoformer');package.__path__=[str(ROOT/'src/topoformer')];sys.modules['topoformer']=package
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT
from topoformer.campaign_semantics_s19_codec import decode_records,KINDS,ROLES,NODE,EDGE,EOS,EMPTY

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 start=time.monotonic();assert verify_vendor_manifest();out={};texts={};checks=0;vocab=['<unknown>','"parent"','"unify"','null']
 for arity in (2,3,4,5):
  sets={f:set() for f in (3,4)};lex={f:set() for f in (3,4)};counts=collections.Counter();hist={f:collections.defaultdict(collections.Counter) for f in (3,4)};curves={f:[] for f in (3,4)}
  for attempt in range(8192):
   seed=922000000+arity*100000+attempt;e=build_tcn_example('unification',seed,difficulty={2:0.,3:1/3,4:.5,5:1.}[arity],languages=('english',));g=e.privileged.graph;nodes=list(g.nodes);ids={n.id:i for i,n in enumerate(nodes)};edges=[(ids[x.source],ids[x.target],x.role,x.slot) for x in g.edges]
   byid={n.id:n for n in nodes};outgoing=collections.defaultdict(list)
   for edge in g.edges:outgoing[edge.source].append(edge)
   def children(n,role):return [byid[x.target] for x in sorted(outgoing[n.id],key=lambda x:-1 if x.slot is None else x.slot) if x.role==role]
   root=byid[g.roots[0]];fields={x.role:byid[x.target] for x in outgoing[root.id]};facts=children(fields['field:facts'],'item');pattern=children(fields['field:pattern'],'argument');var=children(fields['field:query'],'argument')[0].value
   assert len(pattern)==arity;matches=[]
   for fact in facts:
    args=children(fact,'argument');assert len(args)==arity
    if all(p.value==var or p.value==a.value for p,a in zip(pattern,args)):matches.append(next(a.value for p,a in zip(pattern,args) if p.value==var))
   assert len(matches)==1 and matches[0]==e.privileged.answer and e.public[0].options[e.privileged.answer_index]==e.privileged.answer
   checks+=1;f=len(facts);counts[f]+=1;names={};alpha=[]
   for n in nodes:
    v=n.value
    if n.kind in ('ident','entity'):names.setdefault(v,len(names));v=['identity',names[v]]
    alpha.append((n.kind,v))
   sets[f].add(digest([alpha,sorted(edges)]));lex[f].add(g.digest());text=e.public[0].text;tok=re.findall(r'\w+|[^\w\s]',text);rec=[]
   for n in nodes:
    if n.kind in ('ident','entity'):
     assert n.value in tok;rec.append((NODE,KINDS.index(n.kind),-1,tok.index(n.value),-1))
    else:rec.append((NODE,KINDS.index(n.kind),vocab.index(json.dumps(n.value)),-1,-1))
   rec.extend((EDGE,i,j,ROLES.index(r),-1 if s is None else s) for i,j,r,s in sorted(edges));rec.append((EOS,*EMPTY));decode_records(rec,token_count=len(tok),vocab_size=len(vocab))
   signature=digest([[(n.kind,n.value) for n in nodes],edges]);assert text not in texts or texts[text]==signature;texts[text]=signature
   for key,val in [('tokens',len(tok)),('nodes',len(nodes)),('records',len(rec))]:hist[f][key][val]+=1
   if attempt+1 in (256,1024,4096,8192):
    for f in (3,4):curves[f].append(dict(arity_attempts=attempt+1,cell_draws=counts[f],unique_alpha=len(sets[f]),unique_lexical=len(lex[f]),alpha_collisions=counts[f]-len(sets[f])))
  for f in (3,4):out[f'{arity}x{f}']=dict(first_seed=922000000+arity*100000,arity_attempts=8192,cell_draws=counts[f],unique_alpha=len(sets[f]),unique_lexical=len(lex[f]),alpha_collisions=counts[f]-len(sets[f]),observed_at_least512=len(sets[f])>=512,curves=curves[f],histograms={k:dict(v) for k,v in hist[f].items()},tree_motifs=1)
 sources=[Path(__file__),Path(__file__).with_name('S21-support-8192-protocol.md'),ROOT/'src/topoformer/_vendor/tcn_language/lessons/unification.py',ROOT/'src/topoformer/_vendor/tcn_language/context.py',ROOT/'src/topoformer/campaign_semantics_s19_codec.py']
 result=dict(scope='Bounded support before historical exclusions; no campaign corpus reservation, confirmation reads, model or semantic extension.',generator_commit=SOURCE_COMMIT,total_attempts=checks,independent_answer_and_codec_checks=checks,public_target_collisions=0,cells=out,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},cpu_wall_seconds=time.monotonic()-start)
 (ROOT/'research/results/campaign-01/semantics/s21-support-8192.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
