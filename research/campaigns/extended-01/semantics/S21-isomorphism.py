"""Supplementary strict scored-graph isomorphism, never a replacement gate."""
import argparse,base64,collections,gzip,hashlib,json,re,signal,time
from pathlib import Path
import networkx as nx
KINDS=('scope','entity','token','str','num','ident','nil','pred','rel','node','tuple','list','record','app')
ROLES=('contains','declares','refers_to','argument','item','binds','binding_scope','field:query','field:substitution','field:pattern','field:fact','field:facts','field:scene')
VOCAB=('<unknown>','"parent"','"unify"','null')
CONFIG='5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3'
def require(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.load(gzip.open(p,'rt'))
def graph(nodes,edges,tokens):
 g=nx.MultiDiGraph()
 for i,(kind,value,copy) in enumerate(nodes):
  require(0<=kind<len(KINDS),'kind')
  if KINDS[kind] in ('ident','entity'):
   require(value==-1 and 0<=copy<len(tokens),'copy');attribute=(KINDS[kind],'public_identity',tokens[copy])
  else:
   require(copy==-1 and 0<=value<len(VOCAB),'finite value');attribute=(KINDS[kind],'finite_json',VOCAB[value])
  g.add_node(i,attribute=attribute)
 for source,target,role,slot in edges:
  require(source in g and target in g and 0<=role<len(ROLES) and -1<=slot<32,'edge schema')
  g.add_edge(source,target,relation=ROLES[role],slot=slot)
 return g
def gold_graph(t,tokens):
 n=sum(t['presence']);require(t['presence']==[True]*n+[False]*(128-n),'canonical gold prefix');nodes=list(zip(t['kind'][:n],t['value'][:n],t['copy'][:n]));e=t['edges'];require(e['shape']==[128,128,13] and e['bitorder']=='little','target shape');buf=base64.b64decode(e['packed_b64'],validate=True);require(len(buf)==128*128*13//8,'edge byte count');bits=int.from_bytes(buf,'little');edges=[]
 while bits:
  low=bits&-bits;pos=low.bit_length()-1;bits^=low;pair,r=divmod(pos,13);i,j=divmod(pair,128);edges.append((i,j,r,t['slots'][i][j]))
 return graph(nodes,edges,tokens)
def predicted_graph(records,tokens):
 nodes=[];edges=[];edge_phase=False;ended=False
 for rec in records:
  require(not ended and len(rec)==5,'record after EOS/shape');tag,a,b,c,d=rec
  if tag==1:require(not edge_phase and d==-1,'NODE phase');nodes.append((a,b,c))
  elif tag==2:edge_phase=True;edges.append((a,b,c,d))
  elif tag==3:require((a,b,c,d)==(-1,-1,-1,-1),'EOS payload');ended=True
  else:raise ValueError('record tag')
 require(ended,'missing EOS');return graph(nodes,edges,tokens)
def edge_attributes(g):return collections.Counter((d['relation'],d['slot']) for _,_,d in g.edges(data=True))
def signatures(g):
 result=collections.Counter()
 for n,d in g.nodes(data=True):
  incoming=collections.Counter((e['relation'],e['slot']) for _,_,e in g.in_edges(n,data=True));outgoing=collections.Counter((e['relation'],e['slot']) for _,_,e in g.out_edges(n,data=True));result[(d['attribute'],tuple(sorted(incoming.items())),tuple(sorted(outgoing.items())))]+=1
 return result
def edge_match(a,b):return collections.Counter((v['relation'],v['slot']) for v in a.values())==collections.Counter((v['relation'],v['slot']) for v in b.values())
class MatchTimeout(Exception):pass
def expired(*_):raise MatchTimeout()
def compare(a,b,seconds=1.):
 if seconds<=0:return 'unresolved','whole_process_internal_deadline'
 if collections.Counter(nx.get_node_attributes(a,'attribute').values())!=collections.Counter(nx.get_node_attributes(b,'attribute').values()):return 'non_isomorphic','node_attribute_multiset'
 if edge_attributes(a)!=edge_attributes(b):return 'non_isomorphic','typed_ordered_edge_multiset'
 if signatures(a)!=signatures(b):return 'non_isomorphic','attributed_directed_degree_multiset'
 previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
 try:
  match=nx.algorithms.isomorphism.MultiDiGraphMatcher(a,b,node_match=lambda x,y:x['attribute']==y['attribute'],edge_match=edge_match).is_isomorphic()
  return ('permutation_equivalent' if match else 'non_isomorphic'),'multidigraph_matcher'
 except MatchTimeout:return 'unresolved','matcher_time_limit'
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
def main(a):
 started=time.monotonic();deadline=started+105;manifest=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text());require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==CONFIG,'closed frozen main');c=manifest['config'];require(c['arms']==['original','broad'] and c['updates']==4096 and c['width']==1024,'main contract');require(sha(a.config)==CONFIG and json.loads(a.config.read_text())==c,'frozen config');require(sha(a.development)==c['inputs']['development']['sha256'],'public data binding')
 public={r['semantic_sha256']:r for r in map(json.loads,gzip.open(a.development,'rt'))};prepared={};inventory={}
 # Validate both complete endpoints before any matching.
 for arm in manifest['results']:
  require([x['update'] for x in arm['curves']]==[0,1024,2048,4096] and arm['added_presentations']==32768,'complete curves/exposure');entry=arm['curves'][-1]['development'];path=a.main/arm['arm']/entry['artifact'];require(sha(path)==entry['sha256'],'endpoint hash');d=load(path);rows=[r for r in d['rows'] if r['cell']=='3x4'];require(len(d['rows'])==3584 and len(rows)==512 and len({r['semantic_sha256'] for r in rows})==512,'full heldout population');prepared[arm['arm']]=rows;inventory[arm['arm']]=dict(path=str(path),sha256=sha(path))
 require(set(prepared)=={'original','broad'},'both arms');byarm={k:{r['semantic_sha256']:r for r in v} for k,v in prepared.items()};require(set(byarm['original'])==set(byarm['broad']),'paired identities')
 for key in byarm['original']:require(byarm['original'][key]['target']==byarm['broad'][key]['target'],'paired targets')
 results=[]
 for arm,rows in prepared.items():
  for row in rows:
   p=public[row['semantic_sha256']];require(row['seed']==p['seed'] and row['graph_sha256']==p['graph_sha256'] and (p['arity'],p['facts'])==(3,4),'public identity');require(type(row['valid'])is bool and type(row['complete'])is bool,'flags');out=dict(arm=arm,semantic_sha256=row['semantic_sha256'],canonical_exact=row['complete'],valid=row['valid'])
   if not row['valid']:require(not row['complete'],'invalid credit');status,reason='invalid_output',row['reason']
   elif time.monotonic()>=deadline:status,reason='unresolved','whole_process_internal_deadline'
   else:
    tokens=re.findall(r'\w+|[^\w\s]',p['text']);g=gold_graph(row['target'],tokens);h=predicted_graph(row['records'],tokens);status,reason=compare(h,g,min(1.,deadline-time.monotonic()))
    if row['complete']:require(status=='permutation_equivalent','canonical exact identity mapping must pass')
   out.update(status=status,reason=reason);results.append(out)
 summary={arm:dict(examples=512,canonical_exact=sum(r['canonical_exact'] for r in results if r['arm']==arm),statuses=dict(collections.Counter(r['status'] for r in results if r['arm']==arm)),reasons=dict(collections.Counter(r['reason'] for r in results if r['arm']==arm))) for arm in ('original','broad')}
 out=dict(scope='Supplementary strict scored-graph permutation equivalence, historical canonical gates unchanged. Exact public copied identity strings, finite JSON values, types, direction, relation, ordered slots and multiplicity preserved. Dataclass provenance/root bookkeeping outside historical scored target is not introduced.',config_sha256=CONFIG,source_sha256=sha(Path(__file__)),networkx_version=nx.__version__,manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),development_sha256=sha(a.development),inventory=inventory,summary=summary,events=results,cpu_wall_seconds=time.monotonic()-started,internal_deadline_seconds=105,per_match_seconds=1.)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('main','receipt','config','development','output'):p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
