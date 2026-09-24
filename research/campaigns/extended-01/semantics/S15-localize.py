"""CPU endpoint residuals and privileged replacements from full S15 packed labels."""
import ast,base64,collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
root=Path('research/results/campaign-01/semantics');load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
constants={}
for n in ast.parse(Path('src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):constants[n.targets[0].id]=ast.literal_eval(n.value)
kinds=constants['KINDS'];roles=constants['ROLES'];arg=roles.index('argument')
cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(root/'s15-shape-cache-v2/development.jsonl.gz','rt'))};sequences=collections.defaultdict(set)
for r in cache.values():sequences[f"{r['arity']}x{r['facts']}"].add(tuple(kinds.index(k) for k,v in r['nodes'] if k not in ('scope','entity')))

def edges(x):
 assert x['bitorder']=='little';return np.unpackbits(np.frombuffer(base64.b64decode(x['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(x['shape'])].reshape(x['shape']).astype(bool)
def unpack(r):return {k:(edges(v) if k=='edges' else np.asarray(v)) for k,v in r.items()}
def components(p,g):
 e=p['edges']&p['presence'][:,None,None]&p['presence'][None,:,None]
 return dict(presence=bool(np.array_equal(p['presence'],g['presence'])),kind=bool(np.all(p['kind'][g['presence']]==g['kind'][g['presence']])),value=bool(np.all(p['value'][g['value']>=0]==g['value'][g['value']>=0])),copy=bool(np.all(p['copy'][g['copy']>=0]==g['copy'][g['copy']>=0])),edges=bool(np.array_equal(e,g['edges'])),slots=bool(np.all(p['slots'][g['edges'].any(-1)]==g['slots'][g['edges'].any(-1)])))
fields=('presence','kind','value','copy','edges','slots');interventions={k:(k,) for k in fields};interventions.update(edges_and_slots=('edges','slots'),presence_and_kind=('presence','kind'),node_attributes=('kind','value','copy'),structure=('presence','edges','slots'),all_except_edges=('presence','kind','value','copy','slots'),all_except_slots=('presence','kind','value','copy','edges'),all_gold=fields)
results={};hashes={}
for arm in ('control','mixed'):
 path=root/f's15-{arm}-main-v3/evaluation-u28672.json.gz';d=load(path);hashes[arm]=sha(path);summaries={}
 for cell in ('3x3','3x4','4x3','4x4'):
  rows=[r for r in d['rows'] if f"{cache[r['semantic_sha256']]['arity']}x{cache[r['semantic_sha256']]['facts']}"==cell]
  result=dict(examples=len(rows),policies={},node_count_pairs=collections.Counter(),kind_sequence_matches=collections.Counter(),exact_presence_and_kind=0)
  for r in rows:
   g=unpack(r['target']);p=unpack(r['raw']);result['node_count_pairs'][f"{int(g['presence'].sum())}->{int(p['presence'].sum())}"]+=1
   seq=tuple(p['kind'][i].item() for i in np.flatnonzero(p['presence']) if kinds[p['kind'][i]] not in ('scope','entity'))
   for name,known in sequences.items():result['kind_sequence_matches'][name]+=seq in known
   result['exact_presence_and_kind']+=components(p,g)['presence'] and components(p,g)['kind']
  for policy in ('raw','calibrated'):
   s=dict(exact_components=collections.Counter(),failure_patterns=collections.Counter(),oracles=collections.Counter(),relation_oracles=collections.Counter(),relations={name:dict(false_positive=0,false_negative=0,wrong_slots_on_present_gold_edges=0,gold_edges=0) for name in roles},arg_slot3_examples=0)
   for r in rows:
    g=unpack(r['target']);p=unpack(r['raw'])
    if policy=='calibrated':p['edges']=edges(r['calibrated_edges'])
    c=components(p,g);assert not np.any(g['value']==0);assert all(c.values())==bool(r[policy+'_metrics']['semantic_equivalence'])
    s['exact_components'].update(k for k,v in c.items() if v);s['failure_patterns'][','.join(k for k,v in c.items() if not v) or 'none']+=1
    for name,keys in interventions.items():s['oracles'][name]+=all(components({**p,**{k:g[k] for k in keys}},g).values())
    e=p['edges']&p['presence'][:,None,None]&p['presence'][None,:,None]
    s['arg_slot3_examples']+=bool(np.any(e[:,:,arg]&(p['slots']==3)))
    for i,name in enumerate(roles):
     pe=e[:,:,i];ge=g['edges'][:,:,i];v=s['relations'][name];v['false_positive']+=int(np.sum(pe&~ge));v['false_negative']+=int(np.sum(ge&~pe));v['gold_edges']+=int(ge.sum());v['wrong_slots_on_present_gold_edges']+=int(np.sum(pe&ge&(p['slots']!=g['slots'])))
     replaced=p['edges'].copy();replaced[:,:,i]=g['edges'][:,:,i];s['relation_oracles'][name]+=all(components({**p,'edges':replaced},g).values())
   result['policies'][policy]=s
  summaries[cell]=result
 results[arm]=summaries
out=dict(source_sha256=sha(Path(__file__)),endpoint_sha256=hashes,results=results,scope='Privileged posthoc exact-label replacements, not learned inference. S15 packs all presence/type/value/copy labels, all pair slots and unmasked raw/cal edge decisions, so finer replacements are recoverable unlike S14 compact archives. No logits/margins or alternative thresholds inferred; no model calls or new data.')
(root/'s15-localization.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,r in results.items():
 for cell in ('3x3','3x4'):
  print(arm,cell,'nodecounts',r[cell]['node_count_pairs'],'kindseq',r[cell]['kind_sequence_matches'])
  for policy,s in r[cell]['policies'].items():print(policy,'exactcomponents',s['exact_components'],'oracles',s['oracles'],'patterns',s['failure_patterns'].most_common(8),'relationoracles',s['relation_oracles'])
