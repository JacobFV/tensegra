"""Artifact-supported CPU oracle localization; no model calls or new corpus."""
import ast,collections,gzip,hashlib,json
from pathlib import Path
root=Path('research/results/campaign-01/semantics/s14-motif-main');load=lambda p:json.load(gzip.open(p,'rt'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
constants={}
for node in ast.parse(Path('src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('KINDS','ROLES'):constants[node.targets[0].id]=ast.literal_eval(node.value)
kinds=constants['KINDS'];roles=constants['ROLES'];argument=roles.index('argument');contains=roles.index('contains')
targets={r['semantic_sha256']:r['target'] for r in load(root/'targets.json.gz')['rows']}
old_sequences=set()
for line in gzip.open(root.parent/'s01-data/train.jsonl.gz','rt'):
 r=json.loads(line);old_sequences.add(tuple(k for k,v in r['nodes'] if k not in ('scope','entity')))
new_sequences={tuple(kinds[g['kind'][i]] for i in g['present'] if kinds[g['kind'][i]] not in ('scope','entity')) for g in targets.values()}

def prepared(record):
 return {**record,'present':set(record['present']),'edges':{tuple(e) for e in record['edges']},'slots':{(i,j):s for i,j,s in record['slots']}}

def exact(p,g):
 return (p['present']==g['present'] and all(p['kind'][i]==g['kind'][i] for i in g['present'])
  and all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0)
  and all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0)
  and p['edges']==g['edges'] and all(p['slots'].get((i,j),-1)==g['slots'].get((i,j),-1) for i,j,r in g['edges']))

interventions={'copy_only':('copy',),'all_node_attributes':('kind','value','copy'),
 'structure_presence_edges_slots':('present','edges','slots'),
 'structure_plus_copy':('present','edges','slots','copy'),
 'structure_plus_kind_value':('present','edges','slots','kind','value'),
 'everything_except_presence':('kind','value','copy','edges','slots'),
 'all_gold_control':('present','kind','value','copy','edges','slots')}
results=[]
for entry in load(root/'manifest.json.gz')['artifacts']:
 data=load(root/entry['artifact']);summaries={p:dict(oracles={name:0 for name in interventions},exact=0,arg_slot3_examples=0,arg_slot3_edges=0,argument_arity_histogram=collections.Counter(),relations={r:[0,0,0,0] for r in roles}) for p in ('raw','calibrated')}
 old_match=new_match=0;positions=collections.defaultdict(lambda:[0,0]);copy_positions=collections.defaultdict(lambda:[0,0]);presence_activation=0
 for row in data['rows']:
  gold=prepared(targets[row['semantic_sha256']]);raw=prepared(row['raw']);cal={**raw,'edges':{tuple(e) for e in row['calibrated_edges']},'slots':{**raw['slots'],**{(i,j):s for i,j,s in row['calibrated_extra_slots']}}}
  sequence=tuple(kinds[raw['kind'][i]] for i in sorted(raw['present']) if kinds[raw['kind'][i]] not in ('scope','entity'))
  old_match+=sequence in old_sequences;new_match+=sequence in new_sequences;presence_activation+=not gold['present'].issubset(raw['present'])
  for i in gold['present']:positions[i][0]+=raw['kind'][i]==gold['kind'][i];positions[i][1]+=1
  for i,v in enumerate(gold['copy']):
   if v>=0:copy_positions[i][0]+=raw['copy'][i]==v;copy_positions[i][1]+=1
  for policy,pred in [('raw',raw),('calibrated',cal)]:
   s=summaries[policy];factual=exact(pred,gold);assert factual==bool(row[policy+'_metrics']['semantic_equivalence']);s['exact']+=factual
   for name,fields in interventions.items():s['oracles'][name]+=exact({**pred,**{field:gold[field] for field in fields}},gold)
   extra=[(i,j,r) for i,j,r in pred['edges'] if r==argument and pred['slots'].get((i,j),-1)==3]
   assert not any(r==argument and gold['slots'].get((i,j),-1)==3 for i,j,r in gold['edges'])
   s['arg_slot3_examples']+=bool(extra);s['arg_slot3_edges']+=len(extra)
   arities=collections.Counter(i for i,j,r in pred['edges'] if r==argument)
   s['argument_arity_histogram'].update(arities.values())
   for r,name in enumerate(roles):
    pe={e for e in pred['edges'] if e[2]==r};ge={e for e in gold['edges'] if e[2]==r};tp=len(pe&ge);otp=sum(pred['slots'].get((i,j),-1)==gold['slots'].get((i,j),-1) for i,j,r in pe&ge)
    for k,v in enumerate((tp,len(pe),len(ge),otp)):s['relations'][name][k]+=v
 for s in summaries.values():
  for name,(tp,np,ng,otp) in list(s['relations'].items()):s['relations'][name]=dict(tp=tp,predicted=np,gold=ng,typed_f1=2*tp/(np+ng) if np+ng else 1.,typed_and_slot_correct=otp)
 results.append(dict(seed=entry['seed'],arm=entry['arm'],policies=summaries,canonical_type_accuracy_by_position={str(i):dict(correct=v[0],count=v[1],accuracy=v[0]/v[1]) for i,v in positions.items()},canonical_copy_accuracy_by_position={str(i):dict(correct=v[0],count=v[1],accuracy=v[0]/v[1]) for i,v in copy_positions.items()},predicted_nonentity_kind_sequence_matches_old_arity4=old_match,predicted_nonentity_kind_sequence_matches_new_arity3=new_match,presence_only_oracle_unidentifiable_examples=presence_activation))
 for policy,s in summaries.items():print(entry['seed'],entry['arm'],policy,'oracles',s['oracles'],'arg_slot3_examples',s['arg_slot3_examples'])
 print('kindseq old/new',old_match,new_match,'presence activation unsupported',presence_activation)
result=dict(source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json.gz'),old_kind_sequence_count=len(old_sequences),new_kind_sequence_count=len(new_sequences),results=results,
 scope='Privileged posthoc evaluation ceilings only; not learned inference. Node attributes are stored for all slots; structure oracle replaces presence+typed edges+slots together.',limitations=['Sparse archive discarded edges at predicted-absent nodes, so presence-only intervention exposing such nodes is unidentifiable.','Edge-only replacement may require discarded slot labels; not reported as an exact oracle.','Kind-sequence matching ignores edges/slots and is suggestive template evidence, not graph reconstruction.'])
(root/'localization.json').write_text(json.dumps(result,indent=2)+'\n')
