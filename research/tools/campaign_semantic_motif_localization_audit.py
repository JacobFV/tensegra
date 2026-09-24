"""Independent archive-identifiable S14 oracle and localization counts."""
import argparse,ast,collections,gzip,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics/s14-motif-main';load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=load(root/'manifest.json.gz');out=json.loads((root/'localization.json').read_text());assert out['manifest_sha256']==sha(root/'manifest.json.gz');assert out['source_sha256']==sha(a.repo/'research/campaigns/extended-01/semantics/S14-localize.py');golds=[r['target'] for r in load(root/'targets.json.gz')['rows']];const={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):const[n.targets[0].id]=ast.literal_eval(n.value)
roles=const['ROLES'];kinds=const['KINDS'];old={tuple(k for k,v in json.loads(line)['nodes'] if k not in ('scope','entity')) for line in gzip.open(root.parent/'s01-data/train.jsonl.gz','rt')};new={tuple(kinds[g['kind'][i]] for i in g['present'] if kinds[g['kind'][i]] not in ('scope','entity')) for g in golds};assert len(old)==out['old_kind_sequence_count'] and len(new)==out['new_kind_sequence_count']
interventions={'copy_only':['copy'],'all_node_attributes':['kind','value','copy'],'structure_presence_edges_slots':['present','edges','slots'],'structure_plus_copy':['present','edges','slots','copy'],'structure_plus_kind_value':['present','edges','slots','kind','value'],'everything_except_presence':['kind','value','copy','edges','slots'],'all_gold_control':['present','kind','value','copy','edges','slots']}
def exact(p,g):
 slots={(i,j):s for i,j,s in p['slots']};gs={(i,j):s for i,j,s in g['slots']}
 return set(p['present'])==set(g['present']) and all(p['kind'][i]==g['kind'][i] for i in g['present']) and all(p[k][i]==v for k in ('value','copy') for i,v in enumerate(g[k]) if v>=0) and set(map(tuple,p['edges']))==set(map(tuple,g['edges'])) and all(slots.get((i,j),-1)==gs.get((i,j),-1) for i,j,r in g['edges'])
verified=0
for rec in m['artifacts']:
 x=load(root/rec['artifact']);reported=next(r for r in out['results'] if (r['seed'],r['arm'])==(rec['seed'],rec['arm']));matches=[0,0];missing=0;pos=collections.defaultdict(lambda:[0,0]);copies=collections.defaultdict(lambda:[0,0]);stats={k:dict(oracles=collections.Counter(),slot_examples=0,slot_edges=0,arity=collections.Counter(),relations={r:[0,0,0,0] for r in roles}) for k in ('raw','calibrated')}
 for row,g in zip(x['rows'],golds):
  raw=row['raw'];seq=tuple(kinds[raw['kind'][i]] for i in raw['present'] if kinds[raw['kind'][i]] not in ('scope','entity'));matches[0]+=seq in old;matches[1]+=seq in new;missing+=bool(set(g['present'])-set(raw['present']))
  for i in g['present']:pos[i][0]+=raw['kind'][i]==g['kind'][i];pos[i][1]+=1
  for i,v in enumerate(g['copy']):
   if v>=0:copies[i][0]+=raw['copy'][i]==v;copies[i][1]+=1
  for mode,pred in [('raw',raw),('calibrated',{**raw,'edges':row['calibrated_edges'],'slots':raw['slots']+row['calibrated_extra_slots']})]:
   s=stats[mode];assert exact(pred,g)==bool(row[mode+'_metrics']['semantic_equivalence'])
   for name,fields in interventions.items():s['oracles'][name]+=exact({**pred,**{k:g[k] for k in fields}},g);verified+=1
   ps={(i,j):v for i,j,v in pred['slots']};gs={(i,j):v for i,j,v in g['slots']};arg=[e for e in pred['edges'] if e[2]==roles.index('argument')];n=sum(ps.get((i,j),-1)==3 for i,j,r in arg);s['slot_examples']+=n>0;s['slot_edges']+=n;s['arity'].update(collections.Counter(i for i,j,r in arg).values())
   for k,name in enumerate(roles):
    pe={tuple(e) for e in pred['edges'] if e[2]==k};ge={tuple(e) for e in g['edges'] if e[2]==k};v=[len(pe&ge),len(pe),len(ge),sum(ps.get((i,j),-1)==gs.get((i,j),-1) for i,j,r in pe&ge)];s['relations'][name]=[i+j for i,j in zip(s['relations'][name],v)]
 assert matches==[reported['predicted_nonentity_kind_sequence_matches_old_arity4'],reported['predicted_nonentity_kind_sequence_matches_new_arity3']] and missing==reported['presence_only_oracle_unidentifiable_examples']
 for key,counts in [('canonical_type_accuracy_by_position',pos),('canonical_copy_accuracy_by_position',copies)]:assert reported[key]=={str(i):dict(correct=v[0],count=v[1],accuracy=v[0]/v[1]) for i,v in counts.items()}
 for mode,s in stats.items():
  r=reported['policies'][mode];assert dict(s['oracles'])==r['oracles'];assert s['slot_examples']==r['arg_slot3_examples'] and s['slot_edges']==r['arg_slot3_edges'];assert {str(k):v for k,v in s['arity'].items()}==r['argument_arity_histogram']
  for name,(tp,n,g,slot) in s['relations'].items():assert r['relations'][name]==dict(tp=tp,predicted=n,gold=g,typed_f1=2*tp/(n+g) if n+g else 1.,typed_and_slot_correct=slot)
receipt=dict(oracle_graph_decisions_verified=verified,all_relation_slot_arity_and_position_counts_verified=True,kind_sequence_matches_verified=True,input_sha256={str(root.relative_to(a.repo)/'localization.json'):sha(root/'localization.json')},cpu_audit_wall_seconds=time.monotonic()-t,scope='Privileged posthoc ceilings only. Joint structure substitution identifiable; absent-node edge/slot distributions discarded, so standalone presence/edge restoration is not claimed. Sequence matching omits topology and does not prove mechanistic template reuse.')
a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
