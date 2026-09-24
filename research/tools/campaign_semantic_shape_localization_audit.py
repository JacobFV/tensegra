"""Independent sparse-set S15 residual and privileged-replacement audit."""
import argparse,ast,collections,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set,components
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/semantics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));reported=json.loads((root/'s15-localization.json').read_text());cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(root/'s15-shape-cache-v2/development.jsonl.gz','rt'))};constants={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):constants[n.targets[0].id]=ast.literal_eval(n.value)
kinds,roles=constants['KINDS'],constants['ROLES'];seqs=collections.defaultdict(set)
for d in cache.values():seqs[f"{d['arity']}x{d['facts']}"].add(tuple(kinds.index(k) for k,v in d['nodes'] if k not in ('scope','entity')))
fields=('presence','kind','value','copy','edges','slots');replacements={k:{k} for k in fields};replacements.update(edges_and_slots={'edges','slots'},presence_and_kind={'presence','kind'},node_attributes={'kind','value','copy'},structure={'presence','edges','slots'},all_except_edges=set(fields)-{'edges'},all_except_slots=set(fields)-{'slots'},all_gold=set(fields));decisions=0;outcomes={};train_exact={}
for arm in ('control','mixed'):
 path=root/f's15-{arm}-main-v3/evaluation-u28672.json.gz';assert sha(path)==reported['endpoint_sha256'][arm];receipt=json.loads((a.review/f'S15-{arm}-main-audit.json').read_text());assert sha(path)==receipt['input_sha256'][str(path.relative_to(a.repo))];document=load(path);rows=document['rows'];result={};train_exact[arm]={policy:sum(components(r['raw'] if policy=='raw' else {**r['raw'],'edges':r['calibrated_edges']},r['target'])['semantic_equivalence'] for r in document['train_rows']) for policy in ('raw','calibrated')}
 for cell in ('3x3','3x4','4x3','4x4'):
  chosen=[r for r in rows if f"{cache[r['semantic_sha256']]['arity']}x{cache[r['semantic_sha256']]['facts']}"==cell];assert len(chosen)==512;summary=dict(examples=512,policies={},node_count_pairs=collections.Counter(),kind_sequence_matches=collections.Counter(),exact_presence_and_kind=0)
  for r in chosen:
   raw,g=r['raw'],r['target'];P={i for i,v in enumerate(raw['presence']) if v};G={i for i,v in enumerate(g['presence']) if v};summary['node_count_pairs'][f'{len(G)}->{len(P)}']+=1;seq=tuple(raw['kind'][i] for i in sorted(P) if kinds[raw['kind'][i]] not in ('scope','entity'))
   for name,known in seqs.items():summary['kind_sequence_matches'][name]+=seq in known
   summary['exact_presence_and_kind']+=P==G and all(raw['kind'][i]==g['kind'][i] for i in G)
  for policy in ('raw','calibrated'):
   v=dict(exact_components=collections.Counter(),failure_patterns=collections.Counter(),oracles=collections.Counter(),relation_oracles=collections.Counter(),relations={role:dict(false_positive=0,false_negative=0,wrong_slots_on_present_gold_edges=0,gold_edges=0) for role in roles},arg_slot3_examples=0)
   for row in chosen:
    p=row['raw'];g=row['target'];P={i for i,b in enumerate(p['presence']) if b};G={i for i,b in enumerate(g['presence']) if b};ge=edge_set(g);full=edge_set(p if policy=='raw' else {**p,'edges':row['calibrated_edges']});pe={e for e in full if e[0] in P and e[1] in P};with_gold_presence={e for e in full if e[0] in G and e[1] in G};c=dict(presence=P==G,kind=all(p['kind'][i]==g['kind'][i] for i in G),value=all(p['value'][i]==z for i,z in enumerate(g['value']) if z>=0),copy=all(p['copy'][i]==z for i,z in enumerate(g['copy']) if z>=0),edges=pe==ge,slots=all(p['slots'][i][j]==g['slots'][i][j] for i,j,r in ge));assert all(c.values())==bool(row[policy+'_metrics']['semantic_equivalence']);v['exact_components'].update(k for k,z in c.items() if z);v['failure_patterns'][','.join(k for k,z in c.items() if not z) or 'none']+=1
    for name,keys in replacements.items():
     # Exact presence is mandatory. After replacing presence, archived unmasked
     # edge decisions are restricted to gold nodes, not original predicted nodes.
     ok=all(c[k] or k in keys for k in fields if k!='edges') and ('edges' in keys or (with_gold_presence if 'presence' in keys else pe)==ge);v['oracles'][name]+=ok;decisions+=1
    v['arg_slot3_examples']+=any(role==roles.index('argument') and p['slots'][i][j]==3 for i,j,role in pe)
    for relation,name in enumerate(roles):
     a0={e for e in pe if e[2]==relation};b0={e for e in ge if e[2]==relation};stat=v['relations'][name];stat['false_positive']+=len(a0-b0);stat['false_negative']+=len(b0-a0);stat['gold_edges']+=len(b0);stat['wrong_slots_on_present_gold_edges']+=sum(p['slots'][i][j]!=g['slots'][i][j] for i,j,r in a0&b0);v['relation_oracles'][name]+=all(c[k] for k in fields if k!='edges') and {e for e in pe if e[2]!=relation}=={e for e in ge if e[2]!=relation};decisions+=1
   summary['policies'][policy]=v
  result[cell]=summary
 # JSON normalizes Counter and bool/int sums for exact comparison.
 assert json.loads(json.dumps(result))==reported['results'][arm]
 outcomes[arm]={cell:result[cell]['policies']['calibrated']['oracles'] for cell in result}
assert train_exact=={'control':{'raw':12,'calibrated':42},'mixed':{'raw':0,'calibrated':14}}
assert reported['source_sha256']==sha(a.repo/'research/campaigns/extended-01/semantics/S15-localize.py')
paths=[root/'s15-localization.json',a.repo/'research/campaigns/extended-01/semantics/S15-localize.py',a.repo/'research/campaigns/extended-01/semantics/S15-localization-report.md'];paths=[p for p in paths if p.exists()]
out=dict(historical_TRAIN128_exact=train_exact,policy_graphs=8192,privileged_replacement_decisions=decisions,all_component_failure_relation_slot_and_template_counts_equal=True,calibrated_oracles=outcomes,input_sha256={str(p.relative_to(a.repo)):sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-t,scope='Posthoc privileged replacements, not learned repairs. Full packed unmasked edges/all slots permit individual substitutions. Kind-sequence matches refer only to DEV-observed templates and do not establish TRAIN copying or cause; S15 gates remain failed.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('calibrated_oracles','input_sha256')})
