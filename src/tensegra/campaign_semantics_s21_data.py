"""S21 bounded, versioned data-distribution intervention; no model inference.

Only ``build`` generates data. Explicit hash-pinned inputs and root clearance
are required. Prior confirmation payloads are reduced to exclusion hashes only.
"""
from __future__ import annotations
import collections,gzip,hashlib,json,random,re,time
from pathlib import Path

VERSION='s21-six-motif-v1'
CELLS=((2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4))
OLD_COUNTS={(3,3):1024,(4,3):512,(4,4):512}
NEW_COUNTS={(2,4):512,(5,3):768,(5,4):768}
PANEL_COUNTS={(3,3):32,(4,3):16,(4,4):16,(2,4):16,(5,3):24,(5,4):24}
DIFFICULTY={2:0.,3:1/3,4:.5,5:1.}
MAX_ATTEMPTS=20000
VOCAB=['<unknown>','"parent"','"unify"','null']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def alpha(nodes,edges):
    names={};result=[]
    for kind,value in nodes:
        if kind in ('ident','entity'):
            names.setdefault(value,len(names));value=['identity',names[value]]
        result.append((kind,value))
    return digest([result,sorted(edges)])

def cell(row): return row['arity'],row['facts']
def read_rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream: yield json.loads(line)

def pinned_path(spec):
    path=Path(spec['path'])
    if sha(path)!=spec['sha256']: raise ValueError('input hash mismatch: '+str(path))
    return path

def select_indices(rows,counts,seed):
    chosen=[]
    for motif in sorted(counts):
        indices=[i for i,r in enumerate(rows) if cell(r)==motif]
        random.Random(seed+CELLS.index(motif)).shuffle(indices)
        if len(indices)<counts[motif]: raise ValueError('insufficient existing TRAIN support')
        chosen.extend(indices[:counts[motif]])
    return chosen

def shape_and_depth(nodes,edges):
    kept=[i for i,(k,_) in enumerate(nodes) if k not in ('entity','scope')]
    positions={i:j for j,i in enumerate(kept)}
    tree_edges=[(i,j,r,s) for i,j,r,s in edges if r in ('argument','item') or r.startswith('field:')]
    encoded=json.dumps([[nodes[i][0] for i in kept],[(positions[i],positions[j],r,s) for i,j,r,s in tree_edges]],separators=(',',':'))
    parent={j:i for i,j,_,_ in tree_edges};depths=[]
    for i in kept:
        seen=set();depth=0
        while i in parent:
            if i in seen: raise ValueError('cycle in structural tree')
            seen.add(i);i=parent[i];depth+=1
        depths.append(depth)
    return hashlib.sha256(encoded.encode()).hexdigest(),max(depths,default=0)

def independent_answer(nodes,edges):
    """Exact compact-graph matcher, independent of generator answer code."""
    out=collections.defaultdict(list)
    for i,j,r,s in edges: out[i].append((j,r,s))
    roots=[i for i in out if any(r=='field:pattern' for _,r,_ in out[i])]
    if len(roots)!=1: raise ValueError('one problem record required')
    fields={r:j for j,r,_ in out[roots[0]]}
    def children(i,role): return [j for j,r,s in sorted(out[i],key=lambda e:-1 if e[2] is None else e[2]) if r==role]
    query=children(fields['field:query'],'argument')
    if len(query)!=1: raise ValueError('one query variable required')
    var=nodes[query[0]][1];pattern=children(fields['field:pattern'],'argument')
    matches=[]
    for fact in children(fields['field:facts'],'item'):
        args=children(fact,'argument')
        if len(args)!=len(pattern): raise ValueError('fact arity mismatch')
        binding={};valid=True
        for p,a in zip(pattern,args):
            pv,av=nodes[p][1],nodes[a][1]
            if pv==var:
                if pv in binding and binding[pv]!=av: valid=False
                binding[pv]=av
            elif pv!=av: valid=False
        if valid: matches.append(binding[var])
    if len(matches)!=1: raise ValueError('nonunique generator match')
    return matches[0]

def generate_row(arity,seed):
    from .tcn_data import build_tcn_example
    from .campaign_semantics_s19_codec import encode_row,records_to_targets
    e=build_tcn_example('unification',seed,difficulty=DIFFICULTY[arity],languages=('english',))
    g=e.privileged.graph;positions={n.id:i for i,n in enumerate(g.nodes)}
    nodes=[[n.kind,n.value] for n in g.nodes]
    edges=[[positions[x.source],positions[x.target],x.role,x.slot] for x in g.edges]
    actual=independent_answer(nodes,edges)
    if actual!=e.privileged.answer or e.public[0].options[e.privileged.answer_index]!=actual:
        raise ValueError('independent answer check failed')
    text=e.public[0].text;tokens=re.findall(r'\w+|[^\w\s]',text,re.UNICODE)
    identities={};normalized=[]
    for k,v in nodes:
        if k in ('ident','entity'):
            identities.setdefault(v,len(identities));v=['identity',identities[v]]
        normalized.append((k,v))
    semantic=[normalized,sorted(tuple(x) for x in edges),[positions[r] for r in g.roots]]
    shape,depth=shape_and_depth(nodes,edges)
    row=dict(seed=seed,text=text,graph_sha256=g.digest(),semantic_sha256=digest(semantic),
        alpha_sha256=alpha(nodes,edges),public_sha256=hashlib.sha256(text.encode()).hexdigest(),
        nodes=nodes,edges=edges,tokens=len(tokens),tree_shape_sha256=shape,structural_depth=depth,
        arity=arity,facts=sum(r=='item' for _,_,r,_ in edges))
    records=encode_row(row,VOCAB)
    # Also rejects the historical metric's unsupported pair-slot multiplicity.
    records_to_targets(records,token_count=len(tokens),vocab_size=len(VOCAB))
    if depth!=3: raise ValueError('unexpected structural depth')
    row['english_first_copy_positions']={v:tokens.index(v) for k,v in nodes if k in ('ident','entity')}
    return row

def reserve(motif,count,first_seed,seen,texts,generate=generate_row):
    rows=[];stats=collections.Counter()
    for attempt in range(MAX_ATTEMPTS):
        row=generate(motif[0],first_seed+attempt);stats['attempts']+=1
        if cell(row)!=motif: stats['other_fact_count']+=1;continue
        signature=digest([row['nodes'],row['edges']])
        if row['text'] in texts and texts[row['text']]!=signature:
            raise ValueError('same public text has incompatible targets')
        key=row['alpha_sha256']
        if key in seen: stats['excluded_or_duplicate']+=1;continue
        seen.add(key);texts[row['text']]=signature;rows.append(row)
        if len(rows)==count: break
    stats['requested']=count;stats['accepted']=len(rows)
    return rows,dict(stats)

def write_rows(path,rows):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
            for row in rows: stream.write((json.dumps(row,separators=(',',':'))+'\n').encode())

class PublicFeatureAudit:
    """Finite dataset identifiability check, not a model acquisition claim."""
    def __init__(self):
        self.seen={'float32':{},'bfloat16':{}};self.token_bits={};self.rows=0;self.max_tokens=0
    def add(self,row):
        import torch
        from .semantic_curriculum import encode_text
        from .campaign_semantics_s19_codec import encode_row
        features,length=encode_text(row['text']);tokens=re.findall(r'\w+|[^\w\s]',row['text'])
        if length!=len(tokens) or features.shape[1]!=length: raise ValueError('input truncation')
        target=digest(encode_row(row,VOCAB))
        for dtype,name in [(torch.float32,'float32'),(torch.bfloat16,'bfloat16')]:
            x=features.to(dtype).contiguous()
            key=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
            if key in self.seen[name] and self.seen[name][key]!=target:
                raise ValueError('public feature sequence has incompatible targets')
            self.seen[name][key]=target
        for token,feature in zip(tokens,features[0]):
            bits=tuple(feature[:64].tolist())
            if bits in self.token_bits and self.token_bits[bits]!=token:
                raise ValueError('lexical feature collision')
            self.token_bits[bits]=token
        self.rows+=1;self.max_tokens=max(self.max_tokens,length)
    def summary(self):
        return dict(rows=self.rows,max_tokens=self.max_tokens,feature_target_collisions=0,
                    lexical_collisions=0,unique_feature_sequences={k:len(v) for k,v in self.seen.items()},
                    scope='Exact public input sequences at FP32/BF16 before learned projection')

def build(config):
    from .tcn_data import verify_vendor_manifest,SOURCE_COMMIT
    if config.get('generation_status')!='root_and_reviewer_authorized_cpu_only':
        raise ValueError('explicit data-contract clearance required')
    if not verify_vendor_manifest(): raise ValueError('vendor integrity failed')
    if config.get('source_sha256')!=sha(__file__): raise ValueError('builder source changed')
    if config.get('cell_index_order')!=[list(c) for c in CELLS]: raise ValueError('cell map changed')
    tick=time.monotonic();out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    baseline=list(read_rows(pinned_path(config['baseline_train'])))
    development=list(read_rows(pinned_path(config['existing_development'])))
    if collections.Counter(cell(r) for r in baseline)!={(3,3):2048,(4,3):1024,(4,4):1024}:
        raise ValueError('wrong inherited TRAIN population')
    if collections.Counter(cell(r) for r in development)!={(3,3):512,(3,4):512,(4,3):512,(4,4):512}:
        raise ValueError('wrong inherited DEV population')
    seen=set();sources=[];inventory_coverage=None
    if 'exclusion_inventory' in config:
        inventory_path=pinned_path(config['exclusion_inventory'])
        inventory=json.load(gzip.open(inventory_path,'rt')) if inventory_path.suffix=='.gz' else json.loads(inventory_path.read_text())
        if isinstance(inventory,list):
            keys=inventory
            metadata_path=pinned_path(config['exclusion_inventory_metadata'])
            inventory_coverage=json.loads(metadata_path.read_text())
            if inventory_coverage['alpha_sha256']!=sha(inventory_path): raise ValueError('inventory metadata hash mismatch')
        else:
            keys=inventory['alpha_sha256'];inventory_coverage=inventory.get('coverage')
        if any(not isinstance(k,str) or len(k)!=64 for k in keys): raise ValueError('invalid alpha hash inventory')
        seen.update(keys)
        sources.append(dict(path=str(inventory_path),sha256=sha(inventory_path),role='independently built alpha-only historical exclusion inventory',unique_alpha=len(keys)))
    # Exclusions retain only opaque hashes, even for previously sealed splits.
    for spec in config.get('exclusion_sources',[]):
        path=pinned_path(spec);keys=set();count=0
        for row in read_rows(path):
            keys.add(alpha(row['nodes'],row['edges']));count+=1
        seen.update(keys);sources.append(dict(path=str(path),sha256=spec['sha256'],rows=count,unique_alpha=len(keys),role=spec['role']))
    for row in baseline+development:
        if row['alpha_sha256']!=alpha(row['nodes'],row['edges']): raise ValueError('inherited alpha metadata mismatch')
    required={alpha(r['nodes'],r['edges']) for r in baseline+development}
    if not required<=seen: raise ValueError('exclusion inventory omits inherited TRAIN/DEV')
    original_panel=json.loads(pinned_path(config['original_panel']).read_text())['mixed']
    if len(original_panel)!=128: raise ValueError('original TRAIN panel count')
    for item in original_panel:
        if any(baseline[item['index']][k]!=item[k] for k in ('seed','semantic_sha256','alpha_sha256')): raise ValueError('original TRAIN panel binding')
    old_indices=select_indices(baseline,OLD_COUNTS,210021)
    broad=[baseline[i] for i in old_indices];texts={}
    for row in baseline+development:
        signature=digest([row['nodes'],row['edges']])
        if row['text'] in texts and texts[row['text']]!=signature: raise ValueError('inherited public collision')
        texts[row['text']]=signature
    records={'train_broad':broad,'development':development,'confirmation':[]};stats={};blocked=[]
    for split,start,counts in [('train_broad',210000000,NEW_COUNTS),('development',211000000,{c:512 for c in NEW_COUNTS}),('confirmation',212000000,{c:512 for c in CELLS})]:
        for motif,count in sorted(counts.items()):
            rows,detail=reserve(motif,count,start+100000*CELLS.index(motif),seen,texts)
            records[split].extend(rows);stats[f'{split}:{motif[0]}x{motif[1]}']=detail
            if len(rows)!=count: blocked.append(f'{split}:{motif[0]}x{motif[1]}')
    random.Random(210022).shuffle(broad)
    panel=select_indices(broad,PANEL_COUNTS,210023) if not blocked else []
    panels=dict(original=original_panel,broad=[dict(index=i,**{k:broad[i][k] for k in ('seed','semantic_sha256','alpha_sha256')}) for i in panel])
    (out/'panels.json').write_text(json.dumps(panels,indent=2)+'\n')
    summaries={};hashes={};features=PublicFeatureAudit()
    # Include every comparator TRAIN row, not only the reused broad subset.
    for row in baseline: features.add(row)
    for split,rows in records.items():
        for row in rows: features.add(row)
        path=out/(split+'.jsonl.gz');write_rows(path,rows);hashes[split]=sha(path)
        summaries[split]=dict(rows=len(rows),unique_alpha=len({r['alpha_sha256'] for r in rows}),
            cells=dict(collections.Counter(f'{r["arity"]}x{r["facts"]}' for r in rows)),
            tokens=sum(r['tokens'] for r in rows),nodes=sum(len(r['nodes']) for r in rows),
            edges=sum(len(r['edges']) for r in rows),records=sum(len(r['nodes'])+len(r['edges'])+1 for r in rows),
            max_records=max((len(r['nodes'])+len(r['edges'])+1 for r in rows),default=0),max_nodes=max((len(r['nodes']) for r in rows),default=0),
            motifs=dict(collections.Counter(r['tree_shape_sha256'] for r in rows)))
    sets={name:{r['alpha_sha256'] for r in rows} for name,rows in records.items()}
    for a in sets:
        for b in sets:
            if a<b and sets[a]&sets[b]: raise ValueError('new split overlap')
    result=dict(version=VERSION,config=config,source_sha256=sha(__file__),generator_commit=SOURCE_COMMIT,
        status='blocked_support_shortfall' if blocked else 'complete_pending_independent_audit',blocked_cells=blocked,
        cache_sha256=hashes,panels_sha256=sha(out/'panels.json'),summaries=summaries,reservation_statistics=stats,excluded_sources=sources,exclusion_inventory_coverage=inventory_coverage,
        cell_index_order=CELLS,value_vocabulary=VOCAB,broad_old_original_indices=old_indices,
        broad_train_panel_indices=panel,overlap_with_baseline=len(sets['train_broad']&{r['alpha_sha256'] for r in baseline}),
        planned_training_exposure=dict(updates=4096,batch_size=8,presentations=32768,visits_per_construction=8),
        public_identifiability=True,public_feature_audit=features.summary(),independent_answer_checked=True,no_model_import_or_inference=True,
        confirmation_sealed_not_for_selection=True,cpu_wall_seconds=time.monotonic()-tick)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
    result=build(json.loads(Path(a.config).read_text()))
    print(json.dumps({k:result[k] for k in ('status','blocked_cells','cpu_wall_seconds')}))
