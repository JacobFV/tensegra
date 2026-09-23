"""S13 prospective versioned cache: constructions split before their surfaces.

Preparation only until campaign authority releases generation. No model code.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from .campaign_semantics import digest
from .campaign_semantics_data import compact_example,load_cache,target as historical_target
from .campaign_semantics_surface_contract import register_surface_target
from .semantic_graph import SemanticGraph,SemanticNode,SemanticEdge
from .semantic_scaling import surface_input,targets,tokens
from .tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT,RENDERER_VERSION
from .thinking_language import ActorInput

CACHE_VERSION='campaign-multisurface-v1'
RENDERERS=('english','spanish')


def public_view(row,language):
    """Only the selected visible text is handed to a future actor."""
    if language not in RENDERERS:raise ValueError('unregistered renderer')
    return ActorInput(row['surfaces'][language]['text'],())


def target(row,vocab,language):
    """Privileged training/evaluation labels, never part of public_view."""
    graph=SemanticGraph(tuple(SemanticNode(str(i),kind,value) for i,(kind,value) in enumerate(row['nodes'])),tuple(SemanticEdge(str(i),str(j),role,slot) for i,j,role,slot in row['edges']),())
    return targets(graph,public_view(row,language),128,vocab,language=language)


def make_record(example,vocab,seen_public):
    english,_=surface_input(example,'english');record=compact_example(example,english);record['cache_version']=CACHE_VERSION;record['surfaces']={};graph=example.privileged.graph
    english_target=targets(graph,english,128,vocab,language='english')
    for language in RENDERERS:
        public,_=surface_input(example,language);gold=targets(graph,public,128,vocab,language=language)
        if gold['value'].eq(0).any():raise ValueError('target outside frozen TRAIN value vocabulary')
        contract=register_surface_target(graph,public,language,seen_public,target=gold)
        if any(not torch.equal(gold[k],english_target[k]) for k in ('presence','kind','value','edges','slots')):raise ValueError('renderer changed non-copy target semantics')
        record['surfaces'][language]=dict(text=public.text,public_sha256=hashlib.sha256(public.text.encode()).hexdigest(),tokens=len(tokens(public)),target_sha256=contract['target_sha256'])
    # Compact target conversion must preserve the audited compiler path.
    for language in RENDERERS:
        gold=targets(graph,public_view(record,language),128,vocab,language=language)
        if any(not torch.equal(gold[k],target(record,vocab,language)[k]) for k in gold):raise ValueError('compact multilingual target mismatch')
    return record


def regenerate_allowed_train(rows,vocab,seen_public):
    result=[]
    for old in rows:
        example=build_tcn_example('unification',old['seed'],difficulty=.5,languages=RENDERERS)
        record=make_record(example,vocab,seen_public)
        for key in ('seed','text','graph_sha256','semantic_sha256','public_sha256','nodes','edges','tokens'):
            if record[key]!=old[key]:raise ValueError(f'Pinned TRAIN regeneration mismatch: {key}, seed {old["seed"]}')
        old_target=historical_target(old,vocab);new_target=target(record,vocab,'english')
        if any(not torch.equal(old_target[k],new_target[k]) for k in old_target):raise ValueError('English historical target changed')
        result.append(record)
    return result


def exclusion_metadata(data,expected_hashes):
    """Read only semantic key fields; no reserved targets enter any inference."""
    excluded=set();counts={}
    for split,expected in expected_hashes.items():
        path=Path(data)/(split+'.jsonl.gz')
        if digest(path)!=expected:raise ValueError(f'Historical split byte mismatch: {split}')
        count=0
        with gzip.open(path,'rt') as stream:
            for line in stream:
                key=json.loads(line)['semantic_sha256'];excluded.add(key);count+=1
        counts[split]=count
    return excluded,counts


def write_rows(path,rows):
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as stream:
        for row in rows:stream.write((json.dumps(row,separators=(',',':'))+'\n').encode())


def run(config):
    if config.get('generation_status')!='authorized':raise ValueError('Prepared only: bulk generation has not been released')
    if config['train_count']!=8192 or config['development_count']!=512 or tuple(config['renderers'])!=RENDERERS:raise ValueError('unregistered corpus contract')
    if not verify_vendor_manifest() or config['source_commit']!=SOURCE_COMMIT:raise ValueError('pinned generator changed')
    torch.set_num_threads(2);tick=time.monotonic();data=Path(config['source_data_dir']);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    excluded,old_counts=exclusion_metadata(data,config['exclude_cache_sha256']);old_train=load_cache(data/'train.jsonl.gz')
    if len(old_train)!=8192:raise ValueError('allowed TRAIN population changed')
    vocab=json.loads((data/'audit.json').read_text())['value_vocabulary'];seen_public={};train=regenerate_allowed_train(old_train,vocab,seen_public)
    development=[];duplicates=0
    for attempt in range(config['max_development_attempts']):
        seed=config['development_seed']+attempt;example=build_tcn_example('unification',seed,difficulty=.5,languages=RENDERERS)
        english,_=surface_input(example,'english');key=compact_example(example,english)['semantic_sha256']
        if key in excluded:duplicates+=1;continue
        record=make_record(example,vocab,seen_public);excluded.add(key);development.append(record)
        if len(development)==512:break
    if len(development)!=512:raise ValueError('declared fresh semantic-support budget exhausted')
    for name,rows in [('train',train),('development',development)]:write_rows(out/(name+'.jsonl.gz'),rows)
    result=dict(cache_version=CACHE_VERSION,source_commit=SOURCE_COMMIT,renderer_version=RENDERER_VERSION,config=config,train_unique_graphs=len(train),development_unique_graphs=len(development),surfaces_per_graph=2,train_surface_count=2*len(train),development_surface_count=2*len(development),old_excluded_counts=old_counts,duplicate_semantic_keys_skipped=duplicates,development_attempts=attempt+1,all_surfaces_grouped_by_construction=True,train_regeneration_exact=True,per_renderer_copy_injectivity_validated=True,public_target_consistency_validated=True,value_vocabulary=vocab,cpu_seconds=time.monotonic()-tick,cache_sha256={n:digest(out/(n+'.jsonl.gz')) for n in ('train','development')},source_sha256={n:digest(Path(__file__).with_name(n)) for n in ('campaign_semantics_multisurface_data.py','campaign_semantics_surface_contract.py','campaign_semantics_data.py','semantic_scaling.py','tcn_data.py','semantic_graph.py')})
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();print(json.dumps(run(json.loads(Path(a.config).read_text()))))
