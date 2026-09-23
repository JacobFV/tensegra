"""Small planning audit of pinned surfaces; no model or reserved cache access."""
import collections,hashlib,json,time
from pathlib import Path
import torch
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest,SOURCE_COMMIT,RENDERER_VERSION,LANGUAGES
from topoformer.semantic_scaling import surface_input,tokens,targets,identifier_forms,semantic_key
from topoformer.semantic_curriculum import encode_text
from topoformer.thinking_language import lexical_bits

def run(output):
    tick=time.monotonic();torch.set_num_threads(2);assert verify_vendor_manifest();rows=[];public_targets={};feature_tokens={};collisions=[]
    for seed in range(900100001,900100009):
        original=build_tcn_example('unification',seed,difficulty=.5)
        mapping={v:f'entityfresh{i}' for i,v in enumerate(original.public[0].options)};mapping.update({v:f'variablefresh{i}' for i,v in enumerate('ABCDE')})
        for renamed,e in [(False,original),(True,build_tcn_example('unification',seed,difficulty=.5,identifier_renaming=mapping))]:
            graph=e.privileged.graph;vocab=['<unknown>','"parent"','"unify"','null']
            for lang in LANGUAGES:
                public,_=surface_input(e,lang);tok=tokens(public);gold=targets(graph,public,128,vocab,language=lang);features,length=encode_text(public.text)
                assert length==len(tok) and features.shape[1]==len(tok)
                ids={str(n.value) for n in graph.nodes if n.kind in ('ident','entity')};positions={v:[j for j,t in enumerate(tok) if t in identifier_forms(v,lang)] for v in ids};assert all(positions.values())
                first={v:p[0] for v,p in positions.items()};assert len(set(first.values()))==len(first),'identity collision'
                signature=json.dumps({'nodes':[(n.kind,n.value) for n in graph.nodes],'edges':[(x.source,x.target,x.role,x.slot) for x in graph.edges]},sort_keys=True)
                key=(lang,public.text)
                if key in public_targets and public_targets[key]!=signature:collisions.append((seed,lang))
                public_targets[key]=signature
                for token in tok:
                    bits=tuple(lexical_bits(token));assert bits not in feature_tokens or feature_tokens[bits]==token;feature_tokens[bits]=token
                row=dict(seed=seed,renamed=renamed,renderer=lang,tokens=len(tok),nodes=len(graph.nodes),graph_sha256=graph.digest(),semantic_sha256=hashlib.sha256(semantic_key(graph).encode()).hexdigest(),text_sha256=hashlib.sha256(public.text.encode()).hexdigest(),first_copy_positions=first,max_slot=max(x.slot for x in graph.edges if x.slot is not None),copy_target_valid=True,unknown_targets=int(gold['value'].eq(0).sum()))
                if seed==900100001:row['example_text']=public.text
                rows.append(row)
    result=dict(scope='16 tiny generated planning cases, 48 rendered views; not coverage/exposure or model evaluation',source_commit=SOURCE_COMMIT,renderer_version=RENDERER_VERSION,adapter_languages=list(LANGUAGES),vendor_manifest_verified=True,public_target_collisions=collisions,distinct_public=len(public_targets),lexical_hash_collisions=0,rows=rows,cpu_seconds=time.monotonic()-tick,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    import sys;run(sys.argv[1])
