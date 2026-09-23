"""Tiny oracle-logit representation fixture; no trained model or reserved data."""
import hashlib,json,time
from pathlib import Path
from unittest.mock import patch
import torch
from topoformer.tcn_data import build_tcn_example,verify_vendor_manifest
from topoformer.semantic_scaling import surface_input,tokens,targets,decode,metrics,identifier_forms
from topoformer.thinking_language import KINDS,ROLES


def oracle_logits(gold,token_count):
    out={}
    for key,n in [('kind',len(KINDS)),('value',4),('copy',token_count),('slots',33)]:
        labels=gold[key]+int(key=='slots');out[key]=torch.full((*labels.shape,n),-20.)
        out[key].scatter_(-1,labels.clamp_min(0)[...,None],20.)
    for key in ('presence','edges'):out[key]=gold[key].float()*40-20
    return out


def run(output):
    tick=time.monotonic();torch.set_num_threads(2);assert verify_vendor_manifest();rows=[]
    variants={'original':{},'translated_visible_alias':{'alice':'red'},'surface_collision_fixture':{'alice':'red','carol':'rojo'}}
    for name,mapping in variants.items():
        example=build_tcn_example('unification',900100001,difficulty=.5,identifier_renaming=mapping);graph=example.privileged.graph
        for lang in ('english','spanish'):
            public,_=surface_input(example,lang);tok=tokens(public);gold=targets(graph,public,128,['<unknown>','"parent"','"unify"','null'],language=lang)
            identities=sorted({str(n.value) for n in graph.nodes if n.kind in ('ident','entity')});forms={v:sorted(identifier_forms(v,lang)) for v in identities};positions={v:[i for i,t in enumerate(tok) if t in forms[v]] for v in identities};first={v:p[0] for v,p in positions.items()};collisions=[(a,b) for ai,a in enumerate(identities) for b in identities[ai+1:] if first[a]==first[b]]
            logits=oracle_logits(gold,len(tok))
            with patch('topoformer.semantic_scaling.identifier_forms',side_effect=AssertionError('inference requested hidden alias renderer')):
                pred=decode(logits,public)
            # Hidden English/canonical spelling never enters identity value labels.
            copy_nodes=[i for i,n in enumerate(graph.nodes) if n.kind in ('ident','entity')]
            assert all(int(gold['value'][i])==-1 for i in copy_nodes)
            table=[dict(node_index=i,kind=graph.nodes[i].kind,canonical_label=graph.nodes[i].value,copy_index=int(gold['copy'][i]),copied_public_token=tok[int(gold['copy'][i])],value_target=int(gold['value'][i])) for i in copy_nodes]
            rows.append(dict(variant=name,renderer=lang,public_text=public.text,graph_sha256=graph.digest(),noncopy_target_sha256=hashlib.sha256(b''.join(gold[k].numpy().tobytes() for k in ('presence','kind','value','edges','slots'))).hexdigest(),first_copy=first,visible_forms=forms,identity_collisions=collisions,copy_node_targets=table,oracle_logit_metric=metrics(pred,gold),inference_alias_lookup_calls=0,qualified_for_surface_dataset=not collisions))
    result=dict(scope='Three fixed same-instance fixtures x two surfaces; oracle logits test representation only, not learned acquisition',rows=rows,cpu_seconds=time.monotonic()-tick,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    import sys;run(sys.argv[1])
