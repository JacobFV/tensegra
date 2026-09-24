"""CPU-only S16 public-input proof audit: no model/graph/target access."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
from topoformer.thinking_language import ActorInput,lexical_bits
from topoformer.semantic_scaling import tokens
from topoformer.semantic_curriculum import encode_text
from topoformer.campaign_semantics_identity_contract import (
    canonicalize_public_identifiers,restore_public_copy_tokens,
    S16_VARIABLE_INVENTORY,S16_NAME_INVENTORY)


def public_texts(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line)['text'] for line in stream]


def file_hash(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(original,renamed,train,output):
    torch.set_num_threads(2);tick=time.monotonic()
    if file_hash(original)!='fc85da89fe4e1a522d1e0256fc1e58f8afb279494cb1af7ee4ac7235403a43f8':
        raise ValueError('not the frozen S12 original public population')
    if file_hash(renamed)!='a3f3bc5ed3d6fb188f5fb787d079af532cc67ec170a6b89937ea1d0f7777a4fb':
        raise ValueError('not the frozen S12 renamed public population')
    original_text=public_texts(original);renamed_text=public_texts(renamed)
    if len(original_text)!=1024 or len(renamed_text)!=1024:raise ValueError('wrong population size')
    train_lexicon={t for text in public_texts(train) for t in tokens(ActorInput(text,()))}
    inventory=(*S16_VARIABLE_INVENTORY,*S16_NAME_INVENTORY)
    if not set(inventory)<=train_lexicon:raise ValueError('canonical lexeme absent from public TRAIN')
    records=[];rejected=[];arities=collections.Counter();facts=collections.Counter();lexemes=set()
    for index,(left,right) in enumerate(zip(original_text,renamed_text)):
        try:
            a=canonicalize_public_identifiers(ActorInput(left,()))
            b=canonicalize_public_identifiers(ActorInput(right,()))
        except ValueError as error:
            rejected.append(dict(index=index,error=str(error)));continue
        if a.canonical_tokens!=b.canonical_tokens or a.identity_positions!=b.identity_positions:
            raise ValueError('normalized paired public token sequence differs')
        af,al=encode_text(a.public.text);bf,bl=encode_text(b.public.text)
        if al!=bl or not torch.equal(af,bf) or not torch.equal(af.to(torch.bfloat16),bf.to(torch.bfloat16)):
            raise ValueError('full actor input features differ')
        for item in (a,b):
            ids=set(item.identity_positions)
            if any(item.original_tokens[i]!=item.canonical_tokens[i] for i in range(len(item.original_tokens)) if i not in ids):
                raise ValueError('structural public token changed')
            if any(item.original_tokens.index(t)!=item.canonical_tokens.index(item.canonical_tokens[i]) for i,t in enumerate(item.original_tokens)):
                raise ValueError('all-token copy first-occurrence contract changed')
            if restore_public_copy_tokens(item,range(len(item.original_tokens)))!=item.original_tokens:
                raise ValueError('original public copy inventory not restored')
            lexemes.update(item.original_tokens);lexemes.update(item.canonical_tokens)
        arities[a.arity]+=1;facts[a.facts]+=1
        records.append(dict(index=index,original_public_sha256=hashlib.sha256(left.encode()).hexdigest(),
            renamed_public_sha256=hashlib.sha256(right.encode()).hexdigest(),
            normalized_public_sha256=hashlib.sha256(a.public.text.encode()).hexdigest(),
            full_fp32_feature_sha256=hashlib.sha256(af.numpy().tobytes()).hexdigest(),
            tokens=al,identifier_occurrences=len(a.identity_positions),
            identities=len(a.original_to_canonical),arity=a.arity,facts=a.facts))
    hashes={};collisions=[]
    for lexeme in sorted(lexemes):
        feature=tuple(lexical_bits(lexeme))
        if feature in hashes:collisions.append([hashes[feature],lexeme])
        hashes[feature]=lexeme
    result=dict(status='pass' if len(records)==1024 and not rejected and not collisions else 'fail',
        attempted=1024,accepted=len(records),rejected=rejected,lexical_collisions=collisions,
        all_accepted_full_fp32_bf16_inputs_equal=True,all_accepted_copy_first_positions_preserved=True,
        all_accepted_original_inventory_restored=True,arities=dict(arities),fact_counts=dict(facts),
        canonical_inventory=list(inventory),all_canonical_tokens_in_public_train=True,
        source_sha256=file_hash(Path(__file__)),
        identity_contract_sha256=file_hash(Path(__import__('topoformer.campaign_semantics_identity_contract',fromlist=['x']).__file__)),
        input_sha256={k:file_hash(p) for k,p in [('original',original),('renamed',renamed),('train',train)]},
        cpu_seconds=time.monotonic()-tick,rows=records,
        scope='public text, supplied renderer grammar and case convention only; no graph/label access, actor instantiation, model forward, or predictions; invariance is programmed, not learned')
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('original','renamed','train','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.original,a.renamed,a.train,a.output)
