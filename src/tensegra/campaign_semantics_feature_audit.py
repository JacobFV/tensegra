"""Finite public feature-multiset identifiability audit, no actor/GPU inference."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import torch
from .campaign_semantics_data import load_cache
from .campaign_semantics import digest
from .semantic_curriculum import encode_text

p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args();torch.set_num_threads(2);start=time.monotonic();seen={k:{} for k in ('lexical_bag','full_fp32','full_bfloat16')};conflicts=collections.Counter();examples=[];lengths=collections.Counter();positions={};total=0
for split in ('train','development'):
    for row in load_cache(Path(a.data)/(split+'.jsonl.gz')):
        features,length=encode_text(row['text']);features=features[0];lengths[length]+=1;total+=1
        variants={'lexical_bag':features[:,:64],'full_fp32':features,'full_bfloat16':features.to(torch.bfloat16).view(torch.int16)}
        positions[str(length)]=dict(fp32_unique=int(features[:,64].unique().numel()),bf16_unique=int(features[:,64].to(torch.bfloat16).unique().numel()),positions=length)
        for variant,values in variants.items():
            key=hashlib.sha256(b''.join(sorted(v.numpy().tobytes() for v in values))).hexdigest();target=row['graph_sha256']
            if key in seen[variant] and seen[variant][key]['target']!=target:
                conflicts[variant]+=1
                if len(examples)<8:examples.append(dict(variant=variant,previous_seed=seen[variant][key]['seed'],seed=row['seed']))
            seen[variant].setdefault(key,dict(target=target,seed=row['seed']))
result=dict(examples=total,lengths=dict(lengths),position_formula=['i/len(tokens)','sin(i)','cos(i)','0'],unique_feature_multisets={k:len(v) for k,v in seen.items()},incompatible_target_collisions={k:conflicts[k] for k in seen},position_uniqueness=positions,collision_examples=examples,data_sha256={s:digest(Path(a.data)/(s+'.jsonl.gz')) for s in ('train','development')},source_sha256=digest(__file__),encoder_sha256=digest(Path(__file__).with_name('thinking_language.py')),cpu_seconds=time.monotonic()-start,scope='Finite 8192 TRAIN+512DEV English unification inputs; BF16 input rounding only, not proof learned projections preserve information.')
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
