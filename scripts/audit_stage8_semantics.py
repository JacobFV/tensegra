"""Recompute semantic metrics from lossless raw bitmap exports, without Torch."""
import argparse
import base64
import gzip
import json
from pathlib import Path


def edge_bits(graph):
    e=graph['edges']
    if e['bitorder']!='little': raise ValueError('unknown edge bit order')
    n,m,r=e['shape'];return int.from_bytes(base64.b64decode(e['packed_b64']),'little'),n,m,r


def reconstruct(pred,gold):
    p,n,m,relations=edge_bits(pred);g,gn,gm,gr=edge_bits(gold)
    assert (n,m,relations)==(gn,gm,gr)
    present=pred['presence'];mask=gold['presence'];role_mask=(1<<relations)-1
    filtered=ordered_p=ordered_g=ordered_tp=0;slot_correct=True
    for i in range(n):
        for j in range(m):
            shift=(i*m+j)*relations
            pe=(p>>shift)&role_mask if present[i] and present[j] else 0
            ge=(g>>shift)&role_mask
            filtered|=pe<<shift
            ps,gs=pred['slots'][i][j],gold['slots'][i][j]
            if ps>=0:ordered_p+=pe.bit_count()
            if gs>=0:ordered_g+=ge.bit_count()
            if ps==gs and gs>=0:ordered_tp+=(pe&ge).bit_count()
            if ge and ps!=gs:slot_correct=False
    def count(tp,np,ng):return dict(true_positive=tp,predicted_count=np,gold_count=ng)
    active=[i for i,x in enumerate(mask) if x]
    copied=[i for i,x in enumerate(gold['copy']) if x>=0]
    valued=[i for i,x in enumerate(gold['value']) if x>=0]
    type_accuracy=sum(pred['kind'][i]==gold['kind'][i] for i in active)/len(active)
    copying=sum(pred['copy'][i]==gold['copy'][i] for i in copied)/len(copied) if copied else 1.
    equiv=sum((pred['copy'][i]==pred['copy'][j])==(gold['copy'][i]==gold['copy'][j]) for i in copied for j in copied)/len(copied)**2 if copied else 1.
    exact=bool(0 not in gold['value'] and present==mask and type_accuracy==1 and filtered==g and
               all(pred['value'][i]==gold['value'][i] for i in valued) and copying==1 and slot_correct)
    return dict(node=count(sum(a and b for a,b in zip(present,mask)),sum(present),sum(mask)),
                typed_edge=count((filtered&g).bit_count(),filtered.bit_count(),g.bit_count()),
                ordered_edge=count(ordered_tp,ordered_p,ordered_g),node_type_accuracy=type_accuracy,
                identity_copy_accuracy=copying,entity_equivalence=equiv,semantic_equivalence=float(exact))


def audit(paths):
    errors=[];rows=0
    for path in paths:
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);calculated=reconstruct(row['prediction'],row['target']);saved=row['metrics']
                for key,value in calculated.items():
                    if isinstance(value,dict):
                        for k,v in value.items():
                            if saved[key][k]!=v:errors.append([str(path),rows,key,k])
                    elif abs(saved[key]-value)>2e-6:errors.append([str(path),rows,key,saved[key],value])
                rows+=1
    return dict(rows=rows,errors=errors,passed=rows>0 and not errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+',type=Path);parser.add_argument('--output',type=Path)
    a=parser.parse_args();r=audit(a.paths)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)
