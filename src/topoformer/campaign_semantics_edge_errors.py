"""CPU-only residual edge errors by relation and endpoint kinds; no model fitting."""
import argparse,collections,gzip,json
from pathlib import Path
import torch
from .semantic_curriculum import unpack_graph
from .thinking_language import ROLES


def summarize(rows,decoded):
    counts=collections.Counter()
    for row in rows:
        pred=unpack_graph(row['raw']);gold=unpack_graph(row['target'])
        if decoded=='calibrated':pred['edges']=unpack_graph({**row['raw'],'edges':row['calibrated_edges']})['edges']
        active=pred['presence'][:,None,None]&pred['presence'][None,:,None]
        actual=pred['edges']&active
        for error,mask in [('false_positive',actual&~gold['edges']),('false_negative',~actual&gold['edges'])]:
            for i,j,r in mask.nonzero().tolist():
                counts[(error,ROLES[r],int(pred['kind'][i]),int(pred['kind'][j]),int(gold['kind'][i]),int(gold['kind'][j]),bool(gold['presence'][i]),bool(gold['presence'][j]))]+=1
    return [dict(error=k[0],relation=k[1],predicted_source_kind=k[2],predicted_target_kind=k[3],gold_source_kind=k[4],gold_target_kind=k[5],gold_source_present=k[6],gold_target_present=k[7],count=v) for k,v in sorted(counts.items())]


if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();x=json.load(gzip.open(a.archive,'rt'))
    result={group:{mode:summarize(rows,mode) for mode in ('raw','calibrated')} for group,rows in [('train',x['train_rows']),('development',x['rows'])]}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
