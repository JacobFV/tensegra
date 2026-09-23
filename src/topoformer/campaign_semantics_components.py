"""CPU-only field/count localization from frozen semantic prediction archives."""
import argparse,collections,gzip,json
from pathlib import Path


def summarize(rows):
    result=collections.Counter();sizes=collections.Counter()
    for row in rows:
        pred=row['raw'];gold=row['target'];present=gold['presence'];n=sum(present);pn=sum(pred['presence']);sizes[(n,pn)]+=1
        result['graphs']+=1;result['gold_nodes']+=n;result['predicted_nodes']+=pn
        result['node_true_positive']+=sum(p and g for p,g in zip(pred['presence'],present))
        result['exact_presence_graphs']+=pred['presence']==present
        for key in ('kind','copy','value'):
            mask=present if key=='kind' else [v>=0 for v in gold[key]]
            matches=[p==g for p,g,m in zip(pred[key],gold[key],mask) if m]
            result[key+'_correct']+=sum(matches);result[key+'_total']+=len(matches);result[key+'_exact_graphs']+=all(matches)
    return dict(counts=dict(result),gold_predicted_node_count_pairs=[dict(gold=k[0],predicted=k[1],graphs=v) for k,v in sorted(sizes.items())])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();x=json.load(gzip.open(a.archive,'rt'))
    result=dict(update=x['update'],train=summarize(x['train_rows']),development=summarize(x['rows']))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['counts'] for k,v in result.items() if k!='update'}))
