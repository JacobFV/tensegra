"""Privileged component-replacement ceilings from archived predictions only."""
import argparse,gzip,json,time
from pathlib import Path
import torch
from .semantic_curriculum import unpack_graph
from .semantic_scaling import metrics
from .campaign_semantics_components import summarize


def run(path,output):
    tick=time.monotonic();torch.set_num_threads(2);archive=json.load(gzip.open(path,'rt'));result=dict(update=archive['update'],scope='Gold replacement is a privileged archived diagnostic, not an acquired interface or deployable decoder',groups={})
    for group,key in [('train','train_rows'),('development','rows')]:
        rows=archive[key];ceiling=[]
        for row in rows:
            gold=unpack_graph(row['target'])
            for decoder in ('raw','calibrated'):
                packed=row['raw'] if decoder=='raw' else {**row['raw'],'edges':row['calibrated_edges']};pred=unpack_graph(packed)
                values={k:metrics({**pred,k:gold[k]},gold)['semantic_equivalence'] for k in ('presence','kind','value','copy','edges','slots')}
                values['all_node_attributes']=metrics({**pred,**{k:gold[k] for k in ('presence','kind','value','copy')}},gold)['semantic_equivalence']
                ceiling.append(dict(seed=row['seed'],decoder=decoder,exact=metrics(pred,gold)['semantic_equivalence'],replacements=values))
        result['groups'][group]=dict(fields=summarize(rows),rows=ceiling,aggregate={d:{k:sum(r['replacements'][k] for r in ceiling if r['decoder']==d) for k in values} for d in ('raw','calibrated')})
    result['cpu_seconds']=time.monotonic()-tick;Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({g:r['aggregate'] for g,r in result['groups'].items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('output');a=p.parse_args();run(a.archive,a.output)
