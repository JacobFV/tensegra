"""Archived component replacement ceilings and per-relation acquisition errors."""
import argparse,gzip,json
from pathlib import Path
from topoformer.semantic_curriculum import unpack_graph
from topoformer.semantic_scaling import metrics
from topoformer.thinking_language import ROLES

p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
x=json.load(gzip.open(a.archive,'rt'));rows=[]
for run in x['runs']:
    point=run['curves'][-1]
    for record in point['rows']:
      gold=unpack_graph(record['target'])
      for decoder in ('raw','calibrated'):
        pred=unpack_graph(record[decoder]);active=pred['presence'][:,None,None]&pred['presence'][None,:,None]
        edge=pred['edges']&active;per_relation=[]
        for r,name in enumerate(ROLES):
            fp=(edge[:,:,r]&~gold['edges'][:,:,r]).nonzero().tolist();fn=(~edge[:,:,r]&gold['edges'][:,:,r]).nonzero().tolist()
            per_relation.append(dict(relation=name,false_positive=len(fp),false_negative=len(fn),first_false_positive=fp[:12],first_false_negative=fn[:12]))
        ceilings={key:metrics({**pred,key:gold[key]},gold)['semantic_equivalence'] for key in ('presence','kind','value','copy','edges','slots')}
        ceilings['all_node_attributes']=metrics({**pred,**{key:gold[key] for key in ('presence','kind','value','copy')}},gold)['semantic_equivalence']
        known_slots=gold['edges'].any(-1)
        counts=dict(presence=int(pred['presence'].ne(gold['presence']).sum()),kind=int(pred['kind'][gold['presence']].ne(gold['kind'][gold['presence']]).sum()),value=int(pred['value'][gold['value'].ge(0)].ne(gold['value'][gold['value'].ge(0)]).sum()),copy=int(pred['copy'][gold['copy'].ge(0)].ne(gold['copy'][gold['copy'].ge(0)]).sum()),slots_on_gold_edges=int(pred['slots'][known_slots].ne(gold['slots'][known_slots]).sum()))
        rows.append(dict(seed=run['seed'],update=point['update'],graph_seed=record['graph_seed'],decoder=decoder,component_error_counts=counts,edge_errors_by_relation=per_relation,oracle_component_exact=ceilings,exact=metrics(pred,gold)['semantic_equivalence']))
a.output.write_text(json.dumps(dict(scope='Archived predictions only; replacing components with gold is a privileged diagnostic, not deployable performance or a new fit.',rows=rows),indent=2)+'\n')
