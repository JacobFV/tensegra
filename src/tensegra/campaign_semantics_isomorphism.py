"""Exact attributed directed graph isomorphism; canonical scores stay primary."""
import argparse,collections,gzip,json,signal,time
from pathlib import Path
import networkx as nx
import torch
from .campaign_semantics import digest,write_gzip
from .semantic_curriculum import unpack_graph
from .thinking_language import KINDS


def attributed_graph(packed):
    graph=unpack_graph(packed);result=nx.DiGraph();copy_kinds={KINDS.index('ident'),KINDS.index('entity')}
    for i in graph['presence'].nonzero().flatten().tolist():
        kind=int(graph['kind'][i]);identity=int(graph['copy'][i]) if kind in copy_kinds else None
        value=int(graph['value'][i]) if kind not in copy_kinds else None
        result.add_node(i,label=(kind,identity,value))
    for i,j in graph['edges'].any(-1).nonzero().tolist():
        if i in result and j in result:
            labels=tuple((r,int(graph['slots'][i,j])) for r in graph['edges'][i,j].nonzero().flatten().tolist())
            result.add_edge(i,j,label=labels)
    return result


def exact_equivalence(a,b):
    if collections.Counter(x['label'] for _,x in a.nodes(data=True))!=collections.Counter(x['label'] for _,x in b.nodes(data=True)):return False,'node_attribute_multiset'
    if collections.Counter(x['label'] for *_,x in a.edges(data=True))!=collections.Counter(x['label'] for *_,x in b.edges(data=True)):return False,'edge_label_multiset'
    return nx.is_isomorphic(a,b,node_match=lambda x,y:x['label']==y['label'],edge_match=lambda x,y:x['label']==y['label']),'exact_isomorphism'


def run(config):
    torch.set_num_threads(2);start=time.monotonic();rows=[]
    def timeout(*_):raise TimeoutError('per-graph exact search budget')
    signal.signal(signal.SIGALRM,timeout)
    for arm in config['arms']:
        archive=json.load(gzip.open(arm['archive'],'rt'))
        for split,key in [('train','train_rows'),('development','rows')]:
            for row in archive[key]:
                gold=attributed_graph(row['target'])
                for policy in ('raw','calibrated'):
                    if time.monotonic()-start>config['cpu_seconds']:raise TimeoutError('declared diagnostic budget exceeded')
                    pred=row['raw'] if policy=='raw' else {**row['raw'],'edges':row['calibrated_edges']}
                    signal.setitimer(signal.ITIMER_REAL,2.)
                    try:answer,reason=exact_equivalence(attributed_graph(pred),gold)
                    except TimeoutError:answer=None;reason='unresolved_timeout'
                    finally:signal.setitimer(signal.ITIMER_REAL,0)
                    rows.append(dict(arm=arm['name'],split=split,policy=policy,seed=row['seed'],isomorphic=answer,reason=reason))
    summary={}
    for arm in config['arms']:
        summary[arm['name']]={}
        for split in ('train','development'):
            summary[arm['name']][split]={policy:dict(collections.Counter(r['reason']+':'+str(r['isomorphic']) for r in rows if r['arm']==arm['name'] and r['split']==split and r['policy']==policy)) for policy in ('raw','calibrated')}
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=False)
    write_gzip(output/'predictions.json.gz',dict(config=config,rows=rows,source_sha256=digest(__file__),archive_sha256={a['name']:digest(a['archive']) for a in config['arms']}))
    (output/'summary.json').write_text(json.dumps(dict(summary=summary,cpu_seconds=time.monotonic()-start,networkx=nx.__version__,semantics='Node kind, relevant canonical visible identity/value, directed relation labels and ordered slots all exact. Only node numbering may vary.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
