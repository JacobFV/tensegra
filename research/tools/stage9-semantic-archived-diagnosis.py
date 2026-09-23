"""Independent archived-prediction diagnosis (no model inference/selection)."""
import argparse,base64,gzip,json
from collections import Counter,defaultdict
from pathlib import Path


def edges(record):
    spec=record['edges'];n,m,r=spec['shape'];out=set()
    for byte_index,byte in enumerate(base64.b64decode(spec['packed_b64'])):
        while byte:
            bit=(byte&-byte).bit_length()-1;flat=byte_index*8+bit;byte&=byte-1
            if flat<n*m*r:out.add((flat//(m*r),(flat//r)%m,flat%r))
    return out


def diagnose(paths):
    groups=defaultdict(Counter);rows=0
    for path in paths:
      with gzip.open(path,'rt') as stream:
       for line in stream:
        row=json.loads(line);p=row['prediction'];g=row['target'];ge=edges(g);pe={e for e in edges(p) if p['presence'][e[0]] and p['presence'][e[1]]}
        active=[i for i,x in enumerate(g['presence']) if x];pairs={(i,j) for i,j,_ in ge}
        good=dict(presence=p['presence']==g['presence'],kind=all(p['kind'][i]==g['kind'][i] for i in active),value=all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0) and 0 not in g['value'],copy=all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0),edges=pe==ge,slots=all(p['slots'][i][j]==g['slots'][i][j] for i,j in pairs))
        assert int(all(good.values()))==row['metrics']['semantic_equivalence']
        assert len(pe&ge)==row['metrics']['typed_edge']['true_positive']
        assert len(pe)==row['metrics']['typed_edge']['predicted_count']
        ordered_p={e for e in pe if p['slots'][e[0]][e[1]]>=0};ordered_g={e for e in ge if g['slots'][e[0]][e[1]]>=0}
        tp=sum(p['slots'][i][j]==g['slots'][i][j] for i,j,_ in ordered_p&ordered_g)
        assert tp==row['metrics']['ordered_edge']['true_positive']
        keys=[(path.parent.name,path.name.startswith('calibrated'),row['context']['seed'],row['language'],row['renamed'],row['lesson'])]
        for key in keys:
            c=groups[key];c['examples']+=1;c['exact']+=all(good.values());c['edge_tp']+=len(pe&ge);c['edge_pred']+=len(pe);c['edge_gold']+=len(ge);c['ordered_tp']+=tp;c['ordered_pred']+=len(ordered_p);c['ordered_gold']+=len(ordered_g)
            for component,correct in good.items():
                c['complete_'+component]+=correct
                # Replacing presence also changes edge masking; report component logical ceiling separately.
                c['other_components_exact_if_'+component+'_corrected']+=all(v for k,v in good.items() if k!=component)
            c['correct_edge_slot_count']+=sum(p['slots'][i][j]==g['slots'][i][j] for i,j,_ in pe&ge);c['correct_edge_count']+=len(pe&ge)
        rows+=1
    return dict(kind='independent numerical reconstruction from archived outputs, no inference rerun',rows=rows,groups=[dict(corpus=k[0],calibrated=k[1],seed=k[2],language=k[3],renamed=k[4],lesson=k[5],counts=dict(v)) for k,v in sorted(groups.items())],limitations=['Oracle component correctness ceilings are logical remaining-error counts, not retrained models or proof a component is learnable.','Presence replacement may change edge masking; its logical ceiling holds the archived masked edge component fixed.','Historical test outcomes are diagnosis only, never Stage9 model selection.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('directories',nargs='+');a=p.parse_args();paths=[]
    for directory in a.directories:
        paths.extend(Path(directory).glob('*predictions-semantic-seed*-p100000.jsonl.gz'))
    Path(a.output).write_text(json.dumps(diagnose(sorted(paths)),indent=2)+'\n')
