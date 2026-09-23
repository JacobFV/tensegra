"""Independent semantic component-replacement diagnostic reconstruction."""
import argparse,gzip,json
from pathlib import Path
from audit_stage11_semantic_text import edge_set

def exact(pred,gold):
 if 0 in gold["value"] or pred["presence"]!=gold["presence"]:return False
 for key in ("kind","value","copy"):
  indices=[i for i,v in enumerate(gold["presence"]) if v] if key=="kind" else [i for i,v in enumerate(gold[key]) if v>=0]
  if any(pred[key][i]!=gold[key][i] for i in indices):return False
 ge=edge_set(gold);pe={e for e in edge_set(pred) if pred["presence"][e[0]] and pred["presence"][e[1]]}
 return pe==ge and all(pred["slots"][i][j]==gold["slots"][i][j] for i,j,r in ge)

def load(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def audit(archive,records):
 x=load(archive); indexed={(r['seed'],r['graph_seed'],r['decoder']):r for r in load(records)['rows']};n=0;totals={}
 for run in x['runs']:
  rows=run['curves'][-1]['rows'] if 'curves' in run else run['rows']
  for row in rows:
   gold=row['target']
   for policy in ('raw','calibrated'):
    pred=row[policy];record=indexed[run['seed'],row['graph_seed'],policy]
    for name in ('presence','kind','value','copy','edges','slots','all_node_attributes'):
     keys=('presence','kind','value','copy') if name=='all_node_attributes' else (name,)
     replacement=dict(pred,**{k:gold[k] for k in keys})
     got=exact(replacement,gold);assert got==record['oracle_component_exact'][name]
     key=f"{run['seed']}/{policy}/{name}";totals[key]=totals.get(key,0)+int(got);n+=1
 return dict(replacements_verified=n,records_verified=len(indexed),ceilings=totals,scope='Privileged archived component substitutions, not inference or deployable performance.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('records',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=audit(a.archive,a.records);a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='ceilings'})
