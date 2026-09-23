"""Privileged component ceilings reconstructed from saved semantic graphs."""
import argparse,gzip,json,time
from pathlib import Path
from audit_stage11_semantic_replacements import exact
p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();x=json.load(gzip.open(a.evaluation,'rt'));result=[]
replacements={k:(k,)for k in ('presence','kind','value','copy','edges','slots')};replacements.update(all_node_attributes=('presence','kind','value','copy'),edges_and_slots=('edges','slots'))
for split,key in [('train','train_rows'),('development','rows')]:
 for policy in ('raw','calibrated'):
  counts={k:0 for k in replacements};none=0
  for row in x[key]:
   pred=row['raw']if policy=='raw'else dict(row['raw'],edges=row['calibrated_edges']);gold=row['target'];none+=exact(pred,gold)
   for name,fields in replacements.items():counts[name]+=exact(dict(pred,**{f:gold[f]for f in fields}),gold)
  result.append(dict(split=split,policy=policy,examples=len(x[key]),original_exact=none,replacement_exact=counts))
out=dict(cpu_audit_wall_seconds=time.monotonic()-start,results=result,scope='Posthoc privileged replacements, not inference inputs or deployable performance.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
