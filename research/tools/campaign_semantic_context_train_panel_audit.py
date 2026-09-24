"""Independent final matched actualTRAIN128 component counts for S18 report."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--archive',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';s=json.loads(a.analysis.read_text());selection=json.loads((base/'s17-calibration-selection/selection.json').read_text())['mixed'];counts={}
for arm in ('original','context','workspace_control'):
 path=base/'s17-calibration-main-v1/mixed/evaluation-u28672.json.gz' if arm=='original' else a.archive/f's18-main-v1/{arm}/matched-u28672/evaluation-u28672.json.gz';d=json.load(gzip.open(path,'rt'));out={}
 for cell in ('3x3','4x3','4x4'):
  out[cell]={}
  for label in ('raw','calibrated'):
   comp=collections.Counter();complete=0;num=0
   for row,item in zip(d['train_rows'],selection,strict=True):
    assert row['seed']==item['seed']
    if item['cell']!=cell:continue
    num+=1;p=row['raw'];g=row['target'];P={i for i,v in enumerate(p['presence']) if v};G={i for i,v in enumerate(g['presence']) if v};ge=edge_set(g);pe={e for e in edge_set({**p,'edges':p['edges'] if label=='raw' else row['calibrated_edges']}) if e[0] in P and e[1] in P};flags=dict(presence=P==G,kind=all(p['kind'][i]==g['kind'][i] for i in G),value=all(p['value'][i]==v for i,v in enumerate(g['value']) if v>=0),copy=all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0),edges=pe==ge,slots=all(p['slots'][i][j]==g['slots'][i][j] for i,j,r in ge));comp.update({k:int(v) for k,v in flags.items()});complete+=all(flags.values())
   expected=s['calibration_overlap_train_panels'][arm]['4096']['matched']['cells'][cell][label];assert expected['complete']==complete and expected['exact_components']==comp and expected['examples']==num;out[cell][label]=dict(examples=num,complete=complete,exact_components=dict(comp))
 counts[arm]=out
out=dict(actual_TRAIN128_components=counts,all18_train_cell_policy_component_summaries_exact=True,analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),report_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),cpu_audit_wall_seconds=time.monotonic()-tick,scope='Same128examples fitted global thresholds: optimistic in-sample diagnostic, not independent TRAINgeneralization. Report fixedgate/conditionaluncertainty/cost/canonicalquery and10passphase-shift limitations reviewed; no new model calls.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='actual_TRAIN128_components'})
