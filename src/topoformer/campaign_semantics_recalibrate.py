"""S17 frozen checkpoint, actual-TRAIN128 calibration; no training or DEV fit."""
import argparse,collections,gzip,json,resource,time
from pathlib import Path
import torch
from .campaign_semantics import Dataset,digest,write_gzip,calibrated_evaluation
from . import campaign_semantics as evaluation_base
from .campaign_semantics_data import load_cache
from .semantic_curriculum import SemanticCurriculumActor
from .thinking_language import state_hash,ROLES

def validate(c):
 if c.get('budget_status')!='frozen':raise ValueError('separate source/cap freeze required')
 if c['job'] not in ('profile','main') or c['dev_per_cell']!=({'profile':16,'main':512}[c['job']]):raise ValueError('unregistered evaluation support')
 if [r['arm'] for r in c['runs']]!=['control','mixed'] or c['calibration_count']!=128:raise ValueError('fixed paired128 protocol changed')

def timed_evaluation(model,cal,dev,out,update,calibration_count,evaluation_count):
 """Observe inherited phase boundaries without changing computation or policy."""
 marks={'start':time.monotonic()}
 class ObservedDev:
  def __len__(self):return len(dev)
  def __getitem__(self,index):
   if 'dev_start' not in marks:marks['dev_start']=time.monotonic()
   return dev[index]
 original_save=evaluation_base.np.savez_compressed
 def observed_save(*args,**kwargs):
  if 'export_start' not in marks:marks['export_start']=time.monotonic()
  return original_save(*args,**kwargs)
 evaluation_base.np.savez_compressed=observed_save
 try:result=calibrated_evaluation(model,cal,ObservedDev(),out,update,calibration_count,evaluation_count)
 finally:evaluation_base.np.savez_compressed=original_save
 marks['end']=time.monotonic()
 if not marks['start']<=marks['dev_start']<=marks['export_start']<=marks['end']:raise ValueError('invalid phase timing')
 return result,dict(calibration_forward_fit_train_pack_seconds=marks['dev_start']-marks['start'],dev_forward_metrics_pack_seconds=marks['export_start']-marks['dev_start'],export_and_summary_seconds=marks['end']-marks['export_start'],total_seconds=marks['end']-marks['start'])

def run(c):
 validate(c);start=time.monotonic();torch.set_num_threads(2)
 from .campaign_semantics_recalibrate_freeze import verify
 verify(c,Path(__file__).parent,c['cap_seconds'])
 data=Path(c['data_dir']);selection=json.loads(Path(c['selection']).read_text());vocab=json.loads(Path(c['data_audit']).read_text())['value_vocabulary'];dev_rows=load_cache(data/'development.jsonl.gz')
 dev_rows=[r for a in (3,4) for f in (3,4) for r in [x for x in dev_rows if (x['arity'],x['facts'])==(a,f)][:c['dev_per_cell']]]
 out=Path(c['output_dir']);out.mkdir(exist_ok=False,parents=True);records=[];torch.cuda.reset_peak_memory_stats()
 for entry in c['runs']:
  tick=time.monotonic();arm=entry['arm'];rows=load_cache(data/f'train_{arm}.jsonl.gz');chosen=[];counts=collections.Counter();gold_counts=collections.Counter()
  indices=[item['index'] for item in selection[arm]]
  if len(indices)!=128 or indices!=sorted(set(indices)):raise ValueError('selection must be128distinct TRAIN indices in original order')
  for item in selection[arm]:
   row=rows[item['index']]
   if any(row[k]!=item[k] for k in ('seed','semantic_sha256','alpha_sha256')):raise ValueError('selection identity changed')
   chosen.append(row);counts[f"{row['arity']}x{row['facts']}"]+=1;gold_counts.update(e[2] for e in row['edges'])
  expected={'4x3':64,'4x4':64} if arm=='control' else {'3x3':64,'4x3':32,'4x4':32}
  if dict(counts)!=expected or len(chosen)!=128:raise ValueError('registered selection mixture changed')
  original=json.load(gzip.open(entry['evaluation'],'rt'));original_rows={r['semantic_sha256']:r for r in original['rows']}
  if len(original_rows)!=2048:raise ValueError('incomplete original endpoint')
  checkpoint=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)
  if checkpoint['added_update']!=4096 or checkpoint['update']!=28672 or len(checkpoint['visits'])!=4096 or set(checkpoint['visits'])!={8}:raise ValueError('wrong frozen exposure')
  model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(c['device']);model.load_state_dict(checkpoint['model']);del checkpoint
  if state_hash(model)!=entry['model_state_sha256']:raise ValueError('endpoint model state changed')
  folder=out/arm;folder.mkdir();setup_seconds=time.monotonic()-tick;summary,phase_timing=timed_evaluation(model,Dataset(chosen,vocab),Dataset(dev_rows,vocab),folder,28672,128,len(dev_rows));verification_start=time.monotonic();saved=json.load(gzip.open(folder/summary['artifact'],'rt'))
  for row in saved['rows']:
   old=original_rows[row['semantic_sha256']]
   if row['raw']!=old['raw'] or row['target']!=old['target']:raise ValueError('raw/target replay differs from original S15')
  if state_hash(model)!=entry['model_state_sha256']:raise ValueError('inference mutated model')
  records.append(dict(arm=arm,evaluation=summary,artifact=str(Path(arm)/summary['artifact']),artifact_sha256=digest(folder/summary['artifact']),calibration_npz=str(Path(arm)/saved['calibration_data_artifact']),calibration_npz_sha256=saved['calibration_data_sha256'],calibration_cells=dict(counts),calibration_gold_relation_counts={r:gold_counts[r] for r in ROLES},calibration_absent_relations=[r for r in ROLES if not gold_counts[r]],original_thresholds=original['thresholds'],new_thresholds=saved['thresholds'],original_evaluation_sha256=entry['evaluation_sha256'],checkpoint_sha256=entry['checkpoint_sha256'],model_state_sha256=entry['model_state_sha256'],raw_target_replay_exact=True,examples=len(saved['rows']),setup_seconds=setup_seconds,phase_timing=phase_timing,verification_seconds=time.monotonic()-verification_start,seconds=time.monotonic()-tick))
  del model,original,saved;print(json.dumps(dict(event='endpoint_complete',arm=arm,seconds=records[-1]['seconds'],examples=records[-1]['examples'])),flush=True)
 manifest=dict(config=c,source_sha256=c['source_sha256'],artifacts=records,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_optimizer_or_training=True)
 write_gzip(out/'manifest.json.gz',manifest);return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
