"""Guarded S20 complete-main loader; refuses execution until prospective config pin.
No model/checkpoint deserialization. Strict record and AdamW audits are independent.
"""
import argparse,gzip,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
MAIN_CONFIG_SHA=None # Set only from separately reviewed immutable main freeze.
PINS={'S20-analysis.py':'73099b4858ccf87dcc18f080d8455b17ff13d576d698357b9e5805a21d2bd13a','S18-analysis.py':'28edae0137f2c0159f0f4648698d768b24f150d170948e9089d43ce7011e1687','S19-analysis.py':'c42ac5868230e2e77a08c4df3f0157bb5a77f463215b6ccc908b82e5a5816155'}
STREAM='85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9'
PAIRS='e2fa3128d6d68928310e2b49d3d29d31274eca131c8607ffb9c6483df6a846ed'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,h):require(isinstance(h,str) and sha(p)==h,'missing pin/hash mismatch: '+str(p));return Path(p)
def load(p):return json.load(gzip.open(p,'rt'))
def helper(name):
 p=Path(__file__).with_name(name);checked(p,PINS[name]);s=importlib.util.spec_from_file_location(name.replace('-','_').replace('.','_'),p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def hash_string(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def state_guard(r,c):
 seed=r['seed'];arm=r['arm'];record=arm=='record';parent=c['parents'][str(seed)]
 require(r['added_update']==4096 and r['update']==(4096 if record else 28672) and r['added_presentations']==32768,'endpoint/exposure')
 require(len(r['visits'])==4096 and all(type(v)is int and v==8 for v in r['visits']),'eight visits')
 require(r['construction_sequence_sha256']==STREAM and r['pair_sequence_sha256']==(None if record else PAIRS),'fixed training streams')
 require((r['optimizer_tokens'],r['optimizer_nodes'],r['optimizer_edges'],r['optimizer_records'])==(1720320,988008,2208616,3229392 if record else None),'exact exposure counters')
 require(r['parameters']==(62677315 if record else 57853781),'parameter count')
 require(r['inherited_presentations']==(0 if record else 196608) and r['initial_optimizer_states']==(0 if record else 98) and r['inherited_optimizer_steps']==([] if record else [24576.]),'inherited history')
 require(r['parent_checkpoint_sha256']==(None if record else parent['checkpoint']['sha256']),'parent checkpoint')
 if not record:require(r['initial_state_sha256']==parent['model_state_sha256'],'parent initial state')
 require(all(hash_string(r[k]) for k in ('initial_state_sha256','final_state_sha256','checkpoint_sha256')),'checkpoint/state hashes')
 require(r['checkpoint']==f"model-u{r['update']}.pt" and r['evaluation_population']=='confirmation' and r['worst_case_timing'] is None,'endpoint schema')
 require([x['added_update'] for x in r['losses']]==list(range(128,4097,128)),'complete loss curve')
 require(all(x['seed']==seed and x['arm']==arm and math.isfinite(x['loss']) for x in r['losses']),'loss metadata')
def identity_guard(rows,expected,targets):
 indexed={r['semantic_sha256']:r for r in rows};require(len(indexed)==len(rows)==len(expected) and set(indexed)==set(expected),'event IDs')
 for k,r in indexed.items():
  require(r['seed']==expected[k]['seed'] and r['graph_sha256']==expected[k]['graph_sha256'],'event metadata')
  if k in targets:require(targets[k]==r['target'],'cross-arm target disagreement')
  else:targets[k]=r['target']
 return indexed

def main(a):
 # Check closed top-level completion/config/source BEFORE parsing confirmation.
 checked(a.config,MAIN_CONFIG_SHA);c=json.loads(a.config.read_text());m=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text());core=helper('S20-analysis.py');old=helper('S18-analysis.py');record_helper=helper('S19-analysis.py');components=old.helper()
 require(c==m['config'] and c['budget_status']=='frozen' and c['job']=='main','frozen main')
 require(c['seeds']==list(core.SEEDS) and c['arms']==['original','context','record'] and c['updates']==4096 and c['evaluation_per_cell']==512 and c['final_only'],'study dimensions')
 require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==MAIN_CONFIG_SHA and receipt['cap_seconds']==c['cap_seconds'] and receipt['wrapper_sha256']==c['source_sha256']['campaign_semantics_s20_launch.py'],'completed wrapper')
 for name,h in c['source_sha256'].items():checked(a.source/name,h)
 require([(r['seed'],r['arm']) for r in m['results']]==[(s,arm) for s in core.SEEDS for arm in c['arms']],'complete ordered nine arms')
 for r in m['results']:state_guard(r,c)
 require(set(c['inputs'])=={'train','selection','vocabulary_audit','confirmation'},'no DEV or foreign input')
 confirmation=[json.loads(line) for line in gzip.open(checked(a.confirmation,c['inputs']['confirmation']['sha256']),'rt')];train=[json.loads(line) for line in gzip.open(checked(a.train,c['inputs']['train']['sha256']),'rt')];selection=json.loads(checked(a.selection,c['inputs']['selection']['sha256']).read_text())['mixed']
 require(len(confirmation)==2048 and len(train)==4096 and len(selection)==128,'cache supports')
 expected={r['semantic_sha256']:r for r in confirmation};panel=[train[r['index']] for r in selection];expected_train={r['semantic_sha256']:r for r in panel}
 require(len(expected)==2048 and len(expected_train)==128,'cache duplicates')
 require([r['index'] for r in selection]==sorted({r['index'] for r in selection}),'selection ordering')
 for chosen,r in zip(selection,panel):require(all(chosen[k]==r[k] for k in ('seed','semantic_sha256','alpha_sha256')),'actual TRAIN selection')
 cell=lambda r:f"{r['arity']}x{r['facts']}"
 require({x:sum(cell(r)==x for r in confirmation) for x in core.CELLS}==dict.fromkeys(core.CELLS,512),'confirmation cell support')
 require({x:sum(cell(r)==x for r in panel) for x in core.KNOWN}=={'3x3':64,'4x3':32,'4x4':32},'TRAIN mixture')
 flags={s:{} for s in core.SEEDS};summaries={};inventory={};targets={};train_targets={}
 for r in m['results']:
  seed=r['seed'];arm=r['arm'];key=f'{seed}/{arm}';folder=a.main/f'seed-{seed}'/arm;require(load(folder/'manifest.json.gz')==r,'per-arm/top manifest mismatch');inventory[key+'/manifest']=dict(path=str(folder/'manifest.json.gz'),sha256=sha(folder/'manifest.json.gz'));summaries[key]=dict(history={k:r[k] for k in ('parameters','inherited_presentations','inherited_optimizer_steps','added_presentations','parent_checkpoint_sha256')},losses=r['losses'],timings={k:r[k] for k in ('setup_seconds','training_seconds','evaluation_seconds','checkpoint_export_seconds')},checkpoint={k:r[k] for k in ('checkpoint','checkpoint_sha256','initial_state_sha256','final_state_sha256')})
  flags[seed][arm]={}
  if arm=='record':
   for pop,entry,exp,tg in [('train',r['train'],expected_train,train_targets),('confirmation',r['evaluation'],expected,targets)]:
    path=checked(folder/entry['artifact'],entry['sha256']);d=load(path);record_helper.entry_guard(entry,d);require(d['update']==4096 and d['population']==pop and d['teacher_forced']==entry['teacher_forced'],'record metadata');indexed=identity_guard(d['rows'],exp,tg)
    require(set(d['teacher_forced'])==set(record_helper.FIELDS),'TF fields')
    for f in d['teacher_forced'].values():require(f['count']>0 and 0<=f['correct']<=f['count'] and math.isclose(f['mean_loss'],f['loss_sum']/f['count']) and math.isclose(f['accuracy'],f['correct']/f['count']),'TF arithmetic')
    require(math.isclose(d['teacher_forced_loss'],sum(v['mean_loss'] for v in d['teacher_forced'].values())/8),'TF eight means')
    for k,row in indexed.items():
     require(row['cell']==cell(exp[k]),'record cell identity')
     if row['valid']:
      comp=components.components(components.unpack(row['prediction']),components.unpack(row['target']));require(comp==row['exact_components'] and all(comp.values())==row['complete']==bool(row['metrics']['semantic_equivalence']),'record components')
     else:require(not row['complete'] and not any(row['exact_components'].values()) and row['prediction'] is None and row['metrics'] is None and bool(row['reason']),'invalid credit')
    summary=record_helper.summarize(d['rows']);require(summary['complete']==entry['complete'] and summary['valid']==entry['valid'] and summary['invalid_reasons']==entry['invalid_reasons'],'record summaries');summary['teacher_forced']=d['teacher_forced'];summary['cells']={x:record_helper.summarize([row for k,row in indexed.items() if cell(exp[k])==x]) for x in (core.KNOWN if pop=='train' else core.CELLS)};summaries[key][pop]=summary;inventory[key+'/'+pop]=dict(path=str(path),sha256=sha(path))
    if pop=='confirmation':flags[seed][arm]['categorical']={x:{k:bool(row['complete']) for k,row in indexed.items() if cell(exp[k])==x} for x in core.CELLS}
  else:
   entry=r['matched'];path=checked(folder/r['matched_directory']/entry['artifact'],entry['sha256']);d=load(path);require(d['update']==28672 and r['matched_directory']=='matched-u28672','workspace endpoint');old.evaluation_guard(d,entry,path.name);npz=checked(path.parent/d['calibration_data_artifact'],d['calibration_data_sha256'])
   with np.load(npz,allow_pickle=False) as z:old.calibration_array_guard(d,z)
   require([row['seed'] for row in d['train_rows']]==[row['seed'] for row in panel],'global actual TRAIN128 calibration')
   inventory[key+'/matched']=dict(path=str(path),sha256=sha(path),calibration_path=str(npz),calibration_sha256=sha(npz));summaries[key]['thresholds']=d['thresholds'];summaries[key]['calibration']=d['calibration']
   for pop,rows,exp,tg in [('train',[dict(row,semantic_sha256=panel[i]['semantic_sha256']) for i,row in enumerate(d['train_rows'])],expected_train,train_targets),('confirmation',d['rows'],expected,targets)]:
    indexed=identity_guard(rows,exp,tg);summaries[key][pop]={}
    for policy in core.POLICIES:
     ff={};cc={x:dict(examples=0,complete=0,exact_components=dict.fromkeys(record_helper.COMPONENTS,0)) for x in (core.KNOWN if pop=='train' else core.CELLS)}
     for i,row in enumerate(rows):
      pred=components.unpack(row['raw']);gold=components.unpack(row['target'])
      if policy=='matched':pred['edges']=components.edge(row['calibrated_edges'])
      comp=components.components(pred,gold);exact=all(comp.values());metric=d['train_metrics'][i]['raw' if policy=='raw' else 'calibrated'] if pop=='train' else row['raw_metrics' if policy=='raw' else 'calibrated_metrics'];require(exact==bool(metric['semantic_equivalence']),'workspace raw/cal components');event=row['semantic_sha256'];ff[event]=exact;out=cc[cell(exp[event])];out['examples']+=1;out['complete']+=exact
      for k,v in comp.items():out['exact_components'][k]+=int(v)
     summaries[key][pop][policy]=cc
     if pop=='confirmation':flags[seed][arm][policy]={x:{k:bool(v) for k,v in ff.items() if cell(exp[k])==x} for x in core.CELLS}
 out=core.aggregate(flags);out.update(main_config_sha256=MAIN_CONFIG_SHA,manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),loader_sha256=sha(Path(__file__)),helper_pins=PINS,artifact_inventory=inventory,arms=summaries,process_seconds=m['process_seconds'],audit_scope='Packed components/calibration metadata checked here; independent raw codec/calibration threshold replay and remote model/AdamW checkpoint byte audit remain required. No model calls or further policy fitting.')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('config','main','receipt','source','confirmation','train','selection','output'):p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
