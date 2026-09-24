"""Prospective S21 complete paired-main analysis; no model or sealed cache reads."""
import argparse,collections,gzip,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
CONFIG='5a13908ec22e9ad47350d7d85433b46291dada4552a474300fcef57871f30db3'
PINS={'S19-analysis.py': 'c42ac5868230e2e77a08c4df3f0157bb5a77f463215b6ccc908b82e5a5816155', 'S21-decisions.py': '1ded2fd049cedaad71711b9bde60002e59ac2d4a9cafed62a801b550ec29de30', 'S17-analyze.py': '13a8c12a014f97b95b65b221457ade75be318c0cb2f76e24234c8391e12d8355'}
CELLS=('2x4','3x3','3x4','4x3','4x4','5x3','5x4');STEPS=(0,1024,2048,4096)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,h):require(isinstance(h,str) and sha(p)==h,'hash mismatch '+str(p));return Path(p)
def load(p):return json.load(gzip.open(p,'rt'))
def helper(name):
 p=Path(__file__).with_name(name);checked(p,PINS[name]);spec=importlib.util.spec_from_file_location(name.replace('.','_').replace('-','_'),p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def digest_guard(value):require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'digest format')
def arm_guard(r,c,arm):
 require(r['arm']==arm and r['seed']==2101 and r['parameters']==62677315 and r['inherited_presentations']==0 and r['added_presentations']==32768,'arm/history')
 require(len(r['visits'])==4096 and all(type(n)is int and n==8 for n in r['visits']),'complete visits')
 require(r['construction_sequence_sha256']==c['expected_construction_sequence_sha256'] and r['exposure']=={k:v*8 for k,v in c['expected_epoch_exposures'][arm].items()},'stream/exposure')
 require([v['update'] for v in r['curves']]==list(STEPS),'four curves')
 require(r['curves'][0]['model_state_sha256']==r['initial_state_sha256'] and r['curves'][-1]['model_state_sha256']==r['final_state_sha256'],'initial/final linkage')
 for curve in r['curves']:
  require(curve['presentations']==curve['update']*8 and curve['exposure']=={k:v*(curve['update']//512) for k,v in c['expected_epoch_exposures'][arm].items()},'curve exposure')
  require(curve['checkpoint']==f"model-u{curve['update']}.pt",'checkpoint name');digest_guard(curve['checkpoint_sha256']);digest_guard(curve['model_state_sha256'])
 require(r['worst_case_timing'] is None and [x['update'] for x in r['losses']]==list(range(128,4097,128)),'main/no stress/loss curves')
 for row in r['losses']:
  require(row['arm']==arm and math.isfinite(row['loss']) and set(row['field_loss_sums'])==set(row['field_counts'])=={'type','kind','value','copy','source','target','role','slot'},'eight training fields')
  require(all(type(v)is int and v>=0 for v in row['field_counts'].values()) and all(math.isfinite(v) for v in row['field_loss_sums'].values()),'loss support')
def row_guard(row,expected):
 require(row['semantic_sha256']==expected['semantic_sha256'] and row['seed']==expected['seed'] and row['graph_sha256']==expected['graph_sha256'] and row['cell']==f"{expected['arity']}x{expected['facts']}",'public cache identity')
 require(type(row['valid'])is bool and type(row['complete'])is bool and (not row['complete'] or row['valid']),'strict validity flags')
 require((row['reason'] is None) if row['valid'] else isinstance(row['reason'],str) and bool(row['reason']),'validity reason')
def summarize(rows,h):
 out=h.summarize(rows);out['failure_semantic_ids']=[r['semantic_sha256'] for r in sorted(rows,key=lambda r:r['semantic_sha256']) if not r['complete']];return out

def main(a):
 c=json.loads(checked(a.config,CONFIG).read_text());m=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text());h=helper('S19-analysis.py');g=helper('S21-decisions.py');components=helper('S17-analyze.py')
 require(m['config']==c and c['job']=='main' and c['budget_status']=='frozen' and c['cap_seconds']==1770,'frozen main')
 require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==CONFIG and receipt['cap_seconds']==1770 and receipt['wrapper_sha256']==c['source_sha256']['campaign_semantics_s21_launch.py'],'closed receipt')
 for name,v in c['source_sha256'].items():checked(a.source/name,v)
 require([r['arm'] for r in m['results']]==['original','broad'],'complete ordered pair')
 for r in m['results']:arm_guard(r,c,r['arm'])
 require(m['results'][0]['initial_state_sha256']==m['results'][1]['initial_state_sha256'],'paired initialization')
 require(set(c['inputs'])=={'train_original','train_broad','development','panels','audit','vocabulary_audit'},'no sealed confirmation input')
 paths={'train_original':a.train_original,'train_broad':a.train_broad,'development':a.development,'panels':a.panels,'audit':a.audit,'vocabulary_audit':a.vocabulary_audit}
 for name,p in paths.items():checked(p,c['inputs'][name]['sha256'])
 panels=json.loads(a.panels.read_text());cache={name:list(map(json.loads,gzip.open(paths[name],'rt'))) for name in ('train_original','train_broad','development')};dev={r['semantic_sha256']:r for r in cache['development']}
 require(len(dev)==len(cache['development'])==3584 and collections.Counter(f"{r['arity']}x{r['facts']}" for r in dev.values())==dict.fromkeys(CELLS,512),'seven DEV512')
 results={};flags={};targets={};inventory={};state_inventory={}
 for arm_result in m['results']:
  arm=arm_result['arm'];folder=a.main/arm;require(load(folder/'manifest.json.gz')==arm_result,'per-arm manifest');train=cache['train_'+arm];selection=panels[arm]
  require(len(train)==4096 and len(selection)==128 and len({s['index'] for s in selection})==128,'TRAIN/panel support')
  selected=[]
  for s in selection:
   r=train[s['index']];require(all(s[k]==r[k] for k in ('seed','semantic_sha256','alpha_sha256')),'TRAIN selection');selected.append(r)
  trainmap={r['semantic_sha256']:r for r in selected};require(len(trainmap)==128,'unique panel')
  wanted={'3x3':64,'4x3':32,'4x4':32} if arm=='original' else {'3x3':32,'4x3':16,'4x4':16,'2x4':16,'5x3':24,'5x4':24}
  require(collections.Counter(f"{r['arity']}x{r['facts']}" for r in selected)==wanted,'panel allocation')
  results[arm]={};flags[arm]={};state_inventory[arm]=dict(initial=arm_result['initial_state_sha256'],final=arm_result['final_state_sha256'],checkpoints=[])
  for curve in arm_result['curves']:
   step=curve['update'];results[arm][str(step)]={};state_inventory[arm]['checkpoints'].append({k:curve[k] for k in ('update','checkpoint','checkpoint_sha256','model_state_sha256','presentations','exposure')})
   for pop,expected in [('train',trainmap),('development',dev)]:
    entry=curve[pop];path=checked(folder/entry['artifact'],entry['sha256']);d=load(path);require(d['update']==step and d['population']==pop,'artifact endpoint');h.entry_guard(entry,d);require(d['teacher_forced']==entry['teacher_forced'] and set(d['teacher_forced'])==set(h.FIELDS),'teacher forced fields')
    for f in d['teacher_forced'].values():require(f['count']>0 and 0<=f['correct']<=f['count'] and math.isclose(f['mean_loss'],f['loss_sum']/f['count']) and math.isclose(f['accuracy'],f['correct']/f['count']),'TF arithmetic')
    require(math.isclose(d['teacher_forced_loss'],sum(f['mean_loss'] for f in d['teacher_forced'].values())/8),'eight means')
    rows=h.indexed(d['rows']);require(set(rows)==set(expected),'population IDs')
    for event,row in rows.items():
     row_guard(row,expected[event]);key=(pop,event)
     if key in targets:require(targets[key]==row['target'],'paired/curve target')
     else:targets[key]=row['target']
     require(set(row['exact_components'])==set(h.COMPONENTS) and all(type(v)is bool for v in row['exact_components'].values()),'component flags')
     if row['valid']:
      comp=components.components(components.unpack(row['prediction']),components.unpack(row['target']));require(comp==row['exact_components'] and all(comp.values())==row['complete']==bool(row['metrics']['semantic_equivalence']),'record component replay')
     else:require(row['prediction'] is None and row['metrics'] is None and not any(row['exact_components'].values()),'invalid credit')
    summary=summarize(list(rows.values()),h);require(summary['valid']==entry['valid'] and summary['complete']==entry['complete'] and summary['invalid_reasons']==entry['invalid_reasons'],'entry summaries');summary.update(cells={cell:summarize([r for r in rows.values() if r['cell']==cell],h) for cell in (CELLS if pop=='development' else wanted)},teacher_forced=d['teacher_forced'],teacher_forced_loss=d['teacher_forced_loss']);results[arm][str(step)][pop]=summary;inventory[f'{arm}/{step}/{pop}']=dict(path=str(path),sha256=sha(path))
    if pop=='development':flags[arm][step]={cell:{k:row['complete'] for k,row in rows.items() if row['cell']==cell} for cell in CELLS}
 paired={};ci={};rng=np.random.default_rng(21021)
 for cell in CELLS:
  ids=sorted(flags['original'][4096][cell]);draws=rng.integers(0,512,size=(10000,512))
  for step in STEPS:
   before=np.array([flags['original'][step][cell][i] for i in ids],dtype=np.int8);after=np.array([flags['broad'][step][cell][i] for i in ids],dtype=np.int8);key=f'{cell}/{step}';paired[key]=dict(transitions_original_to_broad=h.paired(before,after),delta_complete=int(after.sum()-before.sum()),component_delta={k:results['broad'][str(step)]['development']['cells'][cell]['exact_components'][k]-results['original'][str(step)]['development']['cells'][cell]['exact_components'][k] for k in h.COMPONENTS})
   if step==4096:ci[cell]=[float(v) for v in np.percentile((after-before)[draws].mean(axis=1)*100,[2.5,97.5])]
 counts={arm:{cell:results[arm]['4096']['development']['cells'][cell]['complete'] for cell in CELLS} for arm in ('original','broad')}
 out=dict(scope='Prospectively fixed paired2101 development corpus intervention; inspected DEV, no sealed confirmation. Event-bootstrap intervals are descriptive conditional on one paired initialization, not seed uncertainty. No calibration, selection or automatic extension. Strict generated-record codec replay/checkpoint AdamW audit remains independent.',config_sha256=CONFIG,analysis_sha256=sha(Path(__file__)),helper_pins=PINS,manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),artifact_inventory=inventory,state_inventory=state_inventory,curves=results,training_loss_curves={r['arm']:r['losses'] for r in m['results']},exposures={r['arm']:r['exposure'] for r in m['results']},paired=paired,endpoint_delta_pp_conditional_ci95=ci,endpoint_counts=counts,decisions=g.decisions(counts['original'],counts['broad']),process_seconds=m['process_seconds'],training_seconds={r['arm']:r['training_seconds'] for r in m['results']})
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('config','main','receipt','source','train-original','train-broad','development','panels','audit','vocabulary-audit','output'):p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
