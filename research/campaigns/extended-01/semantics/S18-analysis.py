"""Prospective S18 CPU analysis. Imports no model and never fits thresholds."""
import argparse,ast,collections,gzip,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
STEPS=(0,1024,2048,4096)
CELLS=('3x3','3x4','4x3','4x4')
ARMS=('original','context','workspace_control')
POLICIES=('raw','historical','matched')
BINDINGS_SHA='4364e758de6792886c391cb372ba3a9d67d5e26c081608492043c08a93c4f74b'
HELPER_SHA='13a8c12a014f97b95b65b221457ade75be318c0cb2f76e24234c8391e12d8355'
SOURCES=('campaign_semantics_s18.py','campaign_semantics_s18_freeze.py','campaign_semantics_s18_launch.py',
 'campaign_semantics_s18_actor.py','campaign_semantics_s18_compute.py','campaign_semantics_grounded_actor.py',
 'campaign_semantics_shape_train.py','campaign_semantics.py','campaign_semantics_data.py','campaign_semantics_continue.py',
 'campaign_semantics_lr.py','semantic_curriculum.py','semantic_text_acquisition.py','semantic_scaling.py',
 'semantic_contracts.py','thinking.py','thinking_language.py','semantic_graph.py','tcn_data.py')
# Fill only from prospectively reviewed immutable freezes, before main-result reads.
PINNED_CONFIG_SHA={'main':None,'reference':None}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.load(gzip.open(path,'rt'))
def require(condition,message):
 if not condition:raise ValueError(message)
def hashcheck(path,expected):require(sha(path)==expected,'changed artifact: '+str(path));return Path(path)
def metadata_guard(config,receipt,actual_config_sha,expected_config_sha):
 require(isinstance(expected_config_sha,str) and len(expected_config_sha)==64,'prospective config pin unavailable')
 require(actual_config_sha==expected_config_sha,'frozen config pin mismatch')
 require(set(config['source_sha256'])==set(SOURCES),'incomplete or extra source freeze')
 require(receipt['wrapper_sha256']==config['source_sha256']['campaign_semantics_s18_launch.py'],'receipt launcher mismatch')
def final_state_guard(result):
 require(result['curves'][-1]['added_update']==4096 and result['curves'][-1]['model_state_sha256']==result['final_state_sha256'],'final curve/state disagreement')
def evaluation_guard(document,entry,filename):
 require(entry['artifact']==filename and entry['update']==document['update'],'manifest evaluation identity mismatch')
 require(document['calibration_data_artifact']==f"calibration-u{document['update']}.npz",'calibration NPZ name mismatch')
 require(len(document['thresholds'])==len(document['calibration'])==13,'relation threshold count')
 require(document['thresholds']==[v['threshold'] for v in document['calibration']],'threshold/calibration record disagreement')
 require(len(document['calibration_data_sha256'])==64,'calibration digest missing')
 for label,rows,key in [('dev_raw',document['rows'],'raw_metrics'),('dev_calibrated',document['rows'],'calibrated_metrics'),('train_raw',document['train_metrics'],'raw'),('train_calibrated',document['train_metrics'],'calibrated')]:
  require(entry[label]['examples']==len(rows) and entry[label]['exact']==sum(r[key]['semantic_equivalence'] for r in rows),'manifest evaluation count disagreement')
  require(np.isclose(entry[label]['copy'],sum(r[key]['identity_copy_accuracy'] for r in rows)/len(rows),rtol=0,atol=1e-12),'manifest copy summary disagreement')
  for kind in ('typed_edge','ordered_edge'):
   tp=sum(r[key][kind]['true_positive'] for r in rows);den=sum(r[key][kind]['predicted_count']+r[key][kind]['gold_count'] for r in rows)
   require(np.isclose(entry[label][kind+'_f1'],2*tp/max(1,den),rtol=0,atol=1e-12),'manifest edge summary disagreement')
def calibration_array_guard(document,npz):
 require(set(npz.files)=={'scores','targets','pairs','offsets'},'calibration NPZ arrays differ')
 scores=npz['scores'];targets=npz['targets'];pairs=npz['pairs'];offsets=npz['offsets']
 require(scores.shape==targets.shape and scores.ndim==2 and scores.shape[1]==13 and pairs.shape==(len(scores),2),'calibration array shapes')
 require(offsets.shape==(129,) and offsets[0]==0 and offsets[-1]==len(scores) and bool(np.all(np.diff(offsets)>=0)),'calibration offsets')
 require(len(document['calibration_records'])==128,'calibration records incomplete')
 require(all(r['start']==int(offsets[i]) and r['stop']==int(offsets[i+1]) and r['seed']==document['train_rows'][i]['seed'] for i,r in enumerate(document['calibration_records'])),'calibration record/array identity differs')
def decisions(counts):
 require(set(counts)==set(ARMS),'all three arms required')
 require(all(set(v)==set(CELLS) and all(type(n) is int and 0<=n<=512 for n in v.values()) for v in counts.values()),'four512cells required')
 c=counts['context'];controls=[counts[a] for a in ('original','workspace_control')]
 acquisition=c['3x3']>=52 and all(c['3x3']-z['3x3']>=26 for z in controls)
 retention=c['4x3']+c['4x4']>=103 and all(c['4x3']+c['4x4']-(z['4x3']+z['4x4'])>=-51 for z in controls)
 recombination=c['3x4']>=52 and all(c['3x4']-z['3x4']>=26 for z in controls)
 return dict(acquisition=acquisition,retention=retention,recombination=recombination,development_criteria_met=acquisition and retention,recombination_criteria_met=acquisition and retention and recombination,automatic_extension=False)
def transitions(a,b):
 require(len(a)==len(b),'paired count differs');return dict(collections.Counter(f'{int(x)}->{int(y)}' for x,y in zip(a,b)))
def index_rows(rows,expected):
 result={r['semantic_sha256']:r for r in rows};require(len(result)==len(rows)==len(expected) and set(result)==set(expected),'missing/duplicate/foreign events')
 return result
def stream_guard(result,expected):
 require(len(result['visits'])==4096 and all(v==8 for v in result['visits']),'complete4096eight-visit stream required')
 require(all(result[k]==v for k,v in expected.items()),'training stream/exposure changed')
 require(result['inherited_optimizer_steps']==[24576.],'parent AdamW changed')
def bootstrap(deltas,draws):
 return {name:[float(x) for x in np.percentile(np.asarray(d)[draws].mean(axis=1)*100,[2.5,97.5])] for name,d in deltas.items()}
def helper():
 p=Path(__file__).with_name('S17-analyze.py');hashcheck(p,HELPER_SHA);spec=importlib.util.spec_from_file_location('s17_analysis_components',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def summary(rows,metrics,policy,a,roles):
 counts=collections.Counter();component=collections.Counter();relations={r:dict(false_positive=0,false_negative=0,wrong_slots_on_present_gold=0) for r in roles};micro={k:collections.Counter() for k in ('node','typed_edge','ordered_edge')};macro=collections.Counter();flags=[]
 for row,m in zip(rows,metrics,strict=True):
  gold=a.unpack(row['target']);pred=a.unpack(row['raw'])
  if policy!='raw':pred['edges']=a.edge(row['calibrated_edges'])
  comp=a.components(pred,gold);exact=all(comp.values());require(exact==bool(m['semantic_equivalence']),'component/exact disagreement');flags.append(int(exact));component.update({k:int(v) for k,v in comp.items()})
  for k in micro:
   micro[k].update({n:m[k][n] for n in ('true_positive','predicted_count','gold_count')});macro[k+'_f1']+=m[k]['f1']/len(rows)
  for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):macro[k]+=m[k]/len(rows)
  edges=pred['edges']&pred['presence'][:,None,None]&pred['presence'][None,:,None]
  for i,role in enumerate(roles):
   pe=edges[:,:,i];ge=gold['edges'][:,:,i];s=relations[role];s['false_positive']+=int((pe&~ge).sum());s['false_negative']+=int((ge&~pe).sum());s['wrong_slots_on_present_gold']+=int((pe&ge&(pred['slots']!=gold['slots'])).sum())
 return dict(examples=len(rows),complete=sum(flags),exact_components=dict(component),micro_counts={k:{**v,'f1':2*v['true_positive']/max(1,v['predicted_count']+v['gold_count'])} for k,v in micro.items()},macro_graph_metrics=dict(macro),relations=relations),flags

def main(args):
 a=helper();bindings=json.loads(hashcheck(args.bindings,BINDINGS_SHA).read_text());root=args.archive_root
 roles=None
 for n in ast.parse((args.source/'thinking_language.py').read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES':roles=ast.literal_eval(n.value)
 require(roles is not None,'role schema missing')
 manifests={};configs={};hashes={}
 for job,path,config_path,receipt_path in [('main',args.main,args.main_config,args.main_receipt),('reference',args.reference,args.reference_config,args.reference_receipt)]:
  c=json.loads(config_path.read_text());receipt=json.loads(receipt_path.read_text());metadata_guard(c,receipt,sha(config_path),PINNED_CONFIG_SHA[job]);m=load(path/'manifest.json.gz')
  require(c==m['config'] and c['budget_status']=='frozen' and c['job']==job,'wrong/floating config')
  require(c['bindings']['sha256']==BINDINGS_SHA and c['dev_per_cell']==512 and c['calibration_count']==128,'population changed')
  require(c['arms']==(['context','workspace_control'] if job=='main' else ['original']) and c['checkpoints']==list(STEPS if job=='main' else STEPS[:3]),'arms/curves changed')
  require((c['added_updates'],c['width'],c['capacity'],c['batch_size'],c['learning_rate'],c['schedule_seed'],c['negative_pairs'],c['control_microsteps'])==((4096 if job=='main' else 0),1024,128,8,1e-5,15115,128,10),'recipe changed')
  require(c['baseline_policy']=='reuse_S15_mixed_S17_endpoint_plus_frozen_early_curves' and c['calibration_policy']=='same_S17_mixed128_matched_primary_historical_secondary','policy changed')
  require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==sha(config_path) and receipt['cap_seconds']==c['proposed_cap_seconds'],'receipt/config mismatch')
  for name,h in c['source_sha256'].items():hashcheck(args.source/name,h)
  manifests[job]=m;configs[job]=c;hashes[job]=dict(manifest=sha(path/'manifest.json.gz'),config=sha(config_path),receipt=sha(receipt_path))
 require(configs['main']['source_sha256']==configs['reference']['source_sha256'],'paired source mismatch')
 oldpath=root/'s15-mixed-main-v3';old=load(hashcheck(oldpath/'manifest.json.gz',bindings['original_manifest']['sha256']));require(old['config']==bindings['original_config'],'baseline config mismatch')
 require(len(old['visits'])==4096 and set(old['visits'])=={8} and all(old[k]==v for k,v in bindings['expected_streams'].items()),'baseline stream mismatch')
 s17path=root/'s17-calibration-main-v1';s17=load(hashcheck(s17path/'manifest.json.gz',bindings['s17_manifest']['sha256']));s17mixed=next(x for x in s17['artifacts'] if x['arm']=='mixed')
 require(s17mixed['checkpoint_sha256']==bindings['checkpoints'][-1]['checkpoint']['sha256'] and s17mixed['model_state_sha256']==old['final_state_sha256'],'endpoint state mismatch')
 hashcheck(s17path/'mixed/calibration-u28672.npz',bindings['matched_endpoint_calibration']['sha256'])
 cachepath=hashcheck(root/'s15-shape-cache-v2/development.jsonl.gz',bindings['caches']['development']['sha256']);cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(cachepath,'rt'))};require(len(cache)==2048,'DEV count')
 selection=json.loads(hashcheck(root/'s17-calibration-selection/selection.json',bindings['selection']['sha256']).read_text())['mixed'];selections={r['seed']:r for r in selection};require(len(selections)==128,'TRAIN128 selection')
 events={cell:sorted(k for k,r in cache.items() if f"{r['arity']}x{r['facts']}"==cell) for cell in CELLS};require(all(len(v)==512 for v in events.values()),'cell counts')
 targets={};raw_records={};results={};flags={};train_panels={};inventory={}
 def consume(arm,step,policy,path,expected_sha,entry):
  hashcheck(path,expected_sha);require(entry['sha256']==expected_sha,'manifest evaluation hash disagreement');d=load(path);require(d['update']==24576+step,'artifact checkpoint differs');evaluation_guard(d,entry,path.name);npz_path=hashcheck(path.parent/d['calibration_data_artifact'],d['calibration_data_sha256'])
  with np.load(npz_path,allow_pickle=False) as npz:calibration_array_guard(d,npz)
  inventory[f'{arm}/{step}/{policy}']=dict(evaluation_path=str(path),evaluation_sha256=expected_sha,calibration_path=str(npz_path),calibration_sha256=d['calibration_data_sha256'])
  if arm=='original' and step==4096 and policy=='matched':require(d['thresholds']==s17mixed['new_thresholds'] and d['calibration_data_sha256']==s17mixed['calibration_npz_sha256'],'S17 calibration metadata disagreement')
  rows=index_rows(d['rows'],cache);key=(arm,step);raw_digest=hashlib.sha256(json.dumps([(k,rows[k]['raw'],rows[k]['raw_metrics']) for k in sorted(rows)],sort_keys=True).encode()).hexdigest()
  if key in raw_records:require(raw_records[key]==raw_digest,'raw output changed with policy')
  raw_records[key]=raw_digest
  for event,r in rows.items():
   require(r['graph_sha256']==cache[event]['graph_sha256'] and r['seed']==cache[event]['seed'],'event metadata changed')
   h=hashlib.sha256(json.dumps(r['target'],sort_keys=True).encode()).hexdigest()
   if event in targets:require(targets[event]==h,'event target differs across arms/curves/policies')
   targets[event]=h
  result=results.setdefault(arm,{}).setdefault(str(step),{});f=flags.setdefault((arm,step),{})
  for label in (('raw',policy) if 'raw' not in result else (policy,)):
   result[label]={};f[label]={}
   for cell,ids in events.items():
    rr=[rows[i] for i in ids];mm=[r['raw_metrics' if label=='raw' else 'calibrated_metrics'] for r in rr];result[label][cell],f[label][cell]=summary(rr,mm,label,a,roles)
  require(len(d['train_rows'])==len(d['train_metrics'])==128,'calibration rows missing')
  if policy=='matched':require([r['seed'] for r in d['train_rows']]==[r['seed'] for r in selection],'actual TRAIN selection differs')
  panels={}
  for cell in (('3x3','4x3','4x4') if policy=='matched' else ('historical128',)):
   indices=[i for i,r in enumerate(d['train_rows']) if policy!='matched' or selections[r['seed']]['cell']==cell];rr=[d['train_rows'][i] for i in indices];panels[cell]={}
   for label in ('raw','calibrated'):panels[cell][label],_=summary(rr,[d['train_metrics'][i][label] for i in indices],label,a,roles)
  train_panels.setdefault(arm,{}).setdefault(str(step),{})[policy]=dict(scope='Calibration-overlap fit panel; calibrated results optimistic, not independent generalization.',cells=panels,thresholds=d['thresholds'],calibration=d['calibration'])
 ref=manifests['reference']['results'];require(len(ref)==1 and ref[0]['no_optimizer_or_training'],'reference trained/missing');require([c['added_update'] for c in ref[0]['curves']]==list(STEPS[:3]),'reference curves incomplete')
 for i,step in enumerate(STEPS):
  bound=bindings['checkpoints'][i];require(old['curves'][i]['checkpoint_sha256']==bound['checkpoint']['sha256'],'original checkpoint binding')
  consume('original',step,'historical',oldpath/f"evaluation-u{24576+step}.json.gz",bound['historical_evaluation']['sha256'],old['curves'][i]['evaluation'])
  if step==4096:consume('original',step,'matched',s17path/'mixed/evaluation-u28672.json.gz',bindings['matched_endpoint']['sha256'],s17mixed['evaluation'])
  else:
   curve=ref[0]['curves'][i];require(curve['checkpoint']==bound['checkpoint'] and curve['raw_target_replay_exact'],'reference checkpoint/replay mismatch');consume('original',step,'matched',args.reference/f'u{24576+step}'/curve['evaluation']['artifact'],curve['evaluation']['sha256'],curve['evaluation'])
 require([r['arm'] for r in manifests['main']['results']]==['context','workspace_control'],'new arms incomplete')
 for arm_result in manifests['main']['results']:
  final_state_guard(arm_result);arm=arm_result['arm'];require(arm_result['curves'][0]['model_state_sha256']==bindings['original_initial_state_sha256'],'new arm initial tensors changed');stream_guard(arm_result,bindings['expected_streams']);require([c['added_update'] for c in arm_result['curves']]==list(STEPS),'new curves incomplete')
  for curve in arm_result['curves']:
   step=curve['added_update'];require(curve['update']==24576+step and curve['added_presentations']==step*8,'curve exposure changed')
   for policy in ('matched','historical'):consume(arm,step,policy,args.main/arm/f'{policy}-u{24576+step}'/curve[policy]['artifact'],curve[policy]['sha256'],curve[policy])
 require(all(set(results[arm][str(step)])==set(POLICIES) for arm in ARMS for step in STEPS),'incomplete policies')
 paired={};intervals={};rng=np.random.default_rng(18018)
 for cell in CELLS:
  draws=rng.integers(0,512,size=(10000,512));deltas={}
  for step in STEPS:
   for policy in POLICIES:
    for comparator in ('original','workspace_control'):
     c=np.array(flags[('context',step)][policy][cell]);b=np.array(flags[(comparator,step)][policy][cell]);key=f'{cell}/{step}/{policy}/context-minus-{comparator}';cr=results['context'][str(step)][policy][cell];br=results[comparator][str(step)][policy][cell];paired[key]=dict(transitions_comparator_to_context=transitions(b,c),delta_complete=int(c.sum()-b.sum()),component_delta={k:cr['exact_components'][k]-br['exact_components'][k] for k in cr['exact_components']},relation_error_delta={r:{k:cr['relations'][r][k]-br['relations'][r][k] for k in cr['relations'][r]} for r in roles})
     if step==4096:deltas[key]=c-b
  intervals.update(bootstrap(deltas,draws))
 counts={arm:{cell:results[arm]['4096']['matched'][cell]['complete'] for cell in CELLS} for arm in ARMS}
 out=dict(scope='S15/S17-informed single-parent development; reused inspected baseline. Fixed4096 matched policy decides gates. Event-bootstrap intervals are conditional on this one training lineage, not seed uncertainty, and descriptive rather than multiplicity-adjusted. Same event draws shared across policies/comparators within each cell. No automatic extension or confirmation.',analysis_source_sha256=sha(Path(__file__)),bindings_sha256=BINDINGS_SHA,inputs=hashes,frozen_config_pins=PINNED_CONFIG_SHA,artifact_inventory=inventory,results=results,calibration_overlap_train_panels=train_panels,paired=paired,endpoint_delta_percentage_points_ci95=intervals,decisions=decisions(counts),baseline_inherited_seconds=484.14140757126734,checkpoint_scope='Checkpoint bytes remain remote; input binding/receipts/manifests checked here. Independent remote audit must verify saved model/AdamW bytes and state; this CPU analysis does not deserialize checkpoints.')
 args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('main','reference','main-config','reference-config','main-receipt','reference-receipt','bindings','source','archive-root','output'):p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
