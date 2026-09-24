"""Guarded S20 complete-main loader; refuses execution until prospective config pin.
No model/checkpoint deserialization. Strict record and AdamW audits are independent.
"""
import argparse,gzip,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
MAIN_CONFIG_SHA='4e648a9a274a7cabcbb84c0b2e6e6a2dcbf37925e2146b0758702086245f6c8b' # Reviewed prospective main freeze f21538e6.
PINS={'S20-analysis.py':'73099b4858ccf87dcc18f080d8455b17ff13d576d698357b9e5805a21d2bd13a','S18-analysis.py':'28edae0137f2c0159f0f4648698d768b24f150d170948e9089d43ce7011e1687','S19-analysis.py':'c42ac5868230e2e77a08c4df3f0157bb5a77f463215b6ccc908b82e5a5816155'}
FROZEN_SOURCES={'campaign_semantics.py': '0eda5489c8badddf7a700e398b7d35907392fd99a0bb074d779727ab951cf05d', 'campaign_semantics_continue.py': '6d5a8f4ae5143bdfc93f589fa3c189966a787e0fc6a83dc6bc06382c14ac8585', 'campaign_semantics_data.py': 'fd0c49291b62a25fa2899921bd9fb70698f9a7f798a2bd26e4f0920df4883797', 'campaign_semantics_grounded_actor.py': 'b8351a392d9928876d7a82f8cb32386df91356f561b5fb62b1dc4432feb4adfe', 'campaign_semantics_lr.py': '37382d47849c29f15b407dd9b4b7aa9d66e041ede7ba7477d016e4aa3711dd9a', 'campaign_semantics_s18.py': '993745fb640a10e3eb2862c2eabd3d15fc6509990dded1584be85a68c1b300ed', 'campaign_semantics_s18_actor.py': '7562746da3ccff3b6a1ac6874871c40f0a3f76051977639eebece40dc1435de3', 'campaign_semantics_s18_compute.py': 'fbbca955e9ee5f6a45d9522dc6e7a99fd2f1b899ab559facd0a6cb0aa3cb51c6', 'campaign_semantics_s18_freeze.py': '929b57620f5d7102972cb3039f72c93c8bc433fec01a24e09c668be9e577279e', 'campaign_semantics_s18_launch.py': 'd5d73945a4a781cf5ab08bf77afe289ac49b3c0939faa8474a4ed3f39d69dbad', 'campaign_semantics_s19.py': 'a42035daaf563d4aa63e7ded5bd949a646c77fd85ed86804ff396b4ac82c0407', 'campaign_semantics_s19_actor.py': 'b84dc10df3a34fdc3662730cbcdb88012ccb9f8ffcead4b4eebfe2588549ba17', 'campaign_semantics_s19_codec.py': '88b0606914dd4585f1715900263de16712fed78743938c3c135d89f7c9180cad', 'campaign_semantics_s19_freeze.py': 'f017039010751a7b7bac8b68c8228821f62fc1cc7b9ffcccb3bb4aff0e1bf1df', 'campaign_semantics_s19_launch.py': '5d4b197bcdf761e001aeaa6d70d24eaa884778d07a5e3098648154188587291b', 'campaign_semantics_s20.py': '7261beaabc662899d9c0b6f31809eba40a97bd125fd11989370ea47b5343ef21', 'campaign_semantics_s20_freeze.py': '78c91a9d6d929e53634ec68a4038a5fead9c3896ffab4f617a8bc439ff59d810', 'campaign_semantics_s20_launch.py': 'eb200a5695149914d4530f45d84e4642b3b41995bc77fcd78c54cd7bf2340af9', 'campaign_semantics_shape_train.py': 'bc7812834dffde6e0061b1590c51a2b70c06ddb57d84644c9b4fd396d2e49c36', 'semantic_contracts.py': '914a12a24a14434918412f2ad71368f51085bd248c7398164b0c2b1c9a0e215f', 'semantic_curriculum.py': '69293b602cd42ad2315e8e4a53085e622191a9e21a053e3e287c894e8fe58a3f', 'semantic_graph.py': '95aa8745f4f399d13d54aacf505d5cdd85c46edce2b2c069febdb6a96bb5e2bb', 'semantic_scaling.py': 'd9bf605dad9b7de9637903b980c7c00b6751273f8bea87ac977e0c275aa7f9e9', 'semantic_text_acquisition.py': 'e67269012f0ef6e30b78c0265ab007f72f3278310dfad8fad526d60a38001d59', 'tcn_data.py': '1531f0af7d7e2f718d22fa9a45866d917e8a5ab1e26a45f5a15b4f5fe7c94670', 'thinking.py': '6f719ee9fa6e58f7cd6247b610339639526a98857bfd0a88b24909bc3d2e2f64', 'thinking_language.py': '569cb1f1bab9ce17a7ab9e0df3e53af623c37fa9ad43dd33650982548178dcdf'}
FROZEN_PARENTS={'701': {'checkpoint': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-701/model-u24576.pt', 'sha256': 'f419fe818626fc23b98f8e0bb7a89491ff5e015e529b5facb68ee5281d2e808a'}, 'manifest': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-701/manifest.json.gz', 'sha256': 'bfd24b660c3a4a28577fc1d6cc7bcbdcf3f6ba138907d2660298b89e78c6112a'}, 'model_state_sha256': '0ce7b767ec460dca9ed9004c2379483603a70919e0a18651af3ff74782db7785'}, '702': {'checkpoint': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-702/model-u24576.pt', 'sha256': '64b3b035c386982d3f9563ff7e24bbd459fba8432cadaa60ffc89bf4c5d8261f'}, 'manifest': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-702/manifest.json.gz', 'sha256': 'c3501de19812fa454ed2015df65728fe442b7268dea52d1ec2b1e5a3a0eaf3b2'}, 'model_state_sha256': 'd630372e839ece322d967ea06f1cc3c0404966906e95ad5646160d6ab2f5b0c1'}, '703': {'checkpoint': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-703/model-u24576.pt', 'sha256': 'be5c324a24d5a0741b4c66159e46a92f32b8d1691c51df887a17402ccb3267a2'}, 'manifest': {'path': '/home/brandonin/topoformer-campaign01-semantics/results/s12-decay-703/manifest.json.gz', 'sha256': '9917ac55ece4df83970bc46474bd62938e3e549fe8322d2d8cd782c41a3b6416'}, 'model_state_sha256': '9e7dfa80e845ccda88d7cba737295a58ce1b1fdb841535937f578e7e09829c2c'}}
FROZEN_INPUTS={'train': {'path': 'data/s15-shape-v2/train_mixed.jsonl.gz', 'sha256': '8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2'}, 'selection': {'path': 'data/s17-calibration-selection-v1/selection.json', 'sha256': '5b8091e73320c5ef87bee39189933e6f3aa856c98ac07d93c6cb90d8d1c13172'}, 'vocabulary_audit': {'path': 'data/s01/audit.json', 'sha256': '2582514c2a31e51dc32071fcd4177d0e50d8bf710dfa48d73c66a50b81b3d1d1'}, 'confirmation': {'path': 'data/s15-shape-v2/confirmation.jsonl.gz', 'sha256': '2710e5e1b6fffab6fa33d56105de13e49d46d64c63d81f61c6ea331347229d32'}}
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
 require(c['source_sha256']==FROZEN_SOURCES and c['parents']==FROZEN_PARENTS and c['inputs']==FROZEN_INPUTS and c['cap_seconds']==4300,'exact prospective source/parent/input/cap bindings')
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
