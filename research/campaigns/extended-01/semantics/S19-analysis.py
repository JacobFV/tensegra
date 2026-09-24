"""Prospective complete-main CPU analysis. No model, refitting, or confirmation access."""
import argparse,collections,gzip,hashlib,importlib.util,json,math
from pathlib import Path
CONFIG='07ccff2ddb6d345355ea0b443dd93000d5296e7afeebd6fe2998ab826b645e9b'
REFERENCE='c2010315a0b61ffe2d51b57acedace3b5c7fbad7c332bf09a7805379d82d154a'
CELLS=('3x3','3x4','4x3','4x4');STEPS=(0,1024,2048,4096)
FIELDS=('type','kind','value','copy','source','target','role','slot')
COMPONENTS=('presence','kind','value','copy','edges','slots')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,h):require(sha(p)==h,'hash mismatch: '+str(p));return Path(p)
def load(p):return json.load(gzip.open(p,'rt'))
def decisions(train,early,final):
 require(type(train)is int and 0<=train<=128,'TRAIN128 count')
 for d in (early,final):require(set(d)==set(CELLS) and all(type(n)is int and 0<=n<=512 for n in d.values()),'four DEV512 counts')
 # Integer numerator /2048 avoids floating-point ambiguity at registered boundaries.
 weighted=lambda d:2*d['3x3']+d['4x3']+d['4x4']
 fit=train>=64;acquisition=all(final[c]>=52 for c in ('3x3','4x3','4x4'))
 return dict(train_fit=fit,each_trained_cell=acquisition,promotion=fit and acquisition,heldout_composition=final['3x4']>=52,weighted_trained_accuracy=weighted(final)/2048,weighted_gain_since2048=(weighted(final)-weighted(early))/2048,extension_eligible=fit and weighted(final)*100>=5*2048 and (weighted(final)-weighted(early))*100>=3*2048,automatic_extension=False)
def indexed(rows):
 d={r['semantic_sha256']:r for r in rows};require(len(d)==len(rows),'duplicate event');return d
def paired(old,new):
 require(len(old)==len(new),'paired support');return {f'{a}->{b}':sum(int(x)==a and int(y)==b for x,y in zip(old,new)) for a in (0,1) for b in (0,1)}
def summarize(rows):
 valid=[r for r in rows if r['valid']];components={k:sum(r['exact_components'][k] for r in rows) for k in COMPONENTS}
 macro={k:sum(r['metrics'][k]['f1'] for r in valid)/max(1,len(rows)) for k in ('node','typed_edge','ordered_edge')}
 return dict(examples=len(rows),valid=len(valid),complete=sum(r['complete'] for r in rows),exact_components=components,invalid_reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),macro_graph_f1_invalid_zero=macro,canonical_edge_order_count=sum(r.get('canonical_edge_order',False) for r in rows))
def linkage_guard(m):
 expected={'optimizer_records':3229392,'optimizer_tokens':1720320,'optimizer_nodes':988008,'optimizer_edges':2208616}
 require(all(m[k]==v for k,v in expected.items()),'exact optimizer exposure')
 require(m['curves'][0]['model_state_sha256']==m['initial_state_sha256'],'initial curve/state')
 for curve in m['curves']:
  for key in ('checkpoint_sha256','model_state_sha256'):
   digest=curve[key];require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'checkpoint digest')
  for key in ('presentations','optimizer_records','optimizer_tokens'):require(type(curve[key])is int and curve[key]>=0,'curve counter')
  require(curve['presentations']==curve['update']*8,'curve presentations')
  for key in ('optimizer_records','optimizer_tokens'):
   # Every registered curve is an integral number of complete corpus epochs.
   require(curve[key]*4096==m[key]*curve['update'],'curve exposure linkage')
def entry_guard(entry,d):
 require(entry['examples']==len(d['rows']),'entry examples')
 require(entry['teacher_forced_loss']==d['teacher_forced_loss'],'entry teacher-forced loss')
 cells={cell:dict(examples=sum(r['cell']==cell for r in d['rows']),complete=sum(r['complete'] for r in d['rows'] if r['cell']==cell)) for cell in {r['cell'] for r in d['rows']}}
 require(entry['cells']==cells,'entry cell summary')
def main(a):
 c=json.loads(checked(a.config,CONFIG).read_text());m=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text())
 require(m['config']==c and c['job']=='main' and c['budget_status']=='frozen','main config')
 require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==CONFIG and receipt['cap_seconds']==1000,'completed receipt')
 require(receipt['wrapper_sha256']==c['source_sha256']['campaign_semantics_s19_launch.py'],'wrapper')
 for name,h in c['source_sha256'].items():checked(a.source/name,h)
 require(m['parameters']==62677315 and m['inherited_presentations']==0 and m['added_presentations']==32768,'model/exposure')
 require(len(m['visits'])==4096 and all(v==8 for v in m['visits']),'eight visits')
 require(m['construction_sequence_sha256']==c['expected_construction_sequence_sha256'],'construction stream')
 require([x['update'] for x in m['curves']]==list(STEPS) and m['curves'][-1]['model_state_sha256']==m['final_state_sha256'],'complete curves/final state')
 linkage_guard(m)
 helper_path=Path(__file__).with_name('S17-analyze.py');checked(helper_path,'13a8c12a014f97b95b65b221457ade75be318c0cb2f76e24234c8391e12d8355')
 spec=importlib.util.spec_from_file_location('s19_component_helper',helper_path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 inventory={};results={};documents={};fixed={}
 for curve in m['curves']:
  step=curve['update'];require(curve['presentations']==step*8,'curve exposure');results[str(step)]={}
  for population in ('train','development'):
   entry=curve[population];path=checked(a.main/entry['artifact'],entry['sha256']);d=load(path);inventory[f'{step}/{population}']=dict(path=str(path),sha256=sha(path));rows=d['rows'];ix=indexed(rows)
   entry_guard(entry,d)
   require(d['update']==step and d['population']==population,'evaluation metadata')
   require(len(rows)==(128 if population=='train' else 2048),'full population')
   require(d['teacher_forced']==entry['teacher_forced'] and set(d['teacher_forced'])==set(FIELDS),'eight field diagnostics')
   for f in d['teacher_forced'].values():require(f['count']>0 and 0<=f['correct']<=f['count'] and math.isclose(f['mean_loss'],f['loss_sum']/f['count']) and math.isclose(f['accuracy'],f['correct']/f['count']),'field arithmetic')
   require(math.isclose(d['teacher_forced_loss'],sum(f['mean_loss'] for f in d['teacher_forced'].values())/8),'eight field loss')
   identity={k:(r['seed'],r['graph_sha256'],r['cell'],r['target']) for k,r in ix.items()}
   if population in fixed:require(fixed[population]==identity,'curve events/targets changed')
   else:fixed[population]=identity
   for r in rows:
    require(set(r['exact_components'])==set(COMPONENTS),'component schema')
    if r['valid']:
     comp=h.components(h.unpack(r['prediction']),h.unpack(r['target']));require(comp==r['exact_components'] and all(comp.values())==r['complete']==bool(r['metrics']['semantic_equivalence']),'prediction components')
    else:require(not r['complete'] and not any(r['exact_components'].values()) and r['prediction'] is None and r['metrics'] is None and bool(r['reason']),'invalid output credit')
   summary=summarize(rows);require(summary['complete']==entry['complete'] and summary['valid']==entry['valid'] and summary['invalid_reasons']==entry['invalid_reasons'],'manifest counts')
   cells={cell:summarize([r for r in rows if r['cell']==cell]) for cell in (('3x3','4x3','4x4') if population=='train' else CELLS)}
   require({k:v['examples'] for k,v in cells.items()}==({'3x3':64,'4x3':32,'4x4':32} if population=='train' else dict.fromkeys(CELLS,512)),'cell supports')
   results[str(step)][population]=dict(**summary,cells=cells,teacher_forced=d['teacher_forced'],teacher_forced_loss=d['teacher_forced_loss']);documents[(step,population)]=ix
 reference=json.loads(checked(a.reference_analysis,REFERENCE).read_text());comparisons={}
 for arm in ('original','context','workspace_control'):
  for policy in ('raw','historical','matched'):
   ent=reference['artifact_inventory'][f'{arm}/4096/{"matched" if policy=="raw" else policy}'];suffix=ent['evaluation_path'].split('/semantics/',1)[1];path=checked(a.reference_root/suffix,ent['evaluation_sha256']);old=indexed(load(path)['rows']);new=documents[(4096,'development')];require(set(old)==set(new),'baseline event population');inventory[f'reference/{arm}/{policy}']=dict(path=str(path),sha256=sha(path))
   for event,r in new.items():require(old[event]['target']==r['target'] and old[event]['seed']==r['seed'] and old[event]['graph_sha256']==r['graph_sha256'],'baseline target identity')
   comparisons[f'{arm}/{policy}']={}
   for cell in CELLS:
    ids=sorted(k for k,r in new.items() if r['cell']==cell);before=[bool(old[k]['raw_metrics' if policy=='raw' else 'calibrated_metrics']['semantic_equivalence']) for k in ids];after=[new[k]['complete'] for k in ids];comparisons[f'{arm}/{policy}'][cell]=dict(transitions_reference_to_s19=paired(before,after),delta_complete=sum(after)-sum(before))
 counts=lambda step:{cell:results[str(step)]['development']['cells'][cell]['complete'] for cell in CELLS}
 out=dict(scope='Single scratch seed, inspected development; TRAIN panel is training-exposed. Teacher-forced gold-prefix diagnostics are distinct from public-only free-running output. Historical actors have 196608 inherited presentations; S19 has none. No parameter/FLOP/history-matched architectural attribution, refit, best-checkpoint selection, or confirmation inference.',config_sha256=CONFIG,analysis_sha256=sha(Path(__file__)),manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),reference_analysis_sha256=REFERENCE,artifact_inventory=inventory,curves=results,endpoint_paired=comparisons,decisions=decisions(results['4096']['train']['complete'],counts(2048),counts(4096)),costs={k:m[k] for k in ('training_seconds','preflight_seconds','setup_seconds','process_seconds','optimizer_records','optimizer_tokens','optimizer_nodes','optimizer_edges')},checkpoint_scope='Hashes linked by manifest; independent checkpoint/AdamW byte and strict generated-record codec replay audits remain separate; this analysis checks packed tensor components.')
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('config','main','receipt','source','reference-analysis','reference-root','output'):p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
