"""Prospective complete S22 oracle-boundary analysis; no model or policy fitting."""
import argparse,collections,gzip,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
MAIN_CONFIG_SHA=None # Pin only after separately reviewed main freeze, before outcomes.
CELLS=('2x4','3x3','3x4','4x3','4x4','5x3','5x4')
POLICIES=('oracle_node_count','oracle_node_kinds','oracle_node_prefix')
HELPER_SHA='13a8c12a014f97b95b65b221457ade75be318c0cb2f76e24234c8391e12d8355'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,h):require(isinstance(h,str) and sha(p)==h,'unbound/changed artifact '+str(p));return Path(p)
def load(p):return json.load(gzip.open(p,'rt'))
def helper():
 p=Path(__file__).with_name('S17-analyze.py');checked(p,HELPER_SHA);s=importlib.util.spec_from_file_location('s22_components',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stats_guard(row,policy):
 t=row['target'];n=sum(t['presence']);s=row['controller_stats'];records=row['records']
 keys={'policy','supplied_node_count','forced_node_tags','tag_replacements','forced_node_kinds','kind_replacements','supplied_prefix_records','prefix_value_replacements','prefix_copy_replacements','node_suppression_steps','node_suppression_replacements'}
 require(set(s)==keys and s['policy']==policy,'controller stats schema/policy')
 require(all(type(v)is int and v>=0 for k,v in s.items() if k!='policy'),'counter type')
 require(s['supplied_node_count']==s['forced_node_tags']==n,'forced NODE count')
 kinds=policy!='oracle_node_count';prefix=policy=='oracle_node_prefix'
 require(s['forced_node_kinds']==(n if kinds else 0) and s['supplied_prefix_records']==(n if prefix else 0),'privilege counts')
 require(s['tag_replacements']<=n and s['kind_replacements']<=s['forced_node_kinds'] and s['node_suppression_replacements']<=s['node_suppression_steps'],'replacement bounds')
 require(all(s[k]<= (n if prefix else 0) for k in ('prefix_value_replacements','prefix_copy_replacements')),'prefix replacement bounds')
 require(n<len(records)<=160 and len(records)==n+s['node_suppression_steps'],'prefix/termination record support')
 require(all(isinstance(r,list) and len(r)==5 and all(type(v)is int for v in r) for r in records),'record schema')
 require(all(r[0]==1 for r in records[:n]) and all(r[0] in (2,3) for r in records[n:]),'forced tag direction')
 require(all(r[0]!=3 for r in records[:-1]),'records after EOS')
 if row['valid']:require(records[-1][0]==3,'valid EOS')
 for i,r in enumerate(records[:n]):
  if kinds:require(r[1]==t['kind'][i],'forced kind mismatch')
  if prefix:require(r==[1,t['kind'][i],t['value'][i],t['copy'][i],-1],'supplied full NODE prefix mismatch')
def flags_guard(r):
 require(type(r['valid'])is bool and type(r['complete'])is bool and (not r['complete'] or r['valid']),'validity bools')
 require((r['reason'] is None) if r['valid'] else isinstance(r['reason'],str) and bool(r['reason']),'reason consistency')
def transitions(old,new):return {f'{a}->{b}':sum(int(x)==a and int(y)==b for x,y in zip(old,new)) for a in (0,1) for b in (0,1)}
def summarize(rows):
 keys=('presence','kind','value','copy','edges','slots');valid=[r for r in rows if r['valid']];stats=collections.Counter()
 for r in rows:stats.update({k:v for k,v in r['controller_stats'].items() if k!='policy'})
 return dict(examples=len(rows),valid=len(valid),complete=sum(r['complete'] for r in rows),exact_nodes=sum(r['exact_nodes'] for r in rows),exact_relations=sum(r['exact_relations'] for r in rows),exact_components={k:sum(r['exact_components'][k] for r in rows) for k in keys},invalid_reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])),controller_stats=dict(stats),macro_graph_f1_invalid_zero={k:sum(r['metrics'][k]['f1'] for r in valid)/len(rows) for k in ('node','typed_edge','ordered_edge')},first_failed_semantic_id=min((r['semantic_sha256'] for r in rows if not r['complete']),default=None))
def main(a):
 c=json.loads(checked(a.config,MAIN_CONFIG_SHA).read_text());m=load(a.main/'manifest.json.gz');receipt=json.loads(a.receipt.read_text());h=helper()
 require(c==m['config'] and c['job']=='main' and c['budget_status']=='frozen' and c['dev_per_cell']==512 and c['no_optimizer_or_training'] and m['no_optimizer_or_training'],'complete frozen main')
 require(c['arms']==['original','broad'] and c['policies']==list(POLICIES),'full diagnostic matrix')
 require(receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==MAIN_CONFIG_SHA and receipt['cap_seconds']==c['cap_seconds'] and receipt['wrapper_sha256']==c['source_sha256']['campaign_semantics_s22_launch.py'],'closed receipt')
 for name,v in c['source_sha256'].items():checked(a.source/name,v)
 require(set(c['inputs'])=={'development','vocabulary_audit'},'no confirmation input')
 cache=list(map(json.loads,gzip.open(checked(a.development,c['inputs']['development']['sha256']),'rt')));expected={r['semantic_sha256']:r for r in cache};require(len(cache)==len(expected)==3584,'seven-cell support');events={cell:sorted(k for k,r in expected.items() if f"{r['arity']}x{r['facts']}"==cell) for cell in CELLS};require(all(len(v)==512 for v in events.values()),'512/cell')
 require([r['arm'] for r in m['results']]==c['arms'],'closed both backbones');summaries={};flags={};inventory={};baseline={}
 for result in m['results']:
  arm=result['arm'];bind=c['checkpoints'][arm];folder=a.main/arm;require(load(folder/'manifest.json.gz')==result,'per-arm manifest');require(result['checkpoint_sha256']==bind['checkpoint']['sha256'] and result['model_state_sha256']==bind['model_state_sha256'] and result['no_optimizer_or_training'] and result['parameters']==62677315 and result['worst_case_timing'] is None,'unchanged frozen state')
  refpath=checked(a.references/arm/'development-u4096.json.gz',bind['public_reference']['sha256']);ref=load(refpath);reference={r['semantic_sha256']:r for r in ref['rows']};require(len(reference)==len(ref['rows'])==3584 and set(reference)==set(expected) and ref['update']==4096,'public reference');inventory[arm+'/public']=dict(path=str(refpath),sha256=sha(refpath));baseline[arm]={cell:sum(reference[i]['complete'] for i in ids) for cell,ids in events.items()};summaries[arm]={};flags[arm]={}
  require([p['policy'] for p in result['policies']]==list(POLICIES),'all policies complete')
  for entry in result['policies']:
   policy=entry['policy'];path=checked(folder/entry['artifact'],entry['sha256']);d=load(path);require(d['policy']==policy and entry['model_state_sha256']==bind['model_state_sha256'],'policy/state identity');rows={r['semantic_sha256']:r for r in d['rows']};require(len(rows)==len(d['rows'])==3584 and set(rows)==set(expected),'complete unique events')
   for event,r in rows.items():
    e=expected[event];old=reference[event];require(r['seed']==old['seed']==e['seed'] and r['graph_sha256']==old['graph_sha256']==e['graph_sha256'] and r['cell']==f"{e['arity']}x{e['facts']}" and r['target']==old['target'] and r['policy']==policy,'public identity/targets');flags_guard(r);stats_guard(r,policy)
    require(set(r['exact_components'])=={'presence','kind','value','copy','edges','slots'} and all(type(v)is bool for v in r['exact_components'].values()) and type(r['exact_nodes'])is bool and type(r['exact_relations'])is bool,'strict component flags')
    require(r['public_reference']=={k:old[k] for k in ('valid','complete','reason','exact_components')},'untouched public-only reference')
    if r['valid']:
     comp=h.components(h.unpack(r['prediction']),h.unpack(r['target']));require(comp==r['exact_components'] and all(comp.values())==r['complete']==bool(r['metrics']['semantic_equivalence']),'packed component replay')
    else:require(r['prediction'] is None and r['metrics'] is None and not any(r['exact_components'].values()),'invalid credit')
    require(r['exact_nodes']==all(r['exact_components'][k] for k in ('presence','kind','value','copy')) and r['exact_relations']==(r['exact_components']['edges'] and r['exact_components']['slots']),'conditional components')
   summary=summarize(list(rows.values()));require(all(summary[k]==entry[k] for k in ('examples','valid','complete','invalid_reasons','controller_stats')),'aggregate metadata');summary['cells']={cell:summarize([rows[i] for i in ids]) for cell,ids in events.items()};require(entry['cells']=={cell:{k:v[k] for k in ('examples','valid','complete','exact_nodes','exact_relations')} for cell,v in summary['cells'].items()},'cell metadata');summaries[arm][policy]=summary;flags[arm][policy]={cell:[rows[i]['complete'] for i in ids] for cell,ids in events.items()};inventory[f'{arm}/{policy}']=dict(path=str(path),sha256=sha(path))
  flags[arm]['public']={cell:[reference[i]['complete'] for i in ids] for cell,ids in events.items()}
 paired={};ci={};rng=np.random.default_rng(22023)
 for cell in CELLS:
  draws=rng.integers(0,512,size=(10000,512))
  for arm in c['arms']:
   for policy in POLICIES:
    for comparator in ('public',)+tuple(p for p in POLICIES if POLICIES.index(p)<POLICIES.index(policy)):
     old=np.asarray(flags[arm][comparator][cell],dtype=np.int8);new=np.asarray(flags[arm][policy][cell],dtype=np.int8);key=f'{arm}/{cell}/{policy}-minus-{comparator}';paired[key]=dict(transitions=transitions(old,new),delta_complete=int(new.sum()-old.sum()));ci[key]=[float(v) for v in np.percentile((new-old)[draws].mean(axis=1)*100,[2.5,97.5])]
 out=dict(scope='Exploratory privileged NODE-boundary diagnostic on inspected S21 DEV and fixed checkpoints. No training, calibration, promotion gate, or public-only structural-transfer claim. Conditional on supplied count/kinds/full nodes as named. Strict raw codec/checkpoint audit independent.',analysis_sha256=sha(Path(__file__)),config_sha256=MAIN_CONFIG_SHA,manifest_sha256=sha(a.main/'manifest.json.gz'),receipt_sha256=sha(a.receipt),artifact_inventory=inventory,public_baseline_counts=baseline,results=summaries,paired=paired,conditional_event_delta_pp_ci95=ci,bootstrap=dict(seed=22023,draws=10000,scope='shared sorted event draws per cell across all arms/policies; descriptive fixed-model uncertainty only'),no_acquisition_gate=True,no_automatic_next_run=True,process_seconds=m['process_seconds'])
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('config','main','receipt','source','development','references','output'):p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
