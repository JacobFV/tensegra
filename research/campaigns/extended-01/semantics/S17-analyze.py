"""CPU-only, descriptive S17 policy comparison; no model or threshold fitting."""
import ast,base64,collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('research/results/campaign-01/semantics')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.load(gzip.open(p,'rt'))
def edge(x):
 assert x['bitorder']=='little'
 return np.unpackbits(np.frombuffer(base64.b64decode(x['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(x['shape'])].reshape(x['shape']).astype(bool)
def unpack(x):return {k:edge(v) if k=='edges' else np.asarray(v) for k,v in x.items()}
def components(p,g):
 e=p['edges']&p['presence'][:,None,None]&p['presence'][None,:,None]
 return dict(presence=bool(np.array_equal(p['presence'],g['presence'])),kind=bool(np.all(p['kind'][g['presence']]==g['kind'][g['presence']])),value=bool(np.all(p['value'][g['value']>=0]==g['value'][g['value']>=0])),copy=bool(np.all(p['copy'][g['copy']>=0]==g['copy'][g['copy']>=0])),edges=bool(np.array_equal(e,g['edges'])),slots=bool(np.all(p['slots'][g['edges'].any(-1)]==g['slots'][g['edges'].any(-1)])))
def main():
 path=ROOT/'s17-calibration-main-v1';manifest=load(path/'manifest.json.gz');config=json.loads(Path('configs/campaign-s17-calibration-main-frozen-v1.json').read_text());assert manifest['config']==config and manifest['no_optimizer_or_training']
 assert sha(Path('configs/campaign-s17-calibration-main-frozen-v1.json'))=='95d7962bea362cccb793faf322e109fc4e98dd66ebde47b0ccbd8b94b01ee98b'
 for group in ('source_sha256','launch_source_sha256'):
  for name,expected in config[group].items():assert sha(Path('src/topoformer')/name)==expected
 assert [a['arm'] for a in manifest['artifacts']]==[r['arm'] for r in config['runs']]==['control','mixed']
 receipt=json.loads((path/'s17-calibration-main-v1.occupancy.json').read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['config_sha256']==sha(Path('configs/campaign-s17-calibration-main-frozen-v1.json'))
 assert manifest['source_sha256']==config['source_sha256'] and config['dev_per_cell']==512 and config['calibration_count']==128
 cachepath=ROOT/'s15-shape-cache-v2/development.jsonl.gz';assert sha(cachepath)==config['cache_sha256']['development'];cache={r['semantic_sha256']:r for r in map(json.loads,gzip.open(cachepath,'rt'))};assert len(cache)==2048
 roles=None
 for n in ast.parse(Path('src/topoformer/thinking_language.py').read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROLES':roles=ast.literal_eval(n.value)
 results={};hashes={}
 for a,run in zip(manifest['artifacts'],config['runs'],strict=True):
  arm=run['arm'];assert a['arm']==arm
  newpath=path/a['artifact'];oldpath=ROOT/f's15-{arm}-main-v3/evaluation-u28672.json.gz';assert sha(newpath)==a['artifact_sha256'] and sha(oldpath)==run['evaluation_sha256'];assert sha(path/a['calibration_npz'])==a['calibration_npz_sha256']
  nd=load(newpath);assert nd['thresholds']==a['new_thresholds'] and nd['calibration_data_sha256']==a['calibration_npz_sha256'] and nd['calibration_data_artifact']==Path(a['calibration_npz']).name
  assert a['checkpoint_sha256']==run['checkpoint_sha256'] and a['model_state_sha256']==run['model_state_sha256'] and a['raw_target_replay_exact']
  assert a['original_thresholds']==load(oldpath)['thresholds']
  new=nd['rows'];old={r['semantic_sha256']:r for r in load(oldpath)['rows']};assert len(new)==2048 and len({r['semantic_sha256'] for r in new})==2048 and set(old)==set(cache)
  cells={};hashes[arm]={'new':sha(newpath),'old':sha(oldpath)}
  for cell in ('3x3','3x4','4x3','4x4'):
   rows=[r for r in new if f"{cache[r['semantic_sha256']]['arity']}x{cache[r['semantic_sha256']]['facts']}"==cell];assert len(rows)==512
   s={'examples':512,'policies':{},'historical_to_matched_complete':collections.Counter()}
   for label in ('raw','historical','matched'):
    v={'complete':0,'exact_components':collections.Counter(),'mean_metrics':collections.Counter(),'relations':{role:{'false_positive':0,'false_negative':0,'wrong_slots_on_present_gold':0} for role in roles}}
    for r in rows:
     o=old[r['semantic_sha256']];assert r['raw']==o['raw'] and r['target']==o['target'] and r['raw_metrics']==o['raw_metrics']
     g=unpack(r['target']);p=unpack(r['raw']);m=r['raw_metrics']
     if label!='raw':p['edges']=edge((o if label=='historical' else r)['calibrated_edges']);m=(o if label=='historical' else r)['calibrated_metrics']
     c=components(p,g);assert all(c.values())==bool(m['semantic_equivalence']);v['complete']+=int(all(c.values()));v['exact_components'].update(k for k,b in c.items() if b)
     for k in ('node','typed_edge','ordered_edge'):v['mean_metrics'][k+'_f1']+=m[k]['f1']/512
     for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):v['mean_metrics'][k]+=m[k]/512
     e=p['edges']&p['presence'][:,None,None]&p['presence'][None,:,None]
     for i,role in enumerate(roles):
      pe=e[:,:,i];ge=g['edges'][:,:,i];d=v['relations'][role];d['false_positive']+=int((pe&~ge).sum());d['false_negative']+=int((ge&~pe).sum());d['wrong_slots_on_present_gold']+=int((pe&ge&(p['slots']!=g['slots'])).sum())
    s['policies'][label]=v
   for r in rows:s['historical_to_matched_complete'][f"{int(old[r['semantic_sha256']]['calibrated_metrics']['semantic_equivalence'])}->{int(r['calibrated_metrics']['semantic_equivalence'])}"]+=1
   s['signed_matched_minus_historical']={role:{k:s['policies']['matched']['relations'][role][k]-s['policies']['historical']['relations'][role][k] for k in s['policies']['matched']['relations'][role]} for role in roles};cells[cell]=s
  results[arm]=cells
 out={'scope':'Exploratory S15-informed calibration diagnostic. No new gate, confirmation, checkpoint selection, further fitting or DEV fitting in this analysis, or retrospective S15 promotion. The declared TRAIN128 alternative was fitted by S17. Mean F1 metrics are macro averages across graphs. Signed errors: negative means fewer errors. Components reconstructed from complete packed labels.','source_sha256':sha(Path(__file__)),'endpoint_artifacts':hashes,'calibration_metadata':[{k:v for k,v in a.items() if k not in ('evaluation',)} for a in manifest['artifacts']],'results':results}
 (ROOT/'s17-paired-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
