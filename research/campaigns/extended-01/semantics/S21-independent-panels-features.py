"""Independent stdlib scheduling and finite public-feature identifiability audit."""
import argparse,gzip,hashlib,json,random,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def load(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 start=time.monotonic();base=ROOT/'research/results/campaign-01/semantics';baseline=load(base/'s15-shape-cache-v2/train_mixed.jsonl.gz');broad=load(a.data/'train_broad.jsonl.gz');dev=load(a.data/'development.jsonl.gz');conf=load(a.data/'confirmation.jsonl.gz');audit=json.loads((a.data/'audit.json').read_text());panels=json.loads((a.data/'panels.json').read_text());cells=['2x4','3x3','3x4','4x3','4x4','5x3','5x4'];cell=lambda r:f"{r['arity']}x{r['facts']}"
 def select(rows,counts,seed):
  out=[]
  for c in sorted(counts):
   ix=[i for i,r in enumerate(rows) if cell(r)==c];random.Random(seed+cells.index(c)).shuffle(ix);out.extend(ix[:counts[c]])
  return out
 old=select(baseline,{'3x3':1024,'4x3':512,'4x4':512},210021);assert old==audit['broad_old_original_indices'];assembled=[baseline[i] for i in old]
 for c in ('2x4','5x3','5x4'):assembled.extend(sorted([r for r in broad if cell(r)==c],key=lambda r:r['seed']))
 random.Random(210022).shuffle(assembled);assert [r['alpha_sha256'] for r in assembled]==[r['alpha_sha256'] for r in broad]
 panel=select(broad,{'3x3':32,'4x3':16,'4x4':16,'2x4':16,'5x3':24,'5x4':24},210023);assert panel==audit['broad_train_panel_indices']==[r['index'] for r in panels['broad']]
 for r in panels['broad']:assert all(broad[r['index']][k]==r[k] for k in ('seed','semantic_sha256','alpha_sha256'))
 oldpanel=base/'s17-calibration-selection/selection.json';assert sha(oldpanel)==audit['config']['original_panel']['sha256'];assert panels['original']==json.loads(oldpanel.read_text())['mixed']
 bit_tokens={};sequence_targets={};count=0;maximum=0
 for row in baseline+broad+dev+conf:
  tok=re.findall(r'\w+|[^\w\s]',row['text']);assert len(tok)==row['tokens'];maximum=max(maximum,len(tok));bits=[]
  for token in tok:
   h=hashlib.sha256(json.dumps(token,ensure_ascii=False,sort_keys=True).encode()).digest()[:8]
   assert h not in bit_tokens or bit_tokens[h]==token;bit_tokens[h]=token;bits.append(h)
  # Canonical scored targets: first public identical-token copy indices, finite values, typed/ordered edges.
  nodes=[(k,('copy',tok.index(v)) if k in ('ident','entity') else ('value',json.dumps(v,sort_keys=True))) for k,v in row['nodes']];target=json.dumps([nodes,row['edges']],sort_keys=True,separators=(',',':'));key=tuple(bits)
  assert key not in sequence_targets or sequence_targets[key]==target;sequence_targets[key]=target;count+=1
 assert count==audit['public_feature_audit']['rows'] and maximum==audit['public_feature_audit']['max_tokens']
 assert len(sequence_targets)==audit['public_feature_audit']['unique_feature_sequences']['float32']==audit['public_feature_audit']['unique_feature_sequences']['bfloat16']
 out=dict(scope='No model inference. Feature identifiability independently established using exact64 lexical bits and token sequence lengths; these bits are0/1 and exactly representable inFP32/BF16, so positional-channel rounding cannot merge distinct lexical sequences. Not a learned acquisition claim.',source_sha256=sha(Path(__file__)),audit_sha256=sha(a.data/'audit.json'),panels_sha256=sha(a.data/'panels.json'),source_encoder_sha256=sha(ROOT/'src/topoformer/thinking_language.py'),baseline_rows=len(baseline),all_rows=count,unique_lexical_sequences=len(sequence_targets),unique_tokens=len(bit_tokens),max_tokens=maximum,exact_shared_subset_order=True,exact_broad_shuffle=True,exact_broad_panel=True,original_panel_binding=True,feature_target_collisions=0,lexical_bit_collisions=0,cpu_wall_seconds=time.monotonic()-start)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
