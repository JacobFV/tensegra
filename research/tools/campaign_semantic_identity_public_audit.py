"""Independent S16 public-token mapping audit; CPU only, no graph or model."""
import argparse,collections,gzip,hashlib,json,re,time
from pathlib import Path
import torch
from topoformer.thinking_language import ActorInput
from topoformer.semantic_curriculum import encode_text
from topoformer.campaign_semantics_identity_contract import canonicalize_public_identifiers,restore_public_copy_tokens
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();torch.set_num_threads(2)
root=a.repo/'research/results/campaign-01/semantics';source=a.repo/'research/campaigns/extended-01/semantics/S16-public-audit.json';prior=json.loads(source.read_text())
paths=[root/'s01-data/reserved_confirmation.jsonl.gz',root/'s12-rename-case-audit/renamed_confirmation.jsonl.gz',root/'s01-data/train.jsonl.gz']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for key,path in zip(('original','renamed','train'),paths):assert sha(path)==prior['input_sha256'][key]
# Independent token classifier for these frozen renderer strings: all remaining
# alphabetic tokens after the fixed structural lexicon are identity arguments.
structural=set('You know these facts The pattern is What the unify parent and'.split())
lex=lambda text:re.findall(r'\w+|[^\w\s]',text)
texts=lambda path:[json.loads(line)['text'] for line in gzip.open(path,'rt')]
left,right=texts(paths[0]),texts(paths[1]);assert len(left)==len(right)==1024
inventory=(tuple('ABCDE'),tuple('alice bob carol dave erin frank'.split()))
assert set(sum((list(x) for x in inventory),[])) <= {t for text in texts(paths[2]) for t in lex(text)}
occurrences=0;features=hashlib.sha256();counts=collections.Counter()
for i,(x,y) in enumerate(zip(left,right)):
 normalized=[]
 for text in (x,y):
  original=lex(text);indices=[j for j,t in enumerate(original) if t not in structural and re.fullmatch('[A-Za-z][A-Za-z0-9_]*',t)]
  ranks=[{},{}];expected=original.copy()
  for j in indices:
   token=original[j];ns=0 if token.isupper() else 1;assert token.isupper() or token.islower()
   if token not in ranks[ns]:ranks[ns][token]=inventory[ns][len(ranks[ns])]
   expected[j]=ranks[ns][token]
  got=canonicalize_public_identifiers(ActorInput(text,()))
  assert list(got.identity_positions)==indices and list(got.canonical_tokens)==expected
  assert list(got.original_tokens)==original
  assert [original.index(t) for t in original]==[expected.index(t) for t in expected]
  assert restore_public_copy_tokens(got,range(len(original)))==tuple(original)
  assert restore_public_copy_tokens(got,[-1,0])==(None,'You')
  assert canonicalize_public_identifiers(got.public).public==got.public
  normalized.append(got)
 a0,b0=normalized;assert a0.public==b0.public
 af,n=encode_text(a0.public.text);bf,m=encode_text(b0.public.text)
 assert n==m==len(a0.original_tokens) and torch.equal(af,bf) and torch.equal(af.bfloat16(),bf.bfloat16())
 old=prior['rows'][i];assert old['index']==i
 for key,text in [('original_public_sha256',x),('renamed_public_sha256',y),('normalized_public_sha256',a0.public.text)]:assert old[key]==hashlib.sha256(text.encode()).hexdigest()
 h=hashlib.sha256(af.numpy().tobytes()).hexdigest();assert old['full_fp32_feature_sha256']==h;features.update(bytes.fromhex(h))
 assert old['tokens']==n and old['identifier_occurrences']==len(a0.identity_positions) and old['identities']==len(a0.original_to_canonical)
 occurrences+=len(a0.identity_positions);counts[str(a0.arity)+'x'+str(a0.facts)]+=1
out=dict(examples=1024,identifier_occurrences=occurrences,shape_counts=dict(counts),independent_mapping_equal=True,full_fp32_bf16_pair_equality=True,all_first_copy_positions_preserved=True,copy_restoration_no_repair=True,idempotent=True,per_row_feature_sha256_digest=features.hexdigest(),input_sha256={str(path.relative_to(a.repo)):sha(path) for path in paths+[source]},cpu_audit_wall_seconds=time.monotonic()-tick,scope='Public text and CPU encoding only; no target fields, actor construction, forward, normalization outcome or threshold selection.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
