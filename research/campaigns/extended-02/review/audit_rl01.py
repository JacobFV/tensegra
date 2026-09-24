"""No-inference RL01 raw-record/tuple audit; pure stdlib, independent scorer."""
import argparse,base64,collections,gzip,hashlib,json,pathlib,time
p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args();tick=time.process_time()
r=a.root;data=r/'research/results/campaign-02/rl01-main-v1'
def read(p):
 return json.load(gzip.open(p,'rt') if p.suffix=='.gz' else open(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(data/'manifest.json.gz');c=read(data/'config.json');assert c==m['config'];assert m['status']=='completed'
source={k:sha(r/'src/topoformer'/k)==v for k,v in c['source_sha256'].items()};assert all(source.values())
dm=read(r/'research/campaigns/extended-02/semantics/rl01-data-manifest.json')
for k,v in c['inputs'].items():assert v['sha256']==dm['generated'][k]['sha256']
results=[];identities={};u0={};pair_records={};total=0
for arm in m['results']:
 assert read(data/arm['arm']/'manifest.json.gz')==arm
 assert arm['parent_optimizer_reset'];assert len(arm['visits'])==1024;assert set(arm['visits'])=={8}
 assert arm['curves'][0]['state_sha256']==c['parent_state_sha256']
 for curve in arm['curves']:
  assert curve['presentations']==curve['update']*8
  for split,entry in curve['metrics'].items():
   f=data/arm['arm']/entry['artifact'];assert sha(f)==entry['sha256'];payload=read(f);rows=payload['rows'];total+=len(rows)
   count=collections.Counter();reason=collections.Counter();fields=collections.Counter();pairs=[]
   alpha=[x['alpha_sha256'] for x in rows];assert len(set(alpha))==len(alpha)
   if split in identities:assert identities[split]==alpha
   else:identities[split]=alpha
   for row in rows:
    count['examples']+=1;g=row['target'];pr=row['prediction'];valid=row['valid'];count['valid']+=valid
    components={k:False for k in ('presence','kind','value','copy','edges','slots')}
    if valid:
     rec=row['canonical_records'];nodes=[x for x in rec if x[0]==1];edges=[tuple(x[1:]) for x in rec if x[0]==2]
     assert rec[-1]==[3,-1,-1,-1,-1] and len(rec)<=160
     assert len(edges)==len(set(edges));assert all(0<=i<len(nodes) and 0<=j<len(nodes) and 0<=role<13 and -1<=slot<32 for i,j,role,slot in edges)
     shape=g['edges']['shape'];assert shape==[128,128,13]
     predbytes=bytearray(128*128*13//8);slots={}
     for i,j,role,slot in edges:
      bit=(i*128+j)*13+role;predbytes[bit//8]|=1<<(bit%8)
      assert (i,j) not in slots or slots[i,j]==slot;slots[i,j]=slot
     assert bytes(predbytes)==base64.b64decode(pr['edges']['packed_b64'])
     gb=base64.b64decode(g['edges']['packed_b64']);goldpairs=set()
     for by,bits in enumerate(gb):
      while bits:
       lsb=bits&-bits;bit=by*8+lsb.bit_length()-1;pair=bit//13;goldpairs.add((pair//128,pair%128));bits^=lsb
     n=sum(g['presence']);components['presence']=len(nodes)==n
     components['kind']=all(i<len(nodes) and nodes[i][1]==g['kind'][i] for i in range(n))
     for field,col in [('value',2),('copy',3)]:components[field]=all(i<len(nodes) and nodes[i][col]==v for i,v in enumerate(g[field]) if v>=0)
     components['edges']=bytes(predbytes)==gb
     components['slots']=all(slots.get((i,j),-1)==g['slots'][i][j] for i,j in goldpairs)
    else:reason[row['reason']]+=1
    assert components==row['exact_components']
    complete=all(components.values());relations=components['edges'] and components['slots'];assert complete==row['complete'] and relations==row['exact_relations']
    count['complete']+=complete;count['exact_relations']+=relations
    for k,v in components.items():fields[k]+=v
    for t in row['teacher_forced_relations']:
     correct=t['gold']==t['predicted'];assert correct==t['correct'];count['teacher_'+t['kind']+'_count']+=1;count['teacher_'+t['kind']+'_correct']+=correct
    pairs.append(complete)
   summary={k:count[k] for k in ('examples','complete','exact_relations','valid')}
   for kind in ('edge','eos'):summary['teacher_'+kind]={'count':count['teacher_'+kind+'_count'],'correct':count['teacher_'+kind+'_correct']}
   assert summary==payload['summary'];assert all(entry[k]==v for k,v in summary.items())
   if curve['update']==0:
    fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if split in u0:assert u0[split]==fingerprint
    else:u0[split]=fingerprint
   pair_records[(arm['arm'],split,curve['update'])]=pairs
   results.append(dict(arm=arm['arm'],split=split,update=curve['update'],**summary,components=dict(fields),invalid_reasons=dict(reason),raw_sha256=sha(f)))
assert not(set(identities['known'])&set(identities['heldout']))
assert not(set(identities['train_panel'])&(set(identities['known'])|set(identities['heldout'])))
assert m['results'][0]['construction_stream_sha256']==m['results'][1]['construction_stream_sha256']
paired={}
for split in identities:
 x=pair_records[('all_records',split,1024)];y=pair_records[('relation_only',split,1024)]
 paired[split]=dict(collections.Counter(f'{int(i)}->{int(j)}' for i,j in zip(x,y)))
output=dict(scope='Archived raw-record and local-tuple reconstruction, no inference; one inherited parent; development only.',rows=total,cells=results,paired_final=paired,source_hashes_match=source,data_hashes_match_manifest=True,initial_rows_identical=True,streams_equal=True,visits_per_construction=8,unique_training=1024,presentations_per_arm=8192,parameters=[x['parameters'] for x in m['results']],parent=c['parent'],parent_state=c['parent_state_sha256'],local_checkpoint_verification=False,local_training_data_verification=False,cpu_seconds=time.process_time()-tick)
a.output.write_text(json.dumps(output,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in output.items() if k not in ('cells','source_hashes_match')}));print(json.dumps(results))
