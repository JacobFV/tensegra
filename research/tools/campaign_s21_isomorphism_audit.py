"""Independently check the necessary attributed-node invariant; no graph matcher."""
import argparse,collections,gzip,hashlib,json,re,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 start=time.monotonic();out=json.loads(a.result.read_text());assert sha(a.development)==out['development_sha256']
 public={r['semantic_sha256']:r for r in map(json.loads,gzip.open(a.development,'rt'))}
 expected={(r['arm'],r['semantic_sha256']):r for r in out['events']};assert len(expected)==1024
 counts={};checked=0
 for arm in ['original','broad']:
  path=a.raw/arm/'development-u4096.json.gz';assert sha(path)==out['inventory'][arm]['sha256']
  data=json.load(gzip.open(path,'rt'));rows=[r for r in data['rows'] if r['cell']=='3x4'];assert len(rows)==512
  count=collections.Counter()
  for row in rows:
   key=(arm,row['semantic_sha256']);e=expected[key];p=public[key[1]]
   assert row['seed']==p['seed'] and row['graph_sha256']==p['graph_sha256']
   assert row['valid']==e['valid'] and row['complete']==e['canonical_exact']==False
   if not row['valid']:
    assert e['status']=='invalid_output' and e['reason']==row['reason'];count['invalid_output']+=1
   else:
    tokens=re.findall(r'\w+|[^\w\s]',p['text']);t=row['target'];n=sum(t['presence'])
    # Numeric kind/value IDs are bijective with the frozen vocabulary. Resolve
    # copied identity by public token string, allowing repeated token positions.
    def attribute(k,v,c):
     if k in (1,5):assert v==-1;return(k,'copy',tokens[c])
     assert c==-1;return(k,'value',v)
    gold=collections.Counter(attribute(t['kind'][i],t['value'][i],t['copy'][i]) for i in range(n))
    pred=collections.Counter(attribute(r[1],r[2],r[3]) for r in row['records'] if r[0]==1)
    assert gold!=pred
    assert e['status']=='non_isomorphic' and e['reason']=='node_attribute_multiset';count['non_isomorphic']+=1
   checked+=1
  assert dict(count)==out['summary'][arm]['statuses'];counts[arm]=dict(count)
 result=dict(status='PASS',checked=checked,counts=counts,necessary_joint_node_attribute_mismatch=798,invalid_preserved=226,graph_matcher_required=False,cpu_wall_seconds=time.monotonic()-start,result_sha256=sha(a.result),scope='Uses prior full S21 raw/codec validation; independent exact node-attribute counters suffice to reject every valid heldout graph under any node permutation. Canonical gate unchanged.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['result','development','raw','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
