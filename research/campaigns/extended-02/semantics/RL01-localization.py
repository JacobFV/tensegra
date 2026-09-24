"""CPU-only raw tuple/count reconstruction; no torch, checkpoint, or inference."""
import argparse, base64, collections, gzip, hashlib, json, time
from pathlib import Path

ROLES = ('contains','declares','refers_to','argument','item','binds','binding_scope',
         'field:query','field:substitution','field:pattern','field:fact','field:facts','field:scene')
FIELDS = ('tag','source','target','role','slot')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    with gzip.open(path,'rt') as f: return json.load(f)
def dense_gold_edges(target):
    e=target['edges']; assert e['bitorder']=='little'
    raw=base64.b64decode(e['packed_b64']); n,m,r=e['shape']; edges=set()
    for byte_i,byte in enumerate(raw):
        if not byte: continue
        for bit in range(8):
            if byte & (1<<bit):
                flat=byte_i*8+bit; source=flat//(m*r); dest=(flat//r)%m; role=flat%r
                assert source<n
                edges.add((2,source,dest,role,target['slots'][source][dest]))
    return edges

def main(root,out):
    start=time.monotonic();cpu=time.process_time();results={};bindings={};paired={}
    for arm in ('all_records','relation_only'):
        results[arm]={}
        for update in (0,256,1024):
            for split in ('train_panel','known','heldout'):
                path=root/arm/f'{split}-u{update}.json.gz';doc=load(path);bindings[str(path.relative_to(root))]=sha(path)
                rows=doc['rows']; c=collections.Counter();field=collections.Counter();role=collections.defaultdict(collections.Counter)
                reasons=collections.Counter();first=collections.Counter();examples=[]
                for row in rows:
                    c['examples']+=1;c['complete']+=row['complete'];c['exact_relations']+=row['exact_relations'];c['valid']+=row['valid']
                    if not row['valid']:reasons[row['reason']]+=1
                    local=row['teacher_forced_relations']; gold_edges=set()
                    for t in local:
                        assert t['correct']==(t['predicted']==t['gold'])
                        c['teacher_'+t['kind']+'_count']+=1;c['teacher_'+t['kind']+'_correct']+=t['correct']
                        if t['kind']=='edge':
                            gold,pred=t['gold'],t['predicted'];gold_edges.add((2,*gold[1:4],gold[4]-1))
                            name=ROLES[gold[3]];role[name]['count']+=1;role[name]['tuple_correct']+=t['correct']
                            for i,f in enumerate(FIELDS):
                                field[f+'_correct']+=gold[i]==pred[i]
                                role[name][f+'_wrong']+=gold[i]!=pred[i]
                            c['teacher_premature_eos']+=pred[0]==2
                    if update==1024 and split=='heldout': assert gold_edges==dense_gold_edges(row['target'])
                    predicted=[tuple(r) for r in row['records'] if r[0]==2];p=set(predicted)
                    c['gold_edges']+=len(gold_edges);c['emitted_edges']+=len(predicted)
                    c['correct_emitted_edges']+=len(p&gold_edges);c['missing_edges']+=len(gold_edges-p);c['spurious_edges']+=len(p-gold_edges)
                    ends_eos=row['records'][-1][0]==3
                    c['ended_eos']+=ends_eos;c['eos_before_gold_edge_count']+=ends_eos and len(predicted)<len(gold_edges)
                    c['eos_equal_gold_edge_count']+=ends_eos and len(predicted)==len(gold_edges)
                    c['eos_after_gold_edge_count']+=ends_eos and len(predicted)>len(gold_edges)
                    if row['valid']: assert row['exact_relations']==(p==gold_edges)
                    key=(split,update,row['semantic_sha256'])
                    paired.setdefault(key,{})[arm]=bool(row['complete'])
                    if len(examples)<3 and not row['complete']:
                        examples.append(dict(semantic_sha256=row['semantic_sha256'],reason=row['reason'],
                            emitted_edges=len(predicted),gold_edges=len(gold_edges),missing=sorted(gold_edges-p)[:5],spurious=sorted(p-gold_edges)[:5]))
                for k in ('examples','complete','exact_relations','valid'):assert c[k]==doc['summary'][k]
                for k in ('edge','eos'):
                    assert c['teacher_'+k+'_count']==doc['summary']['teacher_'+k]['count']
                    assert c['teacher_'+k+'_correct']==doc['summary']['teacher_'+k]['correct']
                results[arm][f'{split}-u{update}']=dict(counts=dict(c),teacher_field_correct=dict(field),
                    teacher_by_role={k:dict(v) for k,v in role.items()},invalid_reasons=dict(reasons),failure_examples=examples)
    pair_tables={}
    for (split,update,event),arms in paired.items():
        assert set(arms)=={'all_records','relation_only'}
        key=f'{split}-u{update}';table=pair_tables.setdefault(key,dict(both_correct=0,all_only=0,relation_only=0,both_wrong=0))
        a,b=arms['all_records'],arms['relation_only']
        table['both_correct' if a and b else 'all_only' if a else 'relation_only' if b else 'both_wrong']+=1
    manifest=load(root/'manifest.json.gz')
    for arm in manifest['results']:
        assert len(arm['visits'])==1024 and set(arm['visits'])=={8}
    assert manifest['results'][0]['construction_stream_sha256']==manifest['results'][1]['construction_stream_sha256']
    report=dict(scope='Raw prediction/count reconstruction and failure localization, not inference rerun. Dense gold edge tensors independently checked for 1024 final heldout graphs. Existing strict validity flags retained; codec itself not independently reimplemented.',
        bindings=bindings,results=results,paired_complete=pair_tables,paired_equal_stream=True,
        parameters=[r['parameters'] for r in manifest['results']],
        process_seconds=manifest['process_seconds'],cpu_seconds=manifest['cpu_seconds'],
        analysis_wall_seconds=time.monotonic()-start,analysis_cpu_seconds=time.process_time()-cpu)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(analysis_cpu_seconds=report['analysis_cpu_seconds'],rows=sum(r['counts']['examples'] for arm in results.values() for r in arm.values()))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.root,a.out)
