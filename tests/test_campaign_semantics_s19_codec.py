import unittest
import os,json,gzip,hashlib,time,collections
from pathlib import Path
from tensegra.campaign_semantics_s19_codec import (
    BOS,NODE,EDGE,EOS,PAD,EMPTY,KINDS,ROLES,CodecError,decode_records,
    records_to_targets,canonicalize_public_copies,teacher_forcing_inputs,encode_row,canonical_edge_order)

class RecordCodecTests(unittest.TestCase):
    def setUp(self):
        self.nodes=[(NODE,KINDS.index('record'),0,-1,-1),(NODE,KINDS.index('ident'),-1,2,-1)]
        self.edge=(EDGE,0,1,ROLES.index('argument'),0)
        self.rows=self.nodes+[self.edge,(EOS,*EMPTY)]
    def decode(self,rows,**kwargs):
        return decode_records(rows,token_count=4,vocab_size=2,**kwargs)
    def test_roundtrip_order_and_identity(self):
        g=self.decode(self.rows)
        self.assertEqual(g['nodes'],[(12,0,-1),(5,-1,2)])
        self.assertEqual(g['edges'],[self.edge[1:]])
        reverse=self.nodes+[(EDGE,1,0,self.edge[3],0),(EOS,*EMPTY)]
        self.assertNotEqual(g,self.decode(reverse))
    def test_teacher_forcing_is_previous_only(self):
        self.assertEqual(teacher_forcing_inputs(self.rows),[(BOS,*EMPTY)]+self.rows[:-1])
    def test_public_copy_equality_only(self):
        rows=[(NODE,5,-1,3,-1),(EOS,*EMPTY)]
        self.assertEqual(canonicalize_public_copies(rows,['x','X','z','x'])[0][3],0)
        self.assertEqual(canonicalize_public_copies(rows,['x','X','z','X'])[0][3],1)
    def test_multirelation_and_multislot_lossless(self):
        edges=[(EDGE,0,1,2,-1),(EDGE,0,1,3,0),(EDGE,0,1,3,1)]
        self.assertEqual(len(self.decode(self.nodes+edges+[(EOS,*EMPTY)])['edges']),3)
    def test_historical_multislot_rejected(self):
        with self.assertRaisesRegex(CodecError,'multiple pair-slot'):
            records_to_targets(self.nodes+[(EDGE,0,1,2,-1),self.edge,(EOS,*EMPTY)],token_count=4,vocab_size=2)
    def test_generated_presence_no_gold_count(self):
        result=records_to_targets(self.rows,token_count=4,vocab_size=2)
        self.assertEqual(len(result['presence']),128)
        self.assertEqual(result['presence'].sum().item(),2)
    def test_pad_only_after_eos(self):
        self.assertEqual(self.decode(self.rows),self.decode(self.rows+[(PAD,*EMPTY)]*3))
    def test_invalid_cases(self):
        cases=[self.rows[:-1],self.rows+[self.edge],[(BOS,*EMPTY)]+self.rows,
            [(PAD,*EMPTY)]+self.rows,self.nodes+[self.edge,self.edge,(EOS,*EMPTY)],
            self.nodes+[(EDGE,0,2,3,0),(EOS,*EMPTY)],
            self.nodes+[(EDGE,0,1,3,32),(EOS,*EMPTY)],
            self.nodes+[(EDGE,0,1,99,0),(EOS,*EMPTY)],
            self.nodes+[self.edge,self.nodes[0],(EOS,*EMPTY)],
            [(NODE,5,0,2,-1),(EOS,*EMPTY)],[(NODE,5,-1,4,-1),(EOS,*EMPTY)],
            [(NODE,12,2,-1,-1),(EOS,*EMPTY)],[(EOS,0,-1,-1,-1)],
            [(NODE,True,0,-1,-1),(EOS,*EMPTY)]]
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(CodecError): self.decode(rows)
    def test_edge_permutation_preserves_semantics_not_argument_slots(self):
        import torch
        edges=[self.edge,(EDGE,1,0,3,1)]
        ordered=self.nodes+edges+[(EOS,*EMPTY)]
        permuted=self.nodes+edges[::-1]+[(EOS,*EMPTY)]
        self.assertTrue(canonical_edge_order(ordered))
        self.assertFalse(canonical_edge_order(permuted))
        first=records_to_targets(ordered,token_count=4,vocab_size=2)
        second=records_to_targets(permuted,token_count=4,vocab_size=2)
        self.assertTrue(all(torch.equal(first[k],second[k]) for k in first))
        with self.assertRaisesRegex(CodecError,'noncanonical edge order'):
            self.decode(permuted,require_canonical=True)
        swapped=self.nodes+[(EDGE,0,1,3,1),(EDGE,1,0,3,0)]+[(EOS,*EMPTY)]
        third=records_to_targets(swapped,token_count=4,vocab_size=2)
        self.assertFalse(torch.equal(first['slots'],third['slots']))
    def test_capacity_overflows(self):
        with self.assertRaises(CodecError): self.decode(self.rows,capacity=1)
        with self.assertRaises(CodecError): self.decode(self.rows,max_records=3)
        self.assertEqual(len(self.decode(self.rows,max_records=4)['nodes']),2)

class TrainCacheAudit(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('S19_TRAIN_CACHE'), 'explicit TRAIN cache audit only')
    def test_all_train_rows(self):
        import torch
        from tensegra.campaign_semantics_data import target
        from tensegra.semantic_scaling import tokens
        from tensegra.thinking_language import ActorInput,KINDS as old_kinds,ROLES as old_roles
        tick=time.monotonic(); torch.set_num_threads(2)
        path=Path(os.environ['S19_TRAIN_CACHE'])
        self.assertEqual(path.name,'train_mixed.jsonl.gz')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2')
        vocab=json.loads(Path(os.environ['S19_VOCAB_AUDIT']).read_text())['value_vocabulary']
        self.assertEqual(KINDS,old_kinds); self.assertEqual(ROLES,old_roles)
        counts=collections.Counter(); lengths=[]; digest=hashlib.sha256()
        for line in gzip.open(path,'rt'):
            row=json.loads(line); records=encode_row(row,vocab); lengths.append(len(records))
            tok=tokens(ActorInput(row['text'],()))
            actual=records_to_targets(records,token_count=len(tok),vocab_size=len(vocab))
            expected=target(row,vocab)
            for key in expected: self.assertTrue(torch.equal(actual[key],expected[key]),key)
            self.assertEqual(canonicalize_public_copies(records,tok),records)
            digest.update(json.dumps(records,separators=(',',':')).encode())
            counts['graphs']+=1; counts['nodes']+=len(row['nodes']); counts['edges']+=len(row['edges'])
        self.assertEqual(counts['graphs'],4096)
        self.assertEqual(digest.hexdigest(),'a153c648f1ff7817d6d1811cb5ef840acc6d0dd09f02698aac4a9518ff96c022')
        print(json.dumps(dict(scope='TRAIN only',counts=dict(counts),records_min=min(lengths),
            records_max=max(lengths),record_histogram=dict(collections.Counter(lengths)),
            cpu_wall_seconds=time.monotonic()-tick)))

if __name__=='__main__': unittest.main()
