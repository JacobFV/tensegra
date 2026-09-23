import dataclasses
import unittest

from topoformer.tcn_data import build_tcn_example, build_tcn_corpus, verify_vendor_manifest


class TCNDataTests(unittest.TestCase):
    def test_fixed_construction_multiple_real_surfaces(self):
        for lesson in ('variable_binding','unification','set_operations'):
            example = build_tcn_example(lesson, 17)
            self.assertEqual(example, build_tcn_example(lesson,17))
            self.assertEqual(len({s.text for s in example.public}),3)
            for surface in example.public:
                self.assertEqual(set(dataclasses.asdict(surface)), {'language','text','options'})
                self.assertIn(example.privileged.answer, surface.options)
            one = build_tcn_example(lesson,17,languages=('symbols',))
            self.assertEqual(one.privileged.graph,example.privileged.graph)
            self.assertEqual(one.privileged.answer,example.privileged.answer)
            self.assertEqual(example.audit['semantic_digest'],example.privileged.graph.digest())

    def test_deterministic_corpus_and_source_integrity(self):
        a = build_tcn_corpus(lessons=('variable_binding',),count=3,seed=8)
        self.assertEqual(a,build_tcn_corpus(lessons=('variable_binding',),count=3,seed=8))
        self.assertEqual(len(a),3)
        self.assertTrue(verify_vendor_manifest())
        with self.assertRaises(ValueError):
            build_tcn_example('fake',0)
        with self.assertRaises(ValueError):
            build_tcn_example('variable_binding',0,languages=('fake',))
