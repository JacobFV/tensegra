import dataclasses
import unittest

from tensegra.tcn_data import build_tcn_example, build_tcn_corpus, verify_vendor_manifest


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

    def test_options_use_identifier_renderer_and_answer_index(self):
        from tensegra._vendor.tcn_language._structure import Ident
        from tensegra._vendor.tcn_language.languages import get_language
        for lesson in ('variable_binding','unification','set_operations'):
            example = build_tcn_example(lesson,17)
            index = example.privileged.answer_index
            for surface in example.public:
                self.assertEqual(surface.options[index], get_language(surface.language).render(Ident(example.privileged.answer)))
        self.assertEqual(get_language('spanish').render(Ident('carol')), 'carol')
        self.assertEqual(get_language('spanish').token('carol'), 'Carolina')

    def test_alpha_renaming_is_consistent_and_preserves_structure(self):
        mapping = dict(zip(('alice','bob','carol','dave','erin','frank'),('zali','zobo','zeca','zidu','zenu','zifa')))
        mapping.update({c: 'var'+c for c in 'ABCDE'})
        for lesson in ('variable_binding','unification'):
            base = build_tcn_example(lesson,17)
            renamed = build_tcn_example(lesson,17,identifier_renaming=mapping)
            self.assertEqual(renamed.privileged.answer,mapping[base.privileged.answer])
            self.assertEqual(base.privileged.answer_index,renamed.privileged.answer_index)
            self.assertEqual([n.kind for n in base.privileged.graph.nodes],[n.kind for n in renamed.privileged.graph.nodes])
            for old,new in zip(base.public,renamed.public):
                self.assertNotEqual(old.text,new.text)
                self.assertEqual(new.options,tuple(mapping.get(x,x) for x in old.options))
        scene = build_tcn_example('set_operations',17)
        scene_map = {name:'entity'+name for name in scene.public[0].options}
        renamed_scene = build_tcn_example('set_operations',17,identifier_renaming=scene_map)
        self.assertEqual(renamed_scene.privileged.answer,scene_map[scene.privileged.answer])
        properties = {'red','blue','green','yellow','purple','orange','cube','sphere','cone','prism','disc','rod'}
        self.assertEqual([n.value for n in scene.privileged.graph.nodes if n.value in properties],
                         [n.value for n in renamed_scene.privileged.graph.nodes if n.value in properties])
        self.assertEqual([n.value for n in scene.privileged.graph.nodes if n.kind == 'pred'],
                         [n.value for n in renamed_scene.privileged.graph.nodes if n.kind == 'pred'])
        with self.assertRaises(ValueError):
            build_tcn_example('set_operations',17,identifier_renaming={'red':'blue'})
        with self.assertRaises(ValueError):
            build_tcn_example('variable_binding',17,identifier_renaming={'alice':'bob'})
        with self.assertRaises(ValueError):
            build_tcn_example('variable_binding',17,identifier_renaming={'alice':'bind'})
