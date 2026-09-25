import unittest

from tensegra.semantic_graph import compile_term


class SemanticGraphTests(unittest.TestCase):
    def test_order_binding_identity_and_record_canonicalization(self):
        atom = lambda x: {'t': 'ident', 'v': x}
        term = {'t': 'record', 'fields': {
            'query': atom('X'),
            'substitution': {'t': 'pred', 'head': 'bind', 'args': [atom('X'), atom('alice')]}}}
        graph = compile_term(term)
        self.assertEqual(graph, compile_term({'t':'record','fields':dict(reversed(list(term['fields'].items())))}))
        entities = [n for n in graph.nodes if n.kind == 'entity']
        self.assertEqual({n.value for n in entities}, {'X', 'alice'})
        refs = [e for e in graph.edges if e.role == 'refers_to']
        self.assertEqual(len(refs), 3)
        self.assertEqual(len([e for e in graph.edges if e.role == 'binds']), 1)
        self.assertTrue(any(n.kind == 'scope' for n in graph.nodes))
        self.assertEqual([e.slot for e in graph.edges if e.role == 'argument'], [0, 1])
        swapped = {'t':'pred','head':'bind','args':[atom('alice'),atom('X')]}
        self.assertNotEqual(compile_term(swapped).digest(), compile_term(term['fields']['substitution']).digest())

    def test_all_term_types_and_validation(self):
        types = ['token', 'str', 'num', 'ident', 'nil']
        atoms = [{'t':t, 'v':v} for t,v in zip(types,['yes','text',3,'a',None])]
        terms = atoms + [{'t':t,'head':'h','args':atoms} for t in ['pred','rel','node']]
        terms += [{'t':t,'items':atoms} for t in ['tuple','list']]
        terms += [{'t':'app','fn':'f','args':{'x':atoms[0]}}]
        for term in terms:
            graph = compile_term(term)
            ids = {n.id for n in graph.nodes}
            self.assertEqual(len(ids),len(graph.nodes))
            self.assertTrue(all(e.source in ids and e.target in ids for e in graph.edges))
            self.assertEqual(graph.digest(),compile_term(term).digest())
        with self.assertRaises(ValueError):
            compile_term({'t':'unsupported'})
        with self.assertRaises(ValueError):
            compile_term({'t':'num','v':float('nan')})
