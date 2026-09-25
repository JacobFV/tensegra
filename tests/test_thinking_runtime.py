"""Protected-session tests have no tensor or training dependency."""
from dataclasses import FrozenInstanceError
import math
import unittest

from tensegra.thinking_runtime import Candidate, ProtectedSession, ValueRegister, PRIMITIVES


class ProtectedSessionTests(unittest.TestCase):
    def session(self):
        return ProtectedSession((ValueRegister('a', 7, 'integer', ('literal:a',)),
                                 ValueRegister('b', 3, 'integer', ('literal:b',))))

    def test_order_types_and_persistence(self):
        s = self.session()
        e = s.execute((Candidate('x', 'sub', ('a', 'b')),))[0]
        self.assertEqual((e.value, e.type, e.status), (4, 'integer', 'executed'))
        self.assertEqual(e.arguments, ('a', 'b'))
        self.assertEqual(e.provenance, ('literal:a', 'literal:b', 'candidate:x'))
        reverse = s.execute((Candidate('y', 'sub', ('b', 'a')),))[0]
        self.assertEqual(reverse.value, -4)
        after = s.execute((Candidate('z', 'mul', (e.register_id, 'b')),))[0]
        self.assertEqual(after.value, 12)
        self.assertEqual(s.registers[e.register_id].value, 4)
        self.assertEqual(s.execute((Candidate('c', 'compare', ('b', 'a')),))[0].value, True)

    def test_rejected_and_deferred_do_not_mutate(self):
        s = self.session()
        before = dict(s.registers)
        for c in (Candidate('x', 'unknown', ('a', 'b')), Candidate('x', 'add', ('future', 'b')),
                  Candidate('x', 'add', ('a',)), Candidate('x', 'add', ('a', 'b'), .1),
                  Candidate('x', 'add', ('a', 'b'), math.nan)):
            e = s.execute((c,))[0]
            self.assertIn(e.status, ('rejected', 'deferred'))
            self.assertEqual(dict(s.registers), before)
        self.assertEqual(s.events, ())

    def test_independent_and_shared_read_operations_execute(self):
        s = self.session()
        es = s.execute((Candidate('x', 'add', ('a', 'b')), Candidate('y', 'mul', ('a', 'b'))))
        self.assertEqual([e.status for e in es], ['executed', 'executed'])
        self.assertEqual([e.value for e in es], [10, 21])

    def test_snapshot_disallows_same_step_dependency(self):
        s = self.session()
        es = s.execute((Candidate('x', 'add', ('a', 'b')), Candidate('y', 'neg', ('result:x',))))
        self.assertEqual([e.status for e in es], ['executed', 'rejected'])
        self.assertEqual(s.execute((Candidate('y', 'neg', ('result:x',)),))[0].value, -10)

    def test_duplicate_idempotence_and_symmetric_conflict(self):
        s = self.session()
        c = Candidate('x', 'add', ('a', 'b'))
        es = s.execute((c, c))
        self.assertEqual([e.status for e in es], ['executed', 'duplicate'])
        before = dict(s.registers)
        self.assertEqual(s.execute((c,))[0].status, 'duplicate')
        self.assertEqual(s.execute((Candidate('x', 'sub', ('a', 'b')),))[0].status, 'conflict')
        self.assertEqual(dict(s.registers), before)
        self.assertEqual(len(s.events), 1)
        for proposals in ((c, Candidate('x', 'mul', ('a', 'b'))), (Candidate('x', 'mul', ('a', 'b')), c)):
            fresh = self.session()
            self.assertEqual([e.status for e in fresh.execute(proposals)], ['conflict', 'conflict'])
            self.assertEqual(len(fresh.registers), 2)

    def test_readiness_is_candidate_local(self):
        s = self.session()
        es = s.execute((Candidate('x', 'add', ('a', 'b'), .1), Candidate('y', 'mul', ('a', 'b'), .9)))
        self.assertEqual([e.status for e in es], ['deferred', 'executed'])

    def test_no_hidden_fields_or_mutable_values(self):
        with self.assertRaises(TypeError):
            ProtectedSession({'gold_actions': ()})
        with self.assertRaises(TypeError):
            Candidate('x', 'add', ('a', 'b'), gold=True)
        s = self.session()
        with self.assertRaises(TypeError):
            s.registers['x'] = ValueRegister('x', 1, 'integer')
        with self.assertRaises(FrozenInstanceError):
            s.registers['a'].value = 9
        with self.assertRaises(ValueError):
            ProtectedSession((ValueRegister('x', True, 'integer'),))

    def test_bounds_and_boolean_arithmetic_reject_atomically(self):
        s = ProtectedSession((ValueRegister('x', 100, 'integer'), ValueRegister('b', True, 'boolean')), max_abs_value=100)
        self.assertEqual(s.execute((Candidate('overflow', 'mul', ('x', 'x')),))[0].status, 'rejected')
        self.assertEqual(s.execute((Candidate('bool', 'add', ('x', 'b')),))[0].status, 'rejected')
        self.assertEqual(len(s.registers), 2)
        self.assertIn('compare', PRIMITIVES)

    def test_output_namespace_cannot_overwrite_initial(self):
        s = ProtectedSession((ValueRegister('a', 1, 'integer'), ValueRegister('result:x', 9, 'integer')))
        e = s.execute((Candidate('x', 'neg', ('a',)),))[0]
        self.assertEqual(e.status, 'conflict')
        self.assertEqual(s.registers['result:x'].value, 9)

class BoundaryTests(unittest.TestCase):
    def test_malformed_batch_is_atomic(self):
        s = ProtectedSession((ValueRegister('a', 2, 'integer'),))
        with self.assertRaises(TypeError):
            s.execute((Candidate('x', 'neg', ('a',)), {'future_trace': 'hidden'}))
        self.assertEqual(tuple(s.registers), ('a',))
        self.assertEqual(s.events, ())

    def test_mixed_numeric_types_and_schema_adapters(self):
        s = ProtectedSession((ValueRegister('a', 2, 'integer'), ValueRegister('b', .5, 'float')))
        e = s.execute((Candidate('x', 'add', ('a', 'b')),))[0]
        self.assertEqual((e.value, e.type, e.affected), (2.5, 'float', ('result:x',)))
        self.assertEqual(PRIMITIVES['add'].lower('latent', lambda x, schema: (x, schema.name)), ('latent', 'add'))
        self.assertEqual(PRIMITIVES['add'].lift(e, lambda event, schema: event.value), 2.5)
        with self.assertRaises(TypeError):
            PRIMITIVES['new'] = PRIMITIVES['add']

    def test_invalid_input_values_rejected(self):
        for value, kind in ((float('nan'), 'float'), (float('inf'), 'float'), ([], 'integer'), (10**1000, 'integer')):
            with self.assertRaises(ValueError):
                ProtectedSession((ValueRegister('a', value, kind),))

class LearnedTransitionTests(unittest.TestCase):
    def session(self):
        return ProtectedSession((ValueRegister('a', 7, 'integer'), ValueRegister('b', 3, 'integer')))

    def test_explicit_wrong_arithmetic_is_accepted_and_exact_unchanged(self):
        c = Candidate('x', 'add', ('a', 'b'))
        exact = self.session().execute((c,))[0]
        learned = self.session().execute_learned((c,), {'x': -99})[0]
        self.assertEqual((exact.value, exact.provenance), (10, ('a', 'b', 'candidate:x')))
        self.assertEqual((learned.status, learned.value, learned.type), ('executed', -99, 'integer'))
        self.assertIn('learned_transition:x', learned.provenance)
        self.assertEqual((exact.arguments, exact.register_id), (learned.arguments, learned.register_id))

    def test_missing_invalid_and_nonfinite_override_reject_without_fallback(self):
        c = Candidate('x', 'add', ('a', 'b'))
        for overrides in ({}, {'x': True}, {'x': 1.5}, {'x': float('nan')}, {'x': 10**20}, {'x': []}):
            s = self.session()
            e = s.execute_learned((c,), overrides)[0]
            self.assertEqual(e.status, 'rejected')
            self.assertEqual(tuple(s.registers), ('a', 'b'))
            self.assertEqual(s.events, ())

    def test_simultaneous_duplicates_and_mode_conflicts(self):
        s = self.session()
        a, b = Candidate('x', 'add', ('a', 'b')), Candidate('y', 'sub', ('a', 'b'))
        events = s.execute_learned((a, b, a), {'x': 20, 'y': 30})
        self.assertEqual([e.status for e in events], ['executed', 'executed', 'duplicate'])
        self.assertEqual(s.execute_learned((a,), {'x': 99})[0].value, 20)
        self.assertEqual(len(s.events), 2)
        self.assertEqual(s.execute((a,))[0].status, 'conflict')
        fresh = self.session()
        self.assertEqual([e.status for e in fresh.execute_learned((a, Candidate('x', 'mul', ('a', 'b'))), {'x': 3})], ['conflict', 'conflict'])
        self.assertEqual(len(fresh.registers), 2)

    def test_expected_float_and_comparison_types(self):
        for primitive, override, expected in (('add', 5.0, 'executed'), ('add', 5, 'rejected'), ('compare', False, 'executed'), ('compare', 0, 'rejected')):
            s = ProtectedSession((ValueRegister('a', 7.0, 'float'), ValueRegister('b', 3, 'integer')))
            self.assertEqual(s.execute_learned((Candidate('x', primitive, ('a', 'b')),), {'x': override})[0].status, expected)

    def test_no_exact_overflow_check_in_learned_path(self):
        # Exact product exceeds the bound; learned transition still accepts its
        # own bounded prediction. The exact result cannot act as a validity oracle.
        s = ProtectedSession((ValueRegister('a', 100, 'integer'),), max_abs_value=100)
        c = Candidate('x', 'mul', ('a', 'a'))
        self.assertEqual(s.execute_learned((c,), {'x': 1})[0].value, 1)


if __name__ == '__main__':
    unittest.main()
