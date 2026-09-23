"""Stdlib tests, run via package stub to avoid importing torch."""
import dataclasses
import unittest
from topoformer.thinking_tasks import generate_episode, paired_contexts, audit_episode, render_frames

class ThinkingTaskTests(unittest.TestCase):
    def test_deep_exact_and_reproducible(self):
        for depth in (1, 2, 4, 8, 16, 32):
            episode = generate_episode(17, depth=depth)
            self.assertTrue(audit_episode(episode)['valid'])
            self.assertEqual(episode, generate_episode(17, depth=depth))
            self.assertLessEqual(abs(episode.gold.result), 64)
    def test_context_pair(self):
        a,b = paired_contexts(8, depth=3)
        self.assertEqual(a.gold.result,b.gold.result)
        self.assertNotEqual(a.gold.answer,b.gold.answer)
        self.assertEqual(a.public.frames,b.public.frames)
    def test_separation(self):
        episode = generate_episode(3)
        fields = set(dataclasses.asdict(episode.public))
        self.assertFalse(fields & {'gold','answer','result','trace','graph','readiness'})
        self.assertNotIn('gold',render_frames.__code__.co_varnames)
    def test_readiness_and_competition(self):
        episode = generate_episode(5,depth=2)
        values = {t.readiness for frame in episode.gold.readiness for t in frame}
        self.assertTrue({0.0,0.25,0.5,1.0} <= values)
        self.assertEqual(len(episode.gold.trace[0]),2)
    def test_distinct_matched_surfaces(self):
        items = [generate_episode(12,template=x) for x in ('canonical','infix','prefix','reordered','lexical')]
        self.assertEqual(len({x.gold.graph.digest() for x in items}),1)
        self.assertEqual(len({x.public.frames for x in items}),5)
    def test_bad_inputs(self):
        for kwargs in ({'depth':0},{'depth':33},{'template':'bad'},{'scheduling':'oracle'}):
            with self.assertRaises(ValueError): generate_episode(**kwargs)

    def test_runtime_trace_and_context_readiness(self):
        from topoformer.thinking_runtime import ProtectedSession
        from topoformer.thinking_tasks import readiness_targets
        episode = generate_episode(19,depth=8)
        session = ProtectedSession(episode.public.initial_values)
        for pair in episode.gold.trace[:-1]:
            events = session.execute(pair)
            self.assertTrue(all(e.status == 'executed' for e in events))
        targets = readiness_targets(episode,2,session.registers,context_visible=True)
        self.assertEqual(targets[-1].readiness,1.0)
        event, = session.execute(episode.gold.trace[-1])
        self.assertEqual(int(event.value),episode.gold.answer)

    def test_cross_motif_has_real_merge(self):
        from topoformer.thinking_runtime import ProtectedSession
        for depth in (2,8,32):
            e = generate_episode(42,depth=depth,motif='cross')
            prior = {'result:'+c.id for c in e.gold.trace[0]}
            self.assertEqual(set(e.gold.trace[1][1].arguments),prior)
            self.assertTrue(audit_episode(e)['valid'])
            self.assertLessEqual(audit_episode(e)['maximum_absolute_value'],64)
            runtime = ProtectedSession(e.public.initial_values)
            for group in e.gold.trace:
                self.assertTrue(all(x.status == 'executed' for x in runtime.execute(group)))

    def test_heldout_operator_composition(self):
        for seed in range(20):
            for motif in ('parallel','cross'):
                train = generate_episode(seed,depth=8,motif=motif,operator_composition='train')
                operations = { 'result:'+c.id:c.primitive for group in train.gold.trace for c in group }
                for group in train.gold.trace:
                    for c in group:
                        self.assertFalse(c.primitive == 'mul' and any(operations.get(a) == 'sub' for a in c.arguments))
                heldout = generate_episode(seed,depth=8,motif=motif,operator_composition='heldout')
                self.assertEqual(heldout.gold.trace[0][0].primitive,'sub')
                self.assertEqual(heldout.gold.trace[1][0].primitive,'mul')
                self.assertIn('result:'+heldout.gold.trace[0][0].id,heldout.gold.trace[1][0].arguments)
                self.assertTrue(audit_episode(heldout)['valid'])
        for args in ({'motif':'bad'},{'operator_composition':'bad'},{'depth':1,'motif':'cross'}, {'depth':1,'operator_composition':'heldout'}):
            with self.assertRaises(ValueError): generate_episode(**args)

    def test_completed_candidate_has_zero_readiness(self):
        from topoformer.thinking_tasks import readiness_targets
        e = generate_episode(4,depth=2)
        available = {v.id for v in e.public.initial_values}
        chosen = e.gold.trace[0][0]
        before = readiness_targets(e,0,available)
        matching = [x for x in before if x.candidate.id == chosen.id]
        self.assertTrue(matching)
        self.assertTrue(all(x.readiness > 0 for x in matching))
        available.add('result:'+chosen.id)
        after = readiness_targets(e,0,available,context_visible=True)
        self.assertTrue(all(x.readiness == 0 for x in after if x.candidate.id == chosen.id))
        self.assertEqual(after[-1].readiness,0.0)
        self.assertTrue(all(x.readiness == 0 for x in after if 'unbound' in x.candidate.arguments))
