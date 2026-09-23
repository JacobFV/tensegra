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
