"""Mechanical public-interface fixtures, not reference benchmark results."""
import inspect
from functools import partial
import unittest
from dataclasses import replace

from topoformer.campaign02_references import ReferencePolicy, run_episode
from topoformer.campaign02_world import (Action, Item, Workshop, WorldSpec,
    action_catalog, protocol_executor)
from topoformer.campaign02_protocol import execute

# Mechanical fixtures isolate policy contracts; process isolation is separately
# tested by protocol tests. No empirical workload uses this direct executor.
protocol_executor = partial(protocol_executor, execute_call=execute)


def fixture(obstacle=False, work_limit=4096):
    # Cheap local choice a blocks b; valid plan uses c and b. No hidden target
    # enters policy, all costs revealed through inspect actions.
    return WorldSpec(items=(Item('a',0,1,1),Item('c',0,2,2),Item('b',1,1,1)),
        categories=(0,1), capacity=4,funds=4,incompatible=(('a','b'),),
        edges=((0,1,1),(1,2,1),(0,2,3)),start=0,destination=2,
        blocked_edge=(0,2) if obstacle else None, work_limit=work_limit)


class ReferencesTest(unittest.TestCase):
    def test_public_signature(self):
        self.assertEqual(list(inspect.signature(ReferencePolicy.choose).parameters),
                         ['self','o','catalog'])

    def test_catalog_only_and_complete_constraints(self):
        world = Workshop(fixture(), protocol_executor)
        policy = ReferencePolicy('always_tool')
        observation = world.observe()
        actions = []
        while not observation.done:
            catalog = action_catalog(observation)
            action = policy.choose(observation,catalog)
            self.assertIn(action,catalog)
            self.assertEqual(policy.choose_index(observation,catalog),catalog.index(action))
            actions.append(action)
            observation = world.step(action)
        self.assertTrue(world.evaluate()['verified_success'])
        constraints = [a.arguments['constraint'] for a in actions if a.kind=='add_constraint']
        self.assertEqual(constraints,['capacity','funds','incompatibility'])
        retrieved = [a.arguments['handle'] for a in actions if a.kind=='retrieve']
        used = [a.arguments['handle'] for a in actions if a.kind=='use_return']
        self.assertEqual(retrieved,used)
        self.assertEqual(len(set(retrieved)),2)

    def test_greedy_fails_without_uncharged_backtracking(self):
        result = run_episode(Workshop(fixture(),protocol_executor),ReferencePolicy('cheap'))
        self.assertFalse(result['verified_success'])
        self.assertEqual(result['work_units'],0)
        self.assertGreaterEqual(result['episode_cpu_seconds'],result['controller_cpu_seconds'])

    def test_solver_fallback_and_budget_refusal(self):
        good = run_episode(Workshop(fixture(),protocol_executor),ReferencePolicy())
        self.assertTrue(good['verified_success'])
        limited = run_episode(Workshop(fixture(work_limit=0),protocol_executor),ReferencePolicy())
        self.assertFalse(limited['verified_success'])
        self.assertEqual(limited['work_units'],0)

    def test_obstacle_replanning(self):
        easy = replace(fixture(obstacle=True), incompatible=(),
                       edges=((0,1,3),(1,2,3),(0,2,1)),travel_price=.01)
        result = run_episode(Workshop(easy,protocol_executor),ReferencePolicy())
        self.assertTrue(result['verified_success'])
        self.assertTrue(any(r['feedback'].get('status')=='obstacle' for r in result['trace']))
        self.assertTrue(any(r['action']['kind']=='move' for r in result['trace']))

    def test_cheap_multihop_is_available(self):
        spec = replace(fixture(),incompatible=(),edges=((0,1,1),(1,2,1)))
        result = run_episode(Workshop(spec,protocol_executor),ReferencePolicy('cheap'))
        self.assertTrue(result['verified_success'])
        self.assertEqual(result['work_units'],0)
        self.assertEqual(result['travel_distance'],2)

    def test_verified_timeout_incumbent_is_used(self):
        world = Workshop(fixture(),protocol_executor)
        policy = ReferencePolicy('always_tool')
        o = world.observe()
        while not any(r.get('primitive')=='constrained_subset' for r in o.records):
            o = world.step(policy.choose(o))
        records = tuple({**r,'status':'timeout','feasible_incumbent':True}
                        if r.get('primitive')=='constrained_subset' else r for r in o.records)
        o = replace(o,records=records)
        action = policy.choose(o)
        self.assertEqual(action.kind,'retrieve')
        retrieved = dict(o.retrieved)
        retrieved[action.arguments['handle']] = {'payload':'opaque'}
        o = replace(o,retrieved=retrieved)
        self.assertEqual(policy.choose(o).kind,'use_return')
        # A timeout without validated incumbent is not treated as success.
        invalid = tuple({**r,'certificate_valid':False} if r.get('primitive')=='constrained_subset' else r
                        for r in records)
        self.assertNotEqual(policy.choose(replace(o,records=invalid)).kind,'use_return')

    def test_reference_compute_tariff(self):
        result = run_episode(Workshop(fixture(),protocol_executor),ReferencePolicy('cheap'),2.5)
        self.assertEqual(result['compute_units'],result['steps']*2.5)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            ReferencePolicy('oracle')
