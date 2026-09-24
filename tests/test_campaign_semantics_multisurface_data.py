import unittest
from topoformer.campaign_semantics_multisurface_data import make_record,public_view,target,regenerate_allowed_train,run
from topoformer.campaign_semantics_data import compact_example,target as old_target
from topoformer.campaign_semantics_surface_contract import SurfaceCopyContractError
from topoformer.semantic_scaling import surface_input
from topoformer.tcn_data import build_tcn_example
import torch

class MultisurfacePreparationTests(unittest.TestCase):
    vocab=['<unknown>','"parent"','"unify"','null']
    def test_train_regeneration_and_public_projection(self):
        e=build_tcn_example('unification',900100001,difficulty=.5);public,_=surface_input(e,'english');old=compact_example(e,public)
        row=regenerate_allowed_train([old],self.vocab,{})[0]
        self.assertEqual(set(row['surfaces']),{'english','spanish'})
        self.assertEqual(public_view(row,'english').text,old['text'])
        gold=old_target(old,self.vocab);new=target(row,self.vocab,'english')
        self.assertTrue(all(torch.equal(gold[k],new[k]) for k in gold))
        poison={**row,'nodes':None,'edges':None,'semantic_sha256':'secret'}
        self.assertEqual(public_view(poison,'spanish'),public_view(row,'spanish'))
    def test_translated_copy_position_and_collision(self):
        e=build_tcn_example('unification',900100001,difficulty=.5,identifier_renaming={'alice':'red'});row=make_record(e,self.vocab,{})
        a=target(row,self.vocab,'english');b=target(row,self.vocab,'spanish')
        self.assertEqual(int(a['copy'][7]),11);self.assertEqual(int(b['copy'][7]),13)
        e=build_tcn_example('unification',900100001,difficulty=.5,identifier_renaming={'alice':'red','carol':'rojo'})
        with self.assertRaises(SurfaceCopyContractError):make_record(e,self.vocab,{})
    def test_bulk_generation_requires_explicit_release(self):
        with self.assertRaisesRegex(ValueError,'not been released'):run({'generation_status':'prepared_not_authorized'})

    def test_actual_token_feature_collision_rejected(self):
        from topoformer.campaign_semantics_multisurface_data import register_feature_target
        seen={}
        a=register_feature_target('alpha beta','target-a',seen)
        b=register_feature_target('alpha    beta','target-a',seen)
        self.assertEqual(a,b)
        with self.assertRaisesRegex(ValueError,'feature sequence'):register_feature_target('alpha    beta','target-b',seen)

    def test_slot_conflict_rejected_before_target_assignment(self):
        from dataclasses import replace
        from unittest.mock import patch
        from topoformer.semantic_graph import SemanticEdge
        e=build_tcn_example('unification',900100001,difficulty=.5)
        graph=e.privileged.graph;edge=next(x for x in graph.edges if x.slot is not None)
        bad=replace(graph,edges=graph.edges+(SemanticEdge(edge.source,edge.target,'item',edge.slot+1),))
        bad_example=replace(e,privileged=replace(e.privileged,graph=bad))
        with patch('topoformer.campaign_semantics_multisurface_data.targets',side_effect=AssertionError('target allocation before slot validation')):
            with self.assertRaisesRegex(ValueError,'multiple slot labels'):make_record(bad_example,self.vocab,{})
        # Several relation labels with the same slot remain representable.
        good=replace(graph,edges=graph.edges+(SemanticEdge(edge.source,edge.target,'item',edge.slot),))
        make_record(replace(e,privileged=replace(e.privileged,graph=good)),self.vocab,{})
