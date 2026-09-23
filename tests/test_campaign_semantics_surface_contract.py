import unittest
from dataclasses import replace
from topoformer.tcn_data import build_tcn_example
from topoformer.semantic_scaling import surface_input,targets
from topoformer.campaign_semantics_surface_contract import validate_surface_copy_contract,register_surface_target,SurfaceCopyContractError

class SurfaceContractTests(unittest.TestCase):
    def test_translated_identity_is_surface_local(self):
        e=build_tcn_example('unification',900100001,difficulty=.5,identifier_renaming={'alice':'red'})
        english,_=surface_input(e,'english');spanish,_=surface_input(e,'spanish')
        a=validate_surface_copy_contract(e.privileged.graph,english,'english');b=validate_surface_copy_contract(e.privileged.graph,spanish,'spanish')
        self.assertEqual(a['first_copy']['red'],11);self.assertEqual(b['first_copy']['red'],13)
        target=targets(e.privileged.graph,spanish,128,['<unknown>','"parent"','"unify"','null'],language='spanish')
        validate_surface_copy_contract(e.privileged.graph,spanish,'spanish',target=target)
        i=next(i for i,n in enumerate(e.privileged.graph.nodes) if n.kind=='entity' and n.value=='red')
        target['copy'][i]=0
        with self.assertRaises(SurfaceCopyContractError):validate_surface_copy_contract(e.privileged.graph,spanish,'spanish',target=target)
    def test_injective_canonical_aliases_can_collide_in_spanish(self):
        e=build_tcn_example('unification',900100001,difficulty=.5,identifier_renaming={'alice':'red','carol':'rojo'})
        english,_=surface_input(e,'english');spanish,_=surface_input(e,'spanish')
        validate_surface_copy_contract(e.privileged.graph,english,'english')
        with self.assertRaisesRegex(SurfaceCopyContractError,'collision'):validate_surface_copy_contract(e.privileged.graph,spanish,'spanish')
    def test_same_public_text_different_targets_rejected(self):
        e=build_tcn_example('unification',900100001,difficulty=.5);public,graph=surface_input(e,'english');seen={}
        register_surface_target(graph,public,'english',seen);register_surface_target(graph,public,'english',seen)
        nodes=list(graph.nodes);i=next(i for i,n in enumerate(nodes) if n.kind=='pred');nodes[i]=replace(nodes[i],value='different_operation');other=replace(graph,nodes=tuple(nodes))
        with self.assertRaisesRegex(SurfaceCopyContractError,'incompatible'):register_surface_target(other,public,'english',seen)
