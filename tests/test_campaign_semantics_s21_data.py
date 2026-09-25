"""Mechanical fixtures only: no campaign data generation or confirmation reads."""
import unittest
from unittest.mock import patch
from tensegra import campaign_semantics_s21_data as data

class S21DataTests(unittest.TestCase):
    def test_global_cell_offsets(self):
        self.assertEqual(data.CELLS,((2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4)))
        self.assertEqual(sum(data.OLD_COUNTS.values())+sum(data.NEW_COUNTS.values()),4096)
        self.assertEqual(sum(data.PANEL_COUNTS.values()),128)
        self.assertNotIn((3,4),data.OLD_COUNTS|data.NEW_COUNTS)
        self.assertNotIn((2,3),data.CELLS)
    def test_alpha_renaming_preserves_equality_not_order(self):
        nodes=[['ident','x'],['ident','y'],['ident','x']]
        edges=[[0,1,'argument',0],[0,2,'argument',1]]
        self.assertEqual(data.alpha(nodes,edges),data.alpha([['ident','q'],['ident','r'],['ident','q']],edges[::-1]))
        self.assertNotEqual(data.alpha(nodes,edges),data.alpha([['ident','q'],['ident','r'],['ident','r']],edges))
        self.assertNotEqual(data.alpha(nodes,edges),data.alpha(nodes,[[0,1,'argument',1],[0,2,'argument',0]]))
    def test_selection_deterministic_original_indices(self):
        rows=[dict(arity=a,facts=f,original=i) for a,f in data.CELLS for i in range(40)]
        x=data.select_indices(rows,data.PANEL_COUNTS,210023)
        self.assertEqual(x,data.select_indices(rows,data.PANEL_COUNTS,210023))
        self.assertEqual(len(set(x)),128)
        self.assertEqual({c:sum(data.cell(rows[i])==c for i in x) for c in data.PANEL_COUNTS},data.PANEL_COUNTS)
    def test_reservation_exclusion_cap_and_wrong_cell(self):
        def row(arity,seed):
            return dict(arity=arity,facts=4,alpha_sha256=str(seed%3),text=str(seed%3),nodes=[],edges=[])
        seen={'0'}
        with patch.object(data,'MAX_ATTEMPTS',5):
            rows,stats=data.reserve((2,4),3,0,seen,{},row)
        self.assertEqual(len(rows),2);self.assertEqual(stats['attempts'],5)
        self.assertEqual(stats['accepted'],2)
        with patch.object(data,'MAX_ATTEMPTS',5):
            rows,stats=data.reserve((2,3),1,0,set(),{},row)
        self.assertEqual(rows,[]);self.assertEqual(stats['other_fact_count'],5)
    def test_public_collision_rejected(self):
        def row(arity,seed):
            return dict(arity=arity,facts=4,alpha_sha256=str(seed),text='same',nodes=[['num',seed]],edges=[])
        with self.assertRaisesRegex(ValueError,'incompatible targets'):
            data.reserve((2,4),2,0,set(),{},row)
    def test_clearance_required_before_filesystem_mutation(self):
        with self.assertRaisesRegex(ValueError,'clearance'):
            data.build({})
    def feature_row(self,text='x',value=1):
        return dict(text=text,nodes=[['num',value]],edges=[])
    def test_fp32_feature_target_collision_rejected(self):
        import torch
        audit=data.PublicFeatureAudit();x=torch.zeros(1,1,68)
        with patch('tensegra.semantic_curriculum.encode_text',return_value=(x,1)),patch('tensegra.campaign_semantics_s19_codec.encode_row',side_effect=lambda row,v:[row['nodes'][0][1]]):
            audit.add(self.feature_row())
            with self.assertRaisesRegex(ValueError,'feature sequence'):
                audit.add(self.feature_row(value=2))
    def test_bf16_only_feature_target_collision_rejected(self):
        import torch
        audit=data.PublicFeatureAudit();x=torch.zeros(1,1,68);x[:,:,64]=1.
        y=x.clone();y[:,:,64]=1.0001
        with patch('tensegra.semantic_curriculum.encode_text',side_effect=[(x,1),(y,1)]),patch('tensegra.campaign_semantics_s19_codec.encode_row',side_effect=lambda row,v:[row['nodes'][0][1]]):
            audit.add(self.feature_row())
            with self.assertRaisesRegex(ValueError,'feature sequence'):
                audit.add(self.feature_row(value=2))
        self.assertEqual(len(audit.seen['float32']),2)
        self.assertEqual(len(audit.seen['bfloat16']),1)
    def test_lexical_collision_rejected(self):
        import torch
        audit=data.PublicFeatureAudit();x=torch.zeros(1,1,68)
        with patch('tensegra.semantic_curriculum.encode_text',return_value=(x,1)),patch('tensegra.campaign_semantics_s19_codec.encode_row',return_value=[1]):
            audit.add(self.feature_row('x'))
            with self.assertRaisesRegex(ValueError,'lexical feature collision'):
                audit.add(self.feature_row('y'))
    def test_feature_truncation_rejected(self):
        import torch
        for features,length in [(torch.zeros(1,1,68),1),(torch.zeros(1,1,68),2)]:
            with patch('tensegra.semantic_curriculum.encode_text',return_value=(features,length)):
                with self.assertRaisesRegex(ValueError,'truncation'):
                    data.PublicFeatureAudit().add(self.feature_row('x y'))
    def test_independent_answer(self):
        nodes=[['record',None],['pred','parent'],['pred','unify'],['list',None],['ident','X'],['ident','bob'],['pred','parent'],['ident','alice'],['ident','bob']]
        edges=[[0,1,'field:pattern',None],[0,2,'field:query',None],[0,3,'field:facts',None],
               [1,4,'argument',0],[1,5,'argument',1],[2,4,'argument',0],[3,6,'item',0],
               [6,7,'argument',0],[6,8,'argument',1]]
        self.assertEqual(data.independent_answer(nodes,edges),'alice')
        # Equality matters: replacing the fixed bob with carol destroys the match.
        nodes[-1][1]='carol'
        with self.assertRaisesRegex(ValueError,'nonunique'): data.independent_answer(nodes,edges)

if __name__=='__main__':unittest.main()
