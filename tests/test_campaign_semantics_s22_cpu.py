"""S22 runner/controller boundary fixtures, CPU width16, no campaign outcomes."""
import dataclasses,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tensegra import campaign_semantics_s22 as m
from tensegra.campaign_semantics_s22_controller import NodeCount,NodeKinds,NodePrefix
from tensegra.campaign_semantics_s19_codec import KINDS,records_to_targets
from tensegra.semantic_curriculum import pack_graph
from tensegra.thinking_language import ActorInput,state_hash
class Boundary(unittest.TestCase):
 def fixture(self):
  records=((1,KINDS.index('entity'),-1,0,-1),);gold=records_to_targets(list(records)+[(3,-1,-1,-1,-1)],token_count=1,vocab_size=4)
  row=dict(seed=1,semantic_sha256='a'*64,graph_sha256='b'*64,arity=3,facts=3);example=dict(row=row,public=ActorInput('alice',()),nodes=records,gold=gold)
  old=dict(target=pack_graph(gold),seed=1,graph_sha256='b'*64,valid=True,complete=True,reason=None,exact_components=dict.fromkeys(('presence','kind','value','copy','edges','slots'),True));return records,example,{row['semantic_sha256']:old}
 def test_narrow_types(self):
  records,_,_=self.fixture()
  for policy,kind in zip(m.POLICIES,(NodeCount,NodeKinds,NodePrefix)):
   value=m.boundary(records,policy);self.assertIs(type(value),kind);self.assertFalse(any(k in dataclasses.asdict(value) for k in ('edges','slots','target','answer','row')))
  with self.assertRaises(ValueError):m.boundary(((2,0,0,0,-1),),'oracle_node_prefix')
 def test_public_metadata_scoring_boundary(self):
  torch.set_num_threads(2);records,example,reference=self.fixture();model=m.TypedRecordActor(value_count=4,width=16,heads=4,autocast_dtype='bfloat16');model.eval();before=state_hash(model)
  for policy,kind in zip(m.POLICIES,(NodeCount,NodeKinds,NodePrefix)):
   def spy(actor,publics,*,policy,boundaries):
    self.assertIs(actor,model);self.assertEqual(publics,[example['public']]);self.assertIs(type(boundaries[0]),kind)
    return [dict(records=list(records)+[(3,-1,-1,-1,-1)],status='eos',node_count=1,controller_stats={'policy':policy,'supplied_node_count':1})]
   with tempfile.TemporaryDirectory() as tmp,patch.object(m,'decode',spy),patch.object(torch.cuda,'synchronize',lambda:None):
    result=m.evaluate_policy(model,[example],list(range(4)),policy,reference,Path(tmp));self.assertEqual(result['complete'],1);self.assertEqual(result['controller_stats']['supplied_node_count'],1)
   self.assertEqual(state_hash(model),before)
 def test_invalid_retained(self):
  torch.set_num_threads(2);records,example,reference=self.fixture();model=m.TypedRecordActor(value_count=4,width=16,heads=4,autocast_dtype='bfloat16');model.eval()
  def spy(*args,**kwargs):return [dict(records=list(records),status='missing_eos',node_count=1,controller_stats={'supplied_node_count':1})]
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'decode',spy),patch.object(torch.cuda,'synchronize',lambda:None):
   result=m.evaluate_policy(model,[example],list(range(4)),'oracle_node_count',reference,Path(tmp));self.assertEqual(result['complete'],0);self.assertEqual(result['valid'],0);self.assertEqual(result['invalid_reasons'],{'missing_eos':1})
if __name__=='__main__':unittest.main()
