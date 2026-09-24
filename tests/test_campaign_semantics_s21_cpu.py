"""CPU differential mechanics; small synthetic public/record fixture, no cache reads."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from topoformer import campaign_semantics_s21 as m
from topoformer.campaign_semantics_s19_actor import TypedRecordActor
from topoformer.thinking_language import ActorInput,state_hash
class Differential(unittest.TestCase):
 def test_one_update_identical_to_s19(self):
  torch.set_num_threads(2)
  def factory(**kwargs):return TypedRecordActor(value_count=4,width=16,heads=4,node_capacity=128,max_records=160,max_slot=32,autocast_dtype='bfloat16')
  # entity copies public token0; EOS. This fixture contains no privileged inference call.
  from topoformer.campaign_semantics_s19_codec import KINDS
  x=dict(public=ActorInput('alice',()),records=[(1,KINDS.index('entity'),-1,0,-1),(3,-1,-1,-1,-1)],row={'nodes':[1],'edges':[]})
  train=[x]*4096;c=dict(device='cpu',job='profile',updates=1,checkpoints=[0,1]);evaluations=[]
  def evaluate(model,examples,vocab,out,label,update,batch):evaluations.append((label,update,len(examples),batch));return {'mechanical_fixture':True}
  with tempfile.TemporaryDirectory() as temp,patch.object(m,'TypedRecordActor',factory),patch.object(m,'evaluate',evaluate),patch.object(m,'worst_case_timing',lambda model,publics:{'seconds':0.}),patch.object(torch.cuda,'synchronize',lambda:None):
   result=m.train_arm(c,'original',list(range(4)),train,train[:8],train[:8],Path(temp)/'original')
   saved=torch.load(Path(temp)/'original/model-u1.pt',map_location='cpu',weights_only=True)
   torch.manual_seed(2101);reference=factory();optimizer=torch.optim.AdamW(reference.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
   initial=state_hash(reference);records=m.padded_records([x['records']]*8,'cpu')
   for group in optimizer.param_groups:group['lr']=m.learning_rate(1)
   optimizer.zero_grad();outputs=reference.teacher_forced([x['public']]*8,records);loss,_,_,_=m.field_losses(outputs,records);loss.backward();nn.utils.clip_grad_norm_(reference.parameters(),1.);optimizer.step()
   self.assertEqual(initial,result['initial_state_sha256']);self.assertEqual(state_hash(reference),result['final_state_sha256'])
   for key,tensor in reference.state_dict().items():self.assertTrue(torch.equal(tensor,saved['model'][key]))
   refstate=optimizer.state_dict()
   self.assertEqual(refstate['param_groups'],saved['optimizer']['param_groups'])
   for key,values in refstate['state'].items():
    for name,value in values.items():self.assertTrue(torch.equal(value,saved['optimizer']['state'][key][name]))
   generator=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=generator).tolist();indices,order,position=m.next_indices(order,0,generator,8)
   self.assertEqual(saved['order'],order);self.assertEqual(saved['position'],position);self.assertTrue(torch.equal(saved['schedule'],generator.get_state()));self.assertEqual([i for i,v in enumerate(saved['visits']) if v],sorted(indices))
   paired=m.train_arm(c,'broad',list(range(4)),train,train[:8],train[:8],Path(temp)/'broad')
   self.assertEqual(result['initial_state_sha256'],paired['initial_state_sha256']);self.assertEqual(result['final_state_sha256'],paired['final_state_sha256'])
   self.assertEqual(sum(saved['visits']),8);self.assertEqual(saved['update'],1);self.assertEqual(result['exposure'],dict(tokens=8,nodes=8,edges=0,records=16))
   self.assertEqual(evaluations,[('train',0,8,32),('development',0,8,32),('train',1,8,32),('development',1,8,32)]*2)
   self.assertEqual(set(result['losses'][0]['field_counts']),set(m.field_losses(outputs,records)[2]))
 def test_evaluator_public_only_greedy(self):
  from topoformer.campaign_semantics_s19_codec import KINDS,records_to_targets
  torch.set_num_threads(2);torch.manual_seed(2101);model=TypedRecordActor(value_count=4,width=16,heads=4,autocast_dtype='bfloat16');records=[(1,KINDS.index('entity'),-1,0,-1),(3,-1,-1,-1,-1)];public=ActorInput('alice',());gold=records_to_targets(records,token_count=1,vocab_size=4)
  example=dict(public=public,records=records,gold=gold,row=dict(seed=1,semantic_sha256='a'*64,graph_sha256='b'*64,arity=3,facts=3));seen=[]
  def greedy(publics):
   self.assertEqual(publics,[public]);self.assertTrue(all(isinstance(x,ActorInput) for x in publics));seen.extend(publics);return [dict(records=[],status='missing_eos',node_count=0)]
  with tempfile.TemporaryDirectory() as temp,patch.object(model,'greedy',greedy),patch.object(torch.cuda,'synchronize',lambda:None):
   result=m.evaluate(model,[example],4,Path(temp),'development',0,32)
  self.assertEqual(len(seen),1);self.assertEqual(result['valid'],0);self.assertEqual(result['complete'],0)
if __name__=='__main__':unittest.main()
