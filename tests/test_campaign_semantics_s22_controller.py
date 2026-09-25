"""Small synthetic privileged-boundary tests; no corpus or frozen-model outcomes."""
import copy
import pytest
import torch
from tensegra.campaign_semantics_s19_actor import TypedRecordActor
from tensegra.campaign_semantics_s19_codec import BOS,NODE,EDGE,EOS,PAD,KINDS
from tensegra.campaign_semantics_s22_controller import NodeCount,NodeKinds,NodePrefix,decode
from tensegra.thinking_language import ActorInput,state_hash


def model(dtype=None):
 torch.manual_seed(22022)
 return TypedRecordActor(value_count=4,width=16,heads=4,node_capacity=8,max_records=16,autocast_dtype=dtype)


def core(rows):return [{k:r[k] for k in ('records','status','node_count')} for r in rows]


def force(m,**heads):
 with torch.no_grad():
  for name,index in heads.items():
   h=getattr(m,name+'_head');h.weight.zero_();h.bias.fill_(-100);h.bias[index]=100


@pytest.mark.parametrize('dtype',[None,'bfloat16'])
def test_independent_noop_matches_unchanged_greedy_and_preserves_state(dtype):
 m=model(dtype);publics=[ActorInput('alice bob',()),ActorInput('alice bob carol dave',())];before=state_hash(m)
 for forced in ({},{'type':2},{'type':1,'source':0,'target':0},{'type':0,'kind':KINDS.index('record'),'value':0}):
  force(m,**forced);snapshot=state_hash(m);expected=m.greedy(publics);actual=decode(m,publics)
  assert core(actual)==expected and state_hash(m)==snapshot and m.training
  assert all(r['controller_stats']['supplied_node_count'] is None for r in actual)
 assert before!=state_hash(m) # Only this fixture's explicit head edits mutate weights.


def test_exact_count_forces_nodes_then_preserves_edge_eos_ranking():
 m=model();force(m,type=2,kind=KINDS.index('record'),value=2)
 p=[ActorInput('a b c',())];r=decode(m,p,'oracle_node_count',[NodeCount(2)])[0]
 assert r['records']==[[NODE,KINDS.index('record'),2,-1,-1]]*2+[[EOS,-1,-1,-1,-1]]
 assert r['status']=='eos' and r['node_count']==2
 assert r['controller_stats']['forced_node_tags']==r['controller_stats']['tag_replacements']==2
 assert r['controller_stats']['forced_node_kinds']==0
 # When rawNODE wins, choose the higher learned EDGE/EOS, not forcedEOS.
 force(m,type=0,source=0,target=0,role=1,slot=0)
 with torch.no_grad():m.type_head.bias[1]=50;m.type_head.bias[2]=10
 r=decode(m,p,'oracle_node_count',[NodeCount(1)])[0]
 assert r['records'][1]==[EDGE,0,0,1,-1] and r['status']=='duplicate_edge'
 assert r['controller_stats']['node_suppression_replacements']==2


def test_forced_kind_selects_predicted_copy_or_value_without_gold_identity(monkeypatch):
 m=model();force(m,type=2,kind=KINDS.index('record'),value=3);base=m._heads
 def pointer(x,cache):
  out=base(x,cache);out['copy'].fill_(-100);out['copy'][:,:,1]=100;return out
 monkeypatch.setattr(m,'_heads',pointer)
 r=decode(m,[ActorInput('alice bob',())],'oracle_node_kinds',[NodeKinds((KINDS.index('ident'),KINDS.index('record')))])[0]
 assert r['records'][:2]==[[NODE,KINDS.index('ident'),-1,1,-1],[NODE,KINDS.index('record'),3,-1,-1]]
 assert r['controller_stats']['forced_node_kinds']==2 and r['controller_stats']['kind_replacements']==1
 assert r['controller_stats']['supplied_prefix_records']==0


@pytest.mark.parametrize('dtype,tol',[(None,4e-6),('bfloat16',.035)])
def test_prefix_is_previous_only_sequential_cache_and_full_logits_match(dtype,tol,monkeypatch):
 m=model(dtype);force(m,type=2);publics=[ActorInput('alice bob carol',())];prefix=((NODE,KINDS.index('ident'),-1,1,-1),(NODE,KINDS.index('record'),2,-1,-1));base=m._step;seen=[];logits=[]
 def observe(previous,cache,validate):
  seen.append(previous.clone());out,new=base(previous,cache,validate);logits.append(out);return out,new
 monkeypatch.setattr(m,'_step',observe)
 result=decode(m,publics,'oracle_node_prefix',[NodePrefix(prefix)])[0]
 assert result['records']==[list(r) for r in prefix]+[[EOS,-1,-1,-1,-1]]
 expected=torch.tensor([[[BOS,-1,-1,-1,-1],*prefix]])
 assert torch.equal(torch.stack(seen,1),expected)
 full=m(publics,expected)
 for i,step in enumerate(logits):
  for k in full:
   assert torch.allclose(full[k][:,i],step[k],atol=tol,rtol=tol)
   assert torch.equal(full[k][:,i].argmax(-1),step[k].argmax(-1))
 assert result['controller_stats']['supplied_prefix_records']==2
 # Changing future supplied prefix fields cannot enter an earlier cache input.
 early=[{k:v.clone() for k,v in out.items()} for out in logits[:2]]
 seen.clear();logits.clear();changed=(prefix[0],(NODE,KINDS.index('record'),0,-1,-1))
 decode(m,publics,'oracle_node_prefix',[NodePrefix(changed)])
 assert torch.equal(torch.stack(seen,1)[:,:2],expected[:,:2])
 assert all(torch.equal(a[k],b[k]) for a,b in zip(early,logits[:2]) for k in a)


@pytest.mark.parametrize('policy,boundaries',[
 ('oracle_node_count',[NodeCount(0),NodeCount(2)]),
 ('oracle_node_kinds',[NodeKinds(()),NodeKinds((KINDS.index('record'),KINDS.index('ident')))]),
 ('oracle_node_prefix',[NodePrefix(()),NodePrefix(((NODE,KINDS.index('record'),1,-1,-1),(NODE,KINDS.index('ident'),-1,2,-1)))])])
def test_batched_early_eos_padding_matches_individual(policy,boundaries):
 m=model();force(m,type=2,kind=KINDS.index('record'),value=0)
 publics=[ActorInput('alice',()),ActorInput('alice bob carol',())]
 together=decode(m,publics,policy,boundaries);separate=[decode(m,[p],policy,[b])[0] for p,b in zip(publics,boundaries)]
 assert together==separate and [r['node_count'] for r in together]==[0,2]


def test_invalid_edges_and_cap_are_retained_without_repair():
 m=model();force(m,type=1,source=7,target=7)
 r=decode(m,[ActorInput('alice',())],'oracle_node_count',[NodeCount(1)])[0]
 assert r['status']=='invalid_generated_node_reference' and r['records'][-1][1:3]==[7,7]
 m.max_records=2;force(m,source=0,target=0)
 r=decode(m,[ActorInput('alice',())],'oracle_node_count',[NodeCount(1)])[0]
 assert r['status']=='record_capacity_overflow' and r['records'][-1][0]==EDGE


def test_no_gold_edges_extra_metadata_or_wrong_policy_types():
 m=model();p=[ActorInput('alice bob',())]
 with pytest.raises(TypeError):NodeCount(1,edges=[])
 with pytest.raises(TypeError):decode(m,p,'oracle_node_count',[{'count':1,'edges':[]}])
 with pytest.raises(TypeError):decode(m,p,'oracle_node_count',[NodeKinds((1,))])
 with pytest.raises(TypeError):decode(m,p,'none',[NodeCount(1)])
 with pytest.raises(TypeError):decode(m,[{'text':'alice','graph':'poison'}])
 for count in (-1,True,9,1.5):
  with pytest.raises(ValueError):decode(m,p,'oracle_node_count',[NodeCount(count)])
 with pytest.raises(ValueError):decode(m,p,'oracle_node_prefix',[NodePrefix(((EDGE,0,0,0,-1),))])
 with pytest.raises(ValueError):decode(m,p,'oracle_node_prefix',[NodePrefix(((NODE,KINDS.index('ident'),-1,2,-1),))])
 with pytest.raises(ValueError):decode(m,p,'oracle_node_prefix',[NodePrefix(((NODE,KINDS.index('record'),-1,-1,-1),))])
 assert m.training


def test_encoder_receives_only_unchanged_public_inputs_and_called_once(monkeypatch):
 m=model();force(m,type=2);p=[ActorInput('alice bob',())];calls=[];base=m.begin
 def begin(publics):assert publics==p and all(type(x) is ActorInput for x in publics);calls.append(publics);return base(publics)
 monkeypatch.setattr(m,'begin',begin);before=state_hash(m)
 decode(m,p,'oracle_node_prefix',[NodePrefix(((NODE,KINDS.index('ident'),-1,1,-1),))])
 assert len(calls)==1 and state_hash(m)==before
