"""CPU checkpoint metadata audit: meta tensors only, zero model forwards."""
import argparse,hashlib,json
from pathlib import Path
import torch
from topoformer.semantic_curriculum import SemanticCurriculumActor
from topoformer.campaign_semantics_s18_actor import S18Actor

PARENT_SHA='3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e'

def run(checkpoint,audit):
    with checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PARENT_SHA
    state=torch.load(checkpoint,map_location='meta',weights_only=True)
    vocabulary=json.loads(audit.read_text())['value_vocabulary']
    with torch.device('meta'):
        original=SemanticCurriculumActor(value_count=len(vocabulary),width=1024,capacity=128,workspace_rows=8,microsteps=2)
        expected=[(n,tuple(p.shape)) for n,p in original.named_parameters()]
        assert list(state['model'])==list(original.state_dict())
        for name,tensor in original.state_dict().items():assert tensor.shape==state['model'][name].shape
        groups=state['optimizer']['param_groups'];ids=[i for g in groups for i in g['params']]
        assert len(ids)==len(expected) and len(set(ids))==len(ids)
        assert set(ids)==set(state['optimizer']['state'])
        for (_,shape),i in zip(expected,ids):
            s=state['optimizer']['state'][i]
            assert tuple(s['exp_avg'].shape)==shape and tuple(s['exp_avg_sq'].shape)==shape
        for arm,kw in [('original',{}),('context',{}),('workspace_control',{'control_microsteps':10})]:
            model=S18Actor(arm=arm,value_count=len(vocabulary),width=1024,capacity=128,workspace_rows=8,**kw)
            assert [(n,tuple(p.shape)) for n,p in model.named_parameters()]==expected
    return dict(parent_sha256=PARENT_SHA,parameters=sum(p.numel() for p in original.parameters()),
        named_parameter_count=len(expected),optimizer_group_count=len(groups),optimizer_slot_count=len(ids),
        vocabulary_size=len(vocabulary),all_three_ordered_parameter_and_optimizer_shapes_match=True,
        numerical_adamw_replay='separate width16 mechanical test',model_forwards=0,device='meta (CPU-only deserialization)')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as f:json.dump(run(a.checkpoint,a.audit),f,indent=2);f.write('\n')
