"""Frozen two-read hardening interventions; oracle data is posthoc only."""
import math
import torch
from torch.nn import functional as F


POLICIES = ('unchanged', 'record_hard', 'destination_hard', 'both_hard')


@torch.no_grad()
def forward(model, batch, records, edge_indices, successor, policy='unchanged'):
    if policy not in POLICIES:raise ValueError(policy)
    b,n,_=batch.keys.shape;w=model.embed.embedding_dim;hd=w//model.heads
    h=model.embed(batch.values)
    ek=model.record_k(records.features).reshape(b,-1,model.heads,hd).transpose(1,2)
    nk=F.normalize(model.context_k(batch.keys),dim=-1)
    logits=[];routes=[];weights_out=[];masses=[];diagnostics={}
    bi=torch.arange(b,device=h.device)
    for t in reversed(range(batch.relations.shape[1])):
        rel=F.one_hot(batch.relations[:,t],3).float()[:,None].expand(-1,n,-1)
        desired=batch.instructions[:,t,None].expand(-1,n,-1)
        q=model.record_q(torch.cat((batch.keys,desired,rel),-1)).reshape(b,n,model.heads,hd).transpose(1,2)
        edge_soft=(q@ek.transpose(-1,-2)/math.sqrt(hd)*model.record_log_scale.exp().clamp(max=8)).softmax(-1)
        edge_argmax=edge_soft.argmax(-1)
        edge_weight=F.one_hot(edge_argmax,edge_soft.shape[-1]).to(edge_soft.dtype) if policy in ('record_hard','both_hard') else edge_soft
        returned=edge_weight@records.destination_keys[:,None]
        address=F.normalize(model.context_q(returned),dim=-1)
        destination_soft=(address@nk[:,None].transpose(-1,-2)*model.context_log_scale.exp().clamp(max=64)).softmax(-1)
        destination_argmax=destination_soft.argmax(-1)
        weights=F.one_hot(destination_argmax,n).to(destination_soft.dtype) if policy in ('destination_hard','both_hard') else destination_soft
        v=h.reshape(b,n,model.heads,hd).transpose(1,2)
        retrieved_heads=weights@v
        retrieved=retrieved_heads.transpose(1,2).reshape(b,n,w)
        h=retrieved+.1*model.update(retrieved)
        logits.append(model.readout(model.norm(h)))
        mean=weights.mean(1);routes.append(mean.argmax(-1));weights_out.append(mean)
        masses.append((mean*batch.adjacency[bi,batch.relations[:,t]]).sum(-1))

        # Everything below is observation only. Neither oracle nor indices above
        # affect record/destination scores, selections, values or hidden state.
        oracle=successor[:,t]
        er,es,ed=edge_indices.unbind(-1)
        record_mask=((er[:,None,:]==batch.relations[:,t,None,None]) &
                     (es[:,None,:]==torch.arange(n,device=h.device)[None,:,None]) &
                     (ed[:,None,:]==oracle[:,:,None]))
        assert record_mask.sum(-1).eq(1).all()
        selected_record_correct=record_mask[:,None].expand_as(edge_weight).gather(-1,edge_argmax[...,None]).squeeze(-1)
        selected_destination_correct=destination_argmax==oracle[:,None]
        record_destination=ed[:,None].expand(-1,model.heads,-1).gather(-1,edge_argmax)
        argmax_key=records.destination_keys[:,None].expand(-1,model.heads,-1,-1).gather(2,edge_argmax[...,None].expand(-1,-1,-1,model.key_dim))
        oracle_key=batch.keys.gather(1,oracle[...,None].expand(-1,-1,model.key_dim))[:,None]
        argmax_payload=v.gather(2,destination_argmax[...,None].expand(-1,-1,-1,hd))
        oracle_payload=v.gather(2,oracle[:,None,:,None].expand(-1,model.heads,-1,hd))
        payload_error=(retrieved_heads-argmax_payload).square().mean(-1)
        observations=dict(
            record_soft_target_mass=(edge_soft*record_mask[:,None]).sum(-1),
            record_used_target_mass=(edge_weight*record_mask[:,None]).sum(-1),
            record_soft_max_mass=edge_soft.max(-1).values,
            record_argmax_correct=selected_record_correct.float(),
            record_destination_argmax_correct=(record_destination==oracle[:,None]).float(),
            record_all_heads_same_destination=(record_destination==record_destination[:,:1]).all(1).float().mean(-1)[:,None].expand(-1,model.heads),
            destination_soft_target_mass=destination_soft.gather(-1,oracle[:,None,:,None].expand(-1,model.heads,-1,1)).squeeze(-1),
            destination_used_target_mass=weights.gather(-1,oracle[:,None,:,None].expand(-1,model.heads,-1,1)).squeeze(-1),
            destination_soft_max_mass=destination_soft.max(-1).values,
            destination_argmax_correct=selected_destination_correct.float(),
            destination_all_heads_same_node=(destination_argmax==destination_argmax[:,:1]).all(1).float().mean(-1)[:,None].expand(-1,model.heads),
            key_mse_to_record_argmax=(returned-argmax_key).square().mean(-1),
            key_mse_to_oracle=(returned-oracle_key).square().mean(-1),
            payload_mse_to_destination_argmax=payload_error,
            payload_argmax_energy=argmax_payload.square().mean(-1),
            payload_mse_to_oracle=(retrieved_heads-oracle_payload).square().mean(-1),
            payload_mse_correct_destination_sum=(payload_error*selected_destination_correct).sum(-1),
            correct_destination_count=selected_destination_correct.sum(-1),
        )
        for name,value in observations.items():
            # Preserve each event/reverse-step/head; average over nodes only.
            if value.ndim==3:value=value.mean(-1)
            diagnostics.setdefault(name,[]).append(value)
    return dict(logits=torch.stack(logits,1),routes=torch.stack(routes,1),
                weights=torch.stack(weights_out,1),edge_mass=torch.stack(masses,1),
                diagnostics={name:torch.stack(values,1) for name,values in diagnostics.items()})
