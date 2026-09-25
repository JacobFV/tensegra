"""Stage9 frozen checkpoints and paired public-prior inference; no actor gold inputs."""
import argparse,gzip,hashlib,json,time,resource
from pathlib import Path
import torch
from .belief_state import BeliefModel,collate,loss
from .belief_contracts import contract_episodes,empty_ledger_frames,ORIGINAL_CONDITIONS,NEW_CONDITIONS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def moved(batch,device):return {g:{k:v.to(device) for k,v in fields.items()} for g,fields in batch.items()}
def metrics(ps,qs):
 p=torch.tensor(ps);q=torch.tensor(qs);pred=p.argmax(-1); correct=q.gather(-1,pred[...,None]).squeeze(-1); imp=(p*(q==0)).sum(-1);l1=(p-q).abs().sum(-1); null=p[:,:,-1];truth=q[:,:,-1]
 frames=[]
 for t in range(p.shape[1]):
  support=(q[:,t]>0).sum(-1);conf=p[:,t].max(-1).values
  bins=[]
  for lo in range(10):
   mask=(conf>=lo/10)&(conf<((lo+1)/10) if lo<9 else conf<=1)
   bins.append(dict(lower=lo/10,count=int(mask.sum()),confidence=float(conf[mask].mean()) if mask.any() else None,expected_correctness=float(correct[:,t][mask].mean()) if mask.any() else None))
  frames.append(dict(frame=t,ambiguous=int((support>1).sum()),null_targets=int(truth[:,t].sum()),support_correct=int((correct[:,t]>0).sum()),posterior_l1=float(l1[:,t].mean()),impossible_mass=float(imp[:,t].mean()),null_signed_error=float((null[:,t]-truth[:,t]).mean()),null_mae=float((null[:,t]-truth[:,t]).abs().mean()),null_brier=float(((null[:,t]-truth[:,t])**2).mean()),no_match_true_positive=int(((pred[:,t]==p.shape[-1]-1)&(truth[:,t]==1)).sum()),no_match_false_positive=int(((pred[:,t]==p.shape[-1]-1)&(truth[:,t]==0)).sum()),max_l1=float(l1[:,t].max()),p99_l1=float(torch.quantile(l1[:,t],.99)),calibration=bins))
 return dict(count=len(ps),frames=frames,mean_l1=float(l1.mean()),mean_impossible=float(imp.mean()),final_support_accuracy=float((correct[:,-1]>0).float().mean()))

def evaluate(model,config,out,mode,seed):
 rows=[];start=time.perf_counter()
 for n in config['eval_candidates']:
  for cond in config['conditions']:
   for split,base_seed in config['split_seeds'].items():
    es=contract_episodes(config['eval_count'],base_seed+seed,n,cond)
    ps=[]; qs=[];empty=[]
    for offset in range(0,len(es),config['chunk']):
     b=moved(collate(es[offset:offset+config['chunk']]),next(model.parameters()).device)
     with torch.no_grad():p=model(b['public'])['logits'].softmax(-1).cpu().tolist()
     ps.extend(p);qs.extend(b['targets']['posterior'].cpu().tolist());empty.extend(empty_ledger_frames(b['public']).cpu().tolist())
    for arm in ('learned_prior','supplied_empty_prior'):
     pred=ps if arm=='learned_prior' else [[([1/n]*n+[0.]) if e else p for p,e in zip(row,mask)] for row,mask in zip(ps,empty)]
     raw=dict(posterior=pred,target=qs,empty_ledger=empty,event_seed=base_seed+seed,candidates=n,condition=cond,event_sha256=hashlib.sha256(json.dumps(es,sort_keys=True).encode()).hexdigest())
     name=f'{arm}-{split}-{n}-{cond}.json.gz'
     with gzip.open(out/name,'wt') as f:json.dump(raw,f,separators=(',',':'))
     m=metrics(pred,qs);m.update(mode=mode,seed=seed,arm=arm,split=split,candidates=n,condition=cond,raw=name,raw_sha256=sha(out/name),regime='iid' if n==8 else 'moderate_ood' if n==16 else 'additional_ood')
     m['passed']=m['count']>=512 and m['final_support_accuracy']>(.98 if n==8 else .95) and m['mean_l1']<.05 and m['mean_impossible']<.01
     rows.append(m)
 (out/'metrics.json').write_text(json.dumps(rows));return {'seconds':time.perf_counter()-start,'rows':len(rows)}

def diagnostic(model):
 rows=[]
 for n in (8,16):
  for cond in ('empty','clean','contradiction','full_retract'):
   b=moved(collate(contract_episodes(32,9000100,n,cond)),next(model.parameters()).device)
   features=[]
   hook=model.null.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().cpu().tolist()))
   output=model(b['public']);hook.remove();logits=output['logits'];p=logits.softmax(-1);q=b['targets']['posterior']
   losses=-(q*logits.log_softmax(-1)).sum(-1)
   # Gradients on head parameters reveal optimization pressure, not optimization history.
   for section,selected in [('initial',losses[:,0].mean()),('later',losses[:,1:].mean() if losses.shape[1]>1 else losses[:,0].mean())]:
    grad=torch.autograd.grad(selected,tuple(model.null.parameters()),retain_graph=True)
    rows.append(dict(candidates=n,condition=cond,section=section,loss=float(selected.detach()),null_head_gradient_l2=float(torch.sqrt(sum(g.square().sum() for g in grad))),null_output_bias_gradient=float(grad[-1]),initial_features=features[0],initial_logits=logits[:,0].detach().cpu().tolist(),initial_null=float(p[:,0,-1].mean().detach()),initial_l1=float((p[:,0]-q[:,0]).abs().sum(-1).mean().detach()),frame_losses=losses.mean(0).detach().cpu().tolist(),frame_null=p[:,:,-1].mean(0).detach().cpu().tolist()))
 return rows

def run(config,out):
 torch.set_num_threads(2);torch.manual_seed(config['seed']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 model=BeliefModel(config['mode'],width=1024,inner=2048,observation_id_features=True).to(config.get('device','cuda'))
 manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('belief_contracts.py'),Path(__file__).with_name('belief_state.py'))},composition_allowed=False,workspace_width=1024)
 if config.get('checkpoint'):
  model.load_state_dict(torch.load(Path(config['checkpoint']).expanduser(),map_location=config.get('device','cuda'),weights_only=True));manifest['checkpoint_sha256']=sha(Path(config['checkpoint']).expanduser())
 start=time.perf_counter()
 if config.get('steps',0):
  opt=torch.optim.AdamW(model.parameters(),lr=.0003);curve=[]
  for step in range(config['steps']):
   cond=('clean','partial','contradiction','retract','duplicate','reorder')[step%6]
   b=moved(collate(contract_episodes(32,config['seed']*10000000+step,8,cond)),config.get('device','cuda'))
   opt.zero_grad();ls=loss(model(b['public']),b);total=ls['posterior']+ls['compatibility'];total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if step%200==0:curve.append(dict(step=step,**{k:float(v.detach()) for k,v in ls.items()}));print(json.dumps(curve[-1]),flush=True)
  torch.save(model.state_dict(),out/'model.pt');manifest['checkpoint_sha256']=sha(out/'model.pt');(out/'curve.json').write_text(json.dumps(curve))
 manifest['training_seconds']=time.perf_counter()-start
 (out/'diagnostic.json').write_text(json.dumps(diagnostic(model)))
 if config.get('eval_count',0):manifest['evaluation']=evaluate(model,config,out,config['mode'],config['seed'])
 manifest.update(total_seconds=time.perf_counter()-start,rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out)
