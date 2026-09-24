"""Evaluate B-PINNs on the same Smooth cases and sensors as archived baselines."""
import argparse,fcntl,hashlib,json,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from baselines.methods.bpinns import BPINNConfig,parameter_count,pair_fields,hmc,posterior_potential
from baselines.common.data_adapter import _deferred_sensor_layout_id
from baselines.common.sensors import _sample_mask_seed,make_sensor_mask,_mask_id

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):
 t=p.with_suffix('.partial.json');t.write_text(json.dumps(a,indent=2,allow_nan=False));t.replace(p)
def main(a):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=json.loads((a.assets/'manifest.json').read_text());config=BPINNConfig()
 if a.pilot:config=BPINNConfig(draws=50,burn=25)
 protocols=json.loads(a.protocols.read_text()) if a.protocols else None
 for pde in a.pdes:
  pm=m['pdes'][pde];entry=pm['splits'][a.split];truth_path=a.assets/entry['file'];assert sha(truth_path)==entry['sha256']
  truth=torch.load(truth_path,map_location='cpu',weights_only=False)['ground_truth'];fields=torch.cat([truth['coef'],truth['sol']],1)
  n=fields.shape[-1];norm=pm['checkpoint_metadata']['normalizer'];mean=torch.tensor(norm['mean'],device=a.device).flatten();std=torch.tensor(norm['std'],device=a.device).flatten()
  axis=(torch.arange(n,device=a.device)+.5)/n if pde=='darcy' else torch.linspace(0,1,n,device=a.device)
  full_x=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
  for task in a.tasks:
   expected=protocols[pde.title()+'/'+task+'/PINN-Sparse'] if protocols else None
   for start in range(a.offset,a.offset+a.count,a.batch_size):
    ids=list(range(start,min(start+a.batch_size,a.offset+a.count)));out=a.output/pde/task/f'batch_{start:04d}';out.mkdir(parents=True,exist_ok=True)
    with (out/'claim.lock').open('a') as lock:
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:continue
     if (out/'receipt.json').exists():continue
     if (out/'error.json').exists():raise RuntimeError(str(out/'error.json'))
     began=time.time();masks=[];maskids=[];sensorids=[]
     for i in ids:
      sid=f'{pde}_test_10000-128-128.mat:{i}'
      mask=make_sensor_mask((1,n,n),500,'random_per_sample',_sample_mask_seed(1,'test',sid,0));masks.append(mask);maskids.append(_mask_id(mask[None]));sensorids.append(sid)
     # Archived runs store the sensor specification hash, rather than tensor hash.
     if expected and start==0 and a.split=='smooth':
      spec_id=_deferred_sensor_layout_id((1,n,n),num_sensors=500,seed=1,split='test',sample_id=sensorids[0],epoch=0,sensor_budget_mode='per_time')
      assert spec_id==expected['mask_id'],(spec_id,expected['mask_id'])
     index=torch.stack([torch.nonzero(x.flatten(),as_tuple=False).flatten() for x in masks]).to(a.device)
     observed=0 if task=='forward' else 1;true=fields[ids].to(a.device)
     obs_y=true[:,observed].flatten(1).gather(1,index);obs_x=full_x[index]
     seed=20260923+start+({'poisson':0,'helmholtz':10000,'darcy':20000}[pde])+(task=='inverse')*100000
     gen=torch.Generator(device=a.device).manual_seed(seed);b=len(ids)
     collocation=torch.rand((b,config.collocation,2),device=a.device,generator=gen)
     t=torch.linspace(0,1,config.boundary_per_side+2,device=a.device)[1:-1];zero=torch.zeros_like(t);one=torch.ones_like(t)
     boundary=torch.cat([torch.stack([zero,t],1),torch.stack([one,t],1),torch.stack([t,zero],1),torch.stack([t,one],1)],0)[None].expand(b,-1,-1)
     initial=torch.randn((b,2*parameter_count(config.hidden)),device=a.device,generator=gen)*.1
     potential=posterior_potential(pde,task,obs_x,obs_y,collocation,boundary,mean,std,config)
     def progress(it,acc,energy):print(pde,task,start,it,'accept',round(acc,3),'potential',round(energy,2),'sec',round(time.time()-began,1),flush=True)
     samples,diagnostics=hmc(potential,initial,config,gen,progress,compile_gradient=a.compile)
     running=torch.zeros((b,2,n*n),device=a.device,dtype=torch.float64);running2=running.clone()
     with torch.no_grad():
      for sample in samples:
       pred=pair_fields(sample,full_x[None].expand(b,-1,-1),mean,std,pde,config.hidden).double();running+=pred;running2+=pred.square()
     pred=(running/len(samples)).reshape(b,2,n,n);var=((running2-running.square()/len(samples))/(len(samples)-1)).clamp_min(0).reshape(b,2,n,n)
     errors=((pred-true.double()).flatten(2).norm(dim=2)/true.double().flatten(2).norm(dim=2)).cpu().tolist();assert torch.isfinite(pred).all()
     tensor_path=out/'posterior.pt';torch.save(dict(ids=ids,samples=samples.cpu(),mean=pred.cpu(),std=var.sqrt().cpu(),truth=true.cpu(),masks=torch.stack(masks),potential=diagnostics.pop('potential'),physical_mean=mean.cpu(),physical_std=std.cpu()),tensor_path)
     record=dict(pde=pde,task=task,split=a.split,ids=ids,global_ids=sensorids,mask_ids=maskids,errors=errors,target_errors=[x[1 if task=='forward' else 0] for x in errors],diagnostics=diagnostics,config=asdict(config),seed=seed,seconds=time.time()-began,truth_sha256=entry['sha256'],source_file=entry['source_file'],source_sha256=entry['source_sha256'],posterior_sha256=sha(tensor_path),driver_sha256=sha(__file__),method_sha256=sha(ROOT/'baselines/methods/bpinns.py'),official_reference='offical/B-PINNs/util.py and 2dnonlinear_inv.py',normalization='training-channel mean/std; positive Darcy coefficient uses training mean times softplus',physical_mean=mean.cpu().tolist(),physical_std=std.cpu().tolist(),target_field='u' if task=='forward' else 'a')
     save(out/'receipt.json',record);print('DONE',pde,task,start,'target mean',sum(record['target_errors'])/b,'seconds',record['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--protocols',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--pdes',nargs='+',default=['poisson','helmholtz','darcy']);p.add_argument('--tasks',nargs='+',default=['forward','inverse']);p.add_argument('--count',type=int,default=100);p.add_argument('--offset',type=int,default=0);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cuda');p.add_argument('--split',default='smooth');p.add_argument('--pilot',action='store_true');p.add_argument('--compile',action='store_true');main(p.parse_args())
