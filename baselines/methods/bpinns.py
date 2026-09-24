"""Gaussian B-PINNs posterior with independent, batched HMC chains.

Adapted from offical/B-PINNs/util.py and 2dnonlinear_inv.py. Each realization
has two independent [2,16,16,1] tanh networks for its unknown a and u fields.
Only the observed field, PDE residual and boundary equations enter the
likelihood. Physical scales are fixed from training data, never test targets.
"""
from dataclasses import dataclass,asdict
import math
import torch

@dataclass(frozen=True)
class BPINNConfig:
    hidden: int = 16
    prior_std: float = 1.0
    likelihood_std: float = 0.1
    step_size: float = 0.001
    draws: int = 400
    burn: int = 200
    leapfrog_steps: int = 100
    collocation: int = 100
    boundary_per_side: int = 25
    adapt_step_size: bool = True
    target_acceptance: float = 0.65


def parameter_count(h=16):return 2*h+h+h*h+h+h+1


def network(theta,x,h=16,derivatives=False):
    """Analytic tanh derivatives; autograd still differentiates network weights."""
    b=theta.shape[0];start=0
    def part(*shape):
        nonlocal start
        n=math.prod(shape);v=theta[:,start:start+n].reshape(b,*shape);start+=n;return v
    w1,b1,w2,b2,w3,b3=part(h,2),part(h),part(h,h),part(h),part(1,h),part(1)
    z=x@w1.transpose(1,2)+b1[:,None];v=z.tanh()
    if derivatives:
        d=(1-v.square())[:,:,:,None]*w1[:,None]
        dd=(-2*v*(1-v.square()))[:,:,:,None]*w1[:,None].square()
    z2=v@w2.transpose(1,2)+b2[:,None];v2=z2.tanh()
    if derivatives:
        dz=torch.einsum('bnoj,bho->bnhj',d,w2)
        ddz=torch.einsum('bnoj,bho->bnhj',dd,w2)
        d2=(1-v2.square())[:,:,:,None]*dz
        dd2=(1-v2.square())[:,:,:,None]*(ddz-2*v2[:,:,:,None]*dz.square())
    y=v2@w3.transpose(1,2)+b3[:,None]
    if not derivatives:return y[...,0]
    dy=torch.einsum('bnhj,bh->bnj',d2,w3[:,0])
    ddy=torch.einsum('bnhj,bh->bnj',dd2,w3[:,0])
    return y[...,0],dy,ddy


def pair_fields(theta,x,mean,std,pde,h=16,derivatives=False):
    b,n,_=x.shape
    par=theta.reshape(b*2,-1);xx=x[:,None].expand(-1,2,-1,-1).reshape(b*2,n,2)
    result=network(par,xx,h,derivatives)
    if derivatives:
        y,dy,ddy=result;y=y.reshape(b,2,n);dy=dy.reshape(b,2,n,2);ddy=ddy.reshape(b,2,n,2)
    else:y=result.reshape(b,2,n)
    if pde=='darcy':
        # Positive coefficient, with characteristic scale fixed from training.
        z=y[:,0];scale=mean[0].clamp_min(1e-6);aa=scale*torch.nn.functional.softplus(z)
        uu=mean[1]+std[1]*y[:,1]
        if derivatives:
            sig=z.sigmoid();da=scale*sig[:,:,None]*dy[:,0]
            dda=scale*(sig[:,:,None]*ddy[:,0]+(sig*(1-sig))[:,:,None]*dy[:,0].square())
            return torch.stack([aa,uu],1),torch.stack([da,std[1]*dy[:,1]],1),torch.stack([dda,std[1]*ddy[:,1]],1)
        return torch.stack([aa,uu],1)
    yy=mean[None,:,None]+std[None,:,None]*y
    if derivatives:return yy,std[None,:,None,None]*dy,std[None,:,None,None]*ddy
    return yy


def hmc(potential,initial,config,generator,progress=None,compile_gradient=False):
    """Euclidean HMC, identity mass, one independent accept/reject per row."""
    assert 0<=config.burn<config.draws and config.leapfrog_steps>0
    def value_grad(x):
        with torch.enable_grad():
            q=x.detach().requires_grad_(True);v=potential(q)
            g=torch.autograd.grad(v.sum(),q)[0]
        return v.detach(),g.detach()
    if compile_gradient:
        def total_and_values(x):
            values=potential(x)
            return values.sum(),values
        gradient_and_value=torch.compile(torch.func.grad_and_value(total_and_values,has_aux=True),fullgraph=True,mode='reduce-overhead')
        reference_v,reference_g=value_grad(initial)
        compiled_g,(_,compiled_v)=gradient_and_value(initial)
        torch.testing.assert_close(compiled_v,reference_v,atol=1e-3,rtol=2e-5)
        torch.testing.assert_close(compiled_g,reference_g,atol=1e-3,rtol=2e-4)
        def value_grad(x):
            gradient,(_,values)=gradient_and_value(x)
            # Clone outputs so subsequent CUDA graph replays cannot overwrite them.
            return values.detach().clone(),gradient.detach().clone()
    q=initial.detach();v,g=value_grad(q)
    if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite initial B-PINNs posterior')
    samples=[];accepted=torch.zeros(len(q),device=q.device);postaccepted=accepted.clone();energies=[]
    eps=torch.full((len(q),1),config.step_size,device=q.device,dtype=q.dtype)
    log_eps=eps.log();log_average=log_eps.clone();center=(10*eps).log();hbar=torch.zeros_like(eps)
    for iteration in range(config.draws):
        momentum=torch.randn(q.shape,device=q.device,dtype=q.dtype,generator=generator)
        prop=q.clone();p=momentum-0.5*eps*g
        for step in range(config.leapfrog_steps):
            prop=prop+eps*p;vp,gp=value_grad(prop)
            p=p-eps*gp*(0.5 if step==config.leapfrog_steps-1 else 1.0)
        difference=v+0.5*momentum.square().sum(1)-vp-0.5*p.square().sum(1)
        decision=torch.isfinite(difference)&(torch.log(torch.rand(len(q),device=q.device,dtype=q.dtype,generator=generator))<difference)
        q=torch.where(decision[:,None],prop,q);v=torch.where(decision,vp,v);g=torch.where(decision[:,None],gp,g)
        accepted+=decision
        if config.adapt_step_size and iteration<config.burn:
            # Dual averaging during warmup only; frozen throughout retained draws.
            number=iteration+1
            probability=torch.where(torch.isfinite(difference),difference.clamp_max(0).exp(),torch.zeros_like(difference))[:,None]
            rate=1.0/(number+10.0)
            hbar=(1-rate)*hbar+rate*(config.target_acceptance-probability)
            log_eps=(center-math.sqrt(number)/.05*hbar).clamp(-16.,-2.)
            weight=number**(-.75)
            log_average=weight*log_eps+(1-weight)*log_average
            eps=(log_average if number==config.burn else log_eps).exp()
        if iteration>=config.burn:
            samples.append(q.clone());postaccepted+=decision;energies.append(v.clone())
        if progress and (iteration+1)%25==0:progress(iteration+1,float(accepted.mean()/(iteration+1)),float(v.mean()))
    return torch.stack(samples),dict(acceptance=(accepted/config.draws).cpu().tolist(),posterior_acceptance=(postaccepted/(config.draws-config.burn)).cpu().tolist(),step_size=eps[:,0].cpu().tolist(),potential=torch.stack(energies).cpu(),config=asdict(config))


def posterior_potential(pde,task,obs_x,obs_y,collocation,boundary,mean,std,config):
    observed=0 if task=='forward' else 1
    def potential(theta):
        obs=pair_fields(theta,obs_x,mean,std,pde,config.hidden)[:,observed]
        obs_residual=(obs-obs_y)/std[observed]
        y,d,dd=pair_fields(theta,collocation,mean,std,pde,config.hidden,True)
        lap=dd[:,1].sum(-1)
        if pde=='poisson':residual=(lap-y[:,0])/std[0]
        elif pde=='helmholtz':residual=(lap+y[:,1]-y[:,0])/std[0]
        elif pde=='darcy':residual=-(d[:,0]*d[:,1]).sum(-1)-y[:,0]*lap-1.0
        else:raise ValueError(pde)
        if pde=='helmholtz':
            by,bd,bdd=pair_fields(theta,boundary,mean,std,pde,config.hidden,True)
            # At a boundary, the generator keeps the tangential second derivative
            # and replaces the normal 1-D Laplacian row by an identity row.
            quarter=boundary.shape[1]//4
            tangent=torch.cat([bdd[:,1,:2*quarter,1],bdd[:,1,2*quarter:,0]],1)
            bc=(tangent+2*by[:,1])/std[0]
        else:bc=pair_fields(theta,boundary,mean,std,pde,config.hidden)[:,1]/std[1]
        return 0.5*theta.square().sum(1)/(config.prior_std**2)+0.5*(obs_residual.square().sum(1)+residual.square().sum(1)+bc.square().sum(1))/(config.likelihood_std**2)
    return potential
