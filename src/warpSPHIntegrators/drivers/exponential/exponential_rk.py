"""Numerical stepping for exponential_rk; configuration and coefficients are supplied separately."""
from dataclasses import fields


import math

import torch

from warpSPHIntegrators.schemes.coefficients.exponential import PROGRAMS


from warpSPHIntegrators.state.fields import flatten_integrated, unflatten_integrated, get_reference_state, integrated_field_names

from warpSPHIntegrators.state.protocol import BaseIntegrationSystem

from warpSPHIntegrators.adapters.rhs import resolve

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.solvers.nonlinear import SolveDiagnostics

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step, split_return

from warpSPHIntegrators.drivers.exponential.legacy import _arnoldi, _linear_map_reattach


from warpSPHIntegrators.solvers.matrix_functions import operator_function
from warpSPHIntegrators.schemes.families.exponential_rk import ExponentialQuadrature, SEMILINEAR

def step(self,state,dt,f,*args,f_t=None,ft_eps=None,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    if f_t is not None and not callable(f_t):raise TypeError('f_t must be callable')
    if ft_eps is not None and (isinstance(ft_eps,bool) or not math.isfinite(ft_eps) or ft_eps<=0):raise ValueError('positive ft_eps required')
    semilinear=self.method in SEMILINEAR
    resolved=resolve(f,scheme_name=self.name,need_linear=semilinear)
    base=snapshot_state(state)
    hook=getattr(base.finalize,'__func__',None)
    if not kwargs.get('_composition_substep',False) and hook is not BaseIntegrationSystem.finalize and not getattr(base,'exponential_compatible',False):
        raise TypeError('Custom finalize requires exponential_compatible=True and ExponentialQuadrature support')
    initializeSystem(base,dt,*args,**kwargs);t=float(base.t);y=flatten_integrated(base)
    if not math.isfinite(t):raise ValueError('finite initial time required')
    ref=get_reference_state(base);names=tuple(integrated_field_names(base));declared={v.name:v for v in fields(ref)}
    if not y.numel() or any(not isinstance(getattr(ref,n),torch.Tensor) or not getattr(ref,n).is_floating_point()
        or getattr(ref,n).dtype!=y.dtype or getattr(ref,n).device!=y.device for n in names):
        raise ValueError('Nonempty floating integrated fields with shared dtype/device required')
    def flat(update):
        result=[]
        for n in names:
            v=getattr(update,declared[n].metadata['update_key']);template=getattr(ref,n)
            if v.shape!=template.shape or v.dtype!=template.dtype or v.device!=template.device:raise ValueError('RHS field layout mismatch')
            result.append(v.reshape(-1))
        return torch.cat(result)
    calls=0;stages=[];endpoint=None
    def evaluate(value,time,probe=False):
        nonlocal calls,endpoint
        calls+=1;anchor=snapshot_state(base) if probe else base
        stage=anchor.initializeNewState(*args,**kwargs);current=get_reference_state(stage);offset=0
        for n in names:
            template=getattr(ref,n);size=template.numel();setattr(current,n,value[offset:offset+size].reshape(template.shape));offset+=size
        stage.t=float(time)
        update,aux=updateStep(anchor,stage,dt,f,*args,**kwargs)
        if not probe:stages.append(StageResult(aux,update));endpoint=stage
        return flat(update)
    if semilinear:
        start=y
        def A(v):
            s=unflatten_integrated(v,base)
            update,_=split_return(resolved.linear(s,dt,*args,**kwargs))
            return dt*flat(update)
    else:
        eps=ft_eps or torch.finfo(y.dtype).eps**(1/3)*max(1.,abs(t),abs(dt))
        partial=(evaluate(y,t+eps,True)-evaluate(y,t-eps,True))/(2*eps) if f_t is None else flat(split_return(f_t(snapshot_state(base),dt,*args,**kwargs))[0])
        start=torch.cat((y.new_tensor([t]),y))
        def A(v):
            jv=torch.func.jvp(lambda z:evaluate(z,t,True),(y,),(v[1:],))[1]
            return torch.cat((torch.zeros_like(v[:1]),dt*(jv+v[0]*partial)))
    n=start.numel();matrix=None;cache={};iterations=0;converged=True
    if n<=self.dense_limit:
        eye=torch.eye(n,dtype=y.dtype,device=y.device)
        matrix=torch.stack([A(v) for v in eye],dim=1)
    elif torch.is_grad_enabled() and A(torch.ones_like(start).detach()).requires_grad:
        raise ValueError('Tracked operator derivatives require n<=dense_limit; increase explicitly after measuring dense cost')
    bases={}
    def raw_action(expr,v,operator=A):
        nonlocal iterations,converged
        if isinstance(expr,(int,float)):return expr*v
        if matrix is not None and operator is A:return operator_function(expr,matrix,cache)@v
        key=(v,operator)
        if key not in bases:
            V,H,norm,count=_arnoldi(operator,v,self.phi_dim,self.phi_tol*.01);iterations+=count
            bases[key]=(V,H,norm,count)
        V,H,norm,count=bases[key]
        if count==0:return torch.zeros_like(v)
        basis=torch.stack(V[:count],dim=1);small=H[:count,:count];e=torch.zeros(count,dtype=v.dtype,device=v.device);e[0]=norm
        result=basis@(operator_function(expr,small,{})@e)
        invariant=float(H[count,count-1])<=self.phi_tol*.01*max(1.,float(torch.linalg.norm(H)))
        if not invariant:
            m=max(1,count//2);coarse=basis[:,:m]@(operator_function(expr,small[:m,:m],{})@e[:m])
            converged=converged and bool(torch.linalg.vector_norm(result-coarse)<=self.phi_tol*max(1.,float(torch.linalg.vector_norm(result))))
        return result
    def action(expr,v):
        if matrix is not None:return raw_action(expr,v)
        with torch.no_grad():value=raw_action(expr,v)
        if not v.requires_grad:return value
        def transpose(w):
            zero=torch.zeros_like(start,requires_grad=True)
            def AT(q):
                with torch.enable_grad():return torch.autograd.grad(A(zero),zero,q,retain_graph=False)[0]
            return raw_action(expr,w,AT)
        return _linear_map_reattach(v,value,transpose)
    rates=[];program=PROGRAMS[self.method]
    def combine(coefficients,*,increment=False):
        c0=coefficients[0]
        # Main result in increment coordinates avoids y + (-y) cancellation.
        if increment:
            c0=['add',c0,-1.]
        result=action(c0,start)
        for c,r in zip(coefficients[1:],rates):
            if c!=0:result=result+action(c,r)
        return result
    for call in program['calls']:
        z=combine(call['coefficients']);clock=t+dt*call['c'] if semilinear else float(z[0].detach())
        rate=dt*evaluate(z if semilinear else z[1:],clock)
        rates.append(rate if semilinear else torch.cat((rate.new_tensor([dt]),rate)))
    delta=combine(program['output'],increment=True);output=start+delta
    values=output if semilinear else output[1:]
    result_state=unflatten_integrated(values,base);result_state.t=float(t+dt)
    defect=None
    if program['error'] is not None:
        err=combine(program['error']);err=err if semilinear else err[1:]
        defect=unflatten_integrated(err,base);defect.t=result_state.t
    coefficients=program['output'];initial=action(['add',coefficients[0],-1.],start)
    increments=tuple(action(c,r) if c!=0 else torch.zeros_like(start) for c,r in zip(coefficients[1:],rates))
    if not semilinear:initial=initial[1:];increments=tuple(v[1:] for v in increments)
    quadrature=ExponentialQuadrature(initial,increments,names)
    valid=converged and bool(torch.isfinite(values).all())
    diagnostics=SolveDiagnostics(converged=valid,termination='tolerance' if valid else 'max_iterations',rhs_evaluations=calls,gmres_iterations=iterations)
    result=IntegrationResult(result_state,stages=stages,error=defect,estimate_order=self.estimate_order,quadrature=quadrature,solver_diagnostics=[diagnostics])
    if valid:finalizeSystem(result_state,base,dt,[s.aux for s in stages],[s.update for s in stages],quadrature,*args,lastStageSystem=endpoint,**kwargs)
    return result
