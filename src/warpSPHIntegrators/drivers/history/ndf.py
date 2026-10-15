"""Numerical stepping for ndf; configuration and coefficients are supplied separately."""
from dataclasses import replace


import math


import torch


from warpSPHIntegrators.state.fields import get_reference_state, flatten_integrated, unflatten_integrated, state_difference, state_norm

from warpSPHIntegrators.state.history import StepHistory, HistoryEntry

from warpSPHIntegrators.solvers.krylov import JFNKSolver

from warpSPHIntegrators.state.protocol import BaseIntegrationSystem

from warpSPHIntegrators.solvers.nonlinear import record_solve_diagnostics

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, updateStateEuler, reject_prior_step


from warpSPHIntegrators.drivers.history._ndf import _record_rhs, _layout, _initial_differences, _STARTER, _STARTER_EXPONENT
from warpSPHIntegrators.schemes.coefficients.ndf import ndf_difference_transform, _HARMONIC
from warpSPHIntegrators.schemes.families.ndf import NDFQuadrature

def step(self,state,dt,f,*args,history=None,solver=None,restart_history=False,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    if type(restart_history) is not bool:raise ValueError('restart_history must be bool')
    reject_prior_step(self.name,kwargs.pop('priorStep',None))
    if history is not None and (not isinstance(history,StepHistory) or history.maxlen<self.order):
        raise ValueError('StepHistory capacity must cover the NDF target order')
    accepted=StepHistory(self.order) if history is None else snapshot_state(history)
    base=snapshot_state(state);ref=get_reference_state(base);flat=flatten_integrated(base)
    if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
    start_time=float(base.t)
    uid=getattr(ref,'uid',None)
    context=dict(method='ndf.v1',order=self.order,layout=_layout(base),uid=snapshot_state(uid),phase='startup')
    valid=not restart_history and accepted.latest is not None and isinstance(accepted.context,dict)
    old=accepted.context;latest=accepted.latest
    changed_h=False
    if valid:
        current=old.get('current');ratio=dt/latest.dt if latest.dt else -1
        valid=(old.get('method')=='ndf.v1' and old.get('order')==self.order
            and old.get('layout')==context['layout'] and (old.get('uid') is None)==(uid is None)
            and (uid is None or torch.equal(uid,old['uid']))
            and isinstance(current,torch.Tensor) and current.shape==flat.shape
            and current.dtype==flat.dtype and current.device==flat.device and torch.equal(current,flat)
            and (1-1e-12)/self.max_step_ratio<=ratio<=self.max_step_ratio*(1+1e-12)
            and math.isclose(latest.t+latest.dt,float(base.t),rel_tol=0,
                abs_tol=8*max(math.ulp(float(base.t)),math.ulp(latest.t+latest.dt))))
        constant_steps=old.get('constant_steps')
        valid=valid and type(constant_steps) is int and constant_steps>=1
        changed_h=not math.isclose(dt,latest.dt,rel_tol=1e-12,abs_tol=0)
        if valid and old.get('phase')=='warm' and changed_h and constant_steps<self.order+1:
            valid=False  # order-preserving one-step restart for rapid caller changes
        if valid and old.get('phase')=='warm':
            d=old.get('differences')
            valid=(isinstance(d,torch.Tensor) and d.shape==(self.order+2,flat.numel())
                and d.dtype==flat.dtype and d.device==flat.device and bool(torch.isfinite(d).all()))
        elif valid and old.get('phase')=='startup':
            previous=float(base.t)
            for entry in reversed(accepted.entries):
                if entry.state is None or _layout(entry.state)!=context['layout'] or (previous-entry.t)*dt<=0:
                    valid=False;break
                value=flatten_integrated(entry.state)
                if value.dtype!=flat.dtype or value.device!=flat.device:valid=False;break
                previous=entry.t
        else:valid=False
    memory=accepted if valid else StepHistory(accepted.maxlen)
    before=flat.clone();initializeSystem(base,dt,*args,**kwargs)
    if float(base.t)!=start_time:raise ValueError('NDF initialize must preserve time')
    new_ref=get_reference_state(base);new_uid=getattr(new_ref,'uid',None)
    if valid and (_layout(base)!=context['layout'] or not torch.equal(before,flatten_integrated(base))
        or (uid is None)!=(new_uid is None) or uid is not None and not torch.equal(context['uid'],new_uid)):
        raise ValueError('Warm NDF initialize must preserve integrated fields and particle IDs; restart history')
    context['layout']=_layout(base);context['uid']=snapshot_state(new_uid)
    start_state=base.initializeNewState(*args,**kwargs);start_state.t=float(base.t)
    hook=getattr(base.finalize,'__func__',None)
    if hook is not BaseIntegrationSystem.finalize and not getattr(base,'ndf_compatible',False):
        raise TypeError('Custom finalize requires ndf_compatible=True and NDFQuadrature support')
    solver=solver or JFNKSolver(tol=1e-12,newton_tol=1e-7)
    physical_rhs=f;rhs_calls=0
    def counted_rhs(stage,h,*a,**kw):
        nonlocal rhs_calls
        rhs_calls+=1
        return physical_rhs(stage,h,*a,**kw)
    f=counted_rhs
    warm=valid and old['phase']=='warm'
    if not warm:
        starter=_STARTER[self.order]
        result=starter(base,dt,f,*args,solver=solver,_initialize=False,_finalize=False,**kwargs)
        result=replace(result,history=accepted,solver_diagnostics=_record_rhs(result.solver_diagnostics,rhs_calls))
        if not assess_step(result).successful:return result
        output=result.state;error=result.error;exponent=_STARTER_EXPONENT[self.order]
        if self.order==1:
            euler=updateStateEuler(base,result.stages[0].update,dt,copyState=True,**kwargs)
            error=state_difference(output,euler);error.t=output.t
        tab=starter.dirkTableau;weights=tab.b[0] if isinstance(tab.b,tuple) else tab.b
        updates=[s.update for s in result.stages];auxes=[s.aux for s in result.stages]
        quad=NDFQuadrature(snapshot_state(base),tuple(float(w) for w in weights)+(0.,),True)
    else:
        d=old['differences']
        if dt!=latest.dt:
            transform=torch.tensor(ndf_difference_transform(self.order,dt/latest.dt),dtype=d.dtype,device=d.device)
            d=torch.cat((transform.T@d[:self.order],d[self.order:]),dim=0)
        y=flatten_integrated(base);prediction=y+d[:self.order].sum(0)
        phi=sum(_HARMONIC[i]*d[i] for i in range(self.order))
        known=unflatten_integrated(prediction-self.beta*phi,base);known.t=float(base.t+dt)
        guess=unflatten_integrated(prediction,base);guess.t=known.t
        def step(stage):
            stage.t=float(base.t+dt)
            update,_=updateStep(base,stage,dt,f,*args,**kwargs)
            out=updateStateEuler(known,update,self.beta*dt,copyState=True,**kwargs);out.t=known.t
            return out
        def norm(a,b):return state_norm(state_difference(a,b),1e-3,1e-6,reference=b)
        options=dict(kwargs.get('solver_opts',{}))
        options['preconditioner_context']={**(options.get('preconditioner_context',getattr(solver,'preconditioner_context',None)) or {}),
            'dt':dt,'stage_index':0,'stage_time':float(base.t+dt),'stage_coefficient':self.beta*dt}
        solved=solver.solve(step,guess,norm,**options)
        output=solved.y;output.t=float(base.t+dt)
        diagnostics=record_solve_diagnostics(solved)
        error=unflatten_integrated(self.error_constant*(flatten_integrated(output)-prediction),base);error.t=output.t
        result=IntegrationResult(output,error=error,history=accepted,solver_diagnostics=diagnostics)
        if not assess_step(result).successful:return result
        updates=[];auxes=[];exponent=self.estimate_order
        quad=NDFQuadrature(snapshot_state(known),(self.beta,),False)
    endpoint=output.initializeNewState(*args,**kwargs);endpoint.t=output.t
    last,last_aux=updateStep(base,endpoint,dt,f,*args,**kwargs)
    diagnostics=_record_rhs(result.solver_diagnostics,rhs_calls)
    updates.append(last);auxes.append(last_aux)
    result=IntegrationResult(output,stages=[StageResult(a,k) for a,k in zip(auxes,updates)],error=error,
        history=accepted,solver_diagnostics=diagnostics,estimate_order=exponent,quadrature=quad)
    if not assess_step(result).successful:return result
    if not torch.equal(flatten_integrated(start_state),flatten_integrated(base)):
        raise ValueError('NDF RHS/lifecycle must preserve the initial integrated anchor')
    before_finalize=flatten_integrated(output).clone()
    finalizeSystem(output,base,dt,auxes,updates,quad,*args,lastStageSystem=endpoint,**kwargs)
    end_uid=getattr(get_reference_state(output),'uid',None)
    if (float(output.t)!=float(base.t+dt) or not torch.equal(before_finalize,flatten_integrated(output))
        or (new_uid is None)!=(end_uid is None) or new_uid is not None and not torch.equal(context['uid'],end_uid)):
        raise ValueError('NDF finalize must preserve integrated values and particle IDs')
    if warm:
        correction=flatten_integrated(output)-prediction
        values=list(d.unbind(0));values[self.order+1]=correction-values[self.order];values[self.order]=correction
        for i in range(self.order-1,-1,-1):values[i]=values[i]+values[i+1]
        context.update(phase='warm',differences=torch.stack(values))
        memory=StepHistory(memory.maxlen)
    else:
        entries=[*memory.entries,HistoryEntry(float(base.t),dt,None,state=start_state)]
        if len(entries)>=self.order:
            context.update(phase='warm',differences=_initial_differences(output,entries,dt,self.order))
            memory=StepHistory(memory.maxlen)
    context['current']=flatten_integrated(output).detach().clone()
    context['constant_steps']=old['constant_steps']+1 if valid and not changed_h else 1
    entry=HistoryEntry(float(base.t),dt,None,state=None if context['phase']=='warm' else start_state)
    return replace(result,history=memory.pushed(entry,preserve_dt=True,context=context))
