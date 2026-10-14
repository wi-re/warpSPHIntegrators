"""Fixed-target-order NDF with quasi-constant backward-difference history.

The warm kernel and R*U timestep transform match the pinned QNDF source. Startup
uses order-preserving L-stable DIRK, not the source's adaptive-order warmup.
Upstream order selection, dense interpolation and mass matrices are separate.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
import math
from numbers import Real
from types import SimpleNamespace

import numpy as np
import torch

from .catalog import SchemeMetadata
from .dirk import TRBDF2,ESDIRK324L2SA,KenCarp4,KenCarp5
from .fields import (get_reference_state,integrated_field_names,flatten_integrated,
    unflatten_integrated,state_difference,state_norm)
from .history import StepHistory,HistoryEntry
from .jfnk import JFNKSolver
from .protocol import BaseIntegrationSystem
from .solvers import record_solve_diagnostics
from .specs import IntegrationResult,StageResult
from .util import initializeSystem,finalizeSystem,updateStep,updateStateEuler,reject_prior_step


_KAPPA=(-37/200,-1/9,-823/10000,-83/2000,0.)
_HARMONIC=tuple(sum(1/j for j in range(1,k+1)) for k in range(1,7))
_STARTER={1:TRBDF2,2:TRBDF2,3:ESDIRK324L2SA,4:KenCarp4,5:KenCarp5}
_STARTER_EXPONENT={1:2,2:3,3:3,4:4,5:5}


def _record_rhs(diagnostics,calls):
    items=list(diagnostics) if isinstance(diagnostics,(tuple,list)) else [diagnostics]
    known=sum(d.rhs_evaluations+d.gmres_iterations for d in items if d is not None)
    for i,d in enumerate(items):
        if d is not None:
            items[i]=replace(d,rhs_evaluations=d.rhs_evaluations+max(0,calls-known));break
    return items if isinstance(diagnostics,(tuple,list)) else items[0]


def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),str(getattr(ref,name).dtype))
        for name in integrated_field_names(state))


@lru_cache(maxsize=128)
def ndf_difference_transform(order,ratio):
    """Immutable R(ratio)*U from the pinned quasi-constant history resampling.

    New difference j = sum_i old difference i * transform[i,j].
    This transforms a polynomial history; it is not variable-order selection.
    """
    if type(order) is not int or not 1<=order<=5:raise ValueError('NDF order must be from 1 through 5')
    if isinstance(ratio,bool) or not isinstance(ratio,Real) or not math.isfinite(ratio) or ratio<=0:
        raise ValueError('positive finite timestep ratio required')
    def matrix(scale):
        out=np.zeros((order,order))
        for r in range(1,order+1):
            out[0,r-1]=-r*scale
            for j in range(2,order+1):out[j-1,r-1]=out[j-2,r-1]*((j-1)-r*scale)/j
        return out
    return tuple(tuple(row) for row in matrix(ratio)@matrix(1.))


def _initial_differences(current,entries,h,order):
    """Interpolate accepted startup values onto the current uniform virtual grid."""
    values=[flatten_integrated(current)]+[flatten_integrated(entry.state) for entry in reversed(entries)]
    nodes=[0.]+[(entry.t-float(current.t))/h for entry in reversed(entries)]
    virtual=[values[0]]
    for j in range(1,order+1):
        value=values[0]
        for i,node in enumerate(nodes):
            weight=1.
            for k,other in enumerate(nodes):
                if k!=i:weight*=(-j-other)/(node-other)
            value=value+weight*(values[i]-values[0])
        virtual.append(value)
    differences=[]
    for _ in range(order):
        virtual=[a-b for a,b in zip(virtual,virtual[1:])];differences.append(virtual[0])
    return torch.stack([*differences,torch.zeros_like(values[0]),torch.zeros_like(values[0])])


@dataclass(frozen=True)
class NDFQuadrature:
    """Full-state affine step: y_new = known_state + h*sum(weights*f_i).

    Bootstrap uses ordinary DIRK weights, the initialized starting state and a
    zero-weight endpoint refresh. A warm step uses its known difference-history
    combination and one endpoint derivative with weight beta.
    """
    known_state: object
    weights: tuple
    bootstrap: bool


@dataclass(frozen=True)
class NDF:
    """Bind order 1–5 with owned variable-h differences and local defect control.

    Thread result.history on direct calls; solve owns trial rollback and portable
    TensorState continuation. Ratios outside the declared bound restart before
    use. Changed state values/layout/UID/time invalidate memory; changed RHS
    parameters require restart_history=True. Warm initialize and all finalize
    hooks preserve integrated values/IDs. Custom finalize explicitly opts into
    ndf_compatible=True and NDFQuadrature. No adaptive order, dense output or DAE.
    """
    order: int=3
    max_step_ratio: float=2.

    def __post_init__(self):
        if type(self.order) is not int or not 1<=self.order<=5:raise ValueError('NDF order must be from 1 through 5')
        if (isinstance(self.max_step_ratio,bool) or not isinstance(self.max_step_ratio,Real)
            or not math.isfinite(self.max_step_ratio) or not 1<=self.max_step_ratio<=2):
            raise ValueError('max_step_ratio must lie between one and two')
    @property
    def name(self):return f'NDF {self.order} (fixed target order)'
    @property
    def identifier(self):return SimpleNamespace(name=f'ndf_{self.order}_{float(self.max_step_ratio).hex()}')
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.order+1
    @property
    def metadata(self):
        return SchemeMetadata(estimate_order=self.estimate_order,history_length=self.order,
            history_requires_constant_dt=False)
    @property
    def kappa(self):return _KAPPA[self.order-1]
    @property
    def beta(self):return 1/((1-self.kappa)*_HARMONIC[self.order-1])
    @property
    def error_constant(self):return self.kappa*_HARMONIC[self.order-1]+1/(self.order+1)

    def next_step_size(self,result,h,proposal):
        """Hold accepted h through a quasi-constant block; reject trials may shrink.

        A deadband avoids repeated small resamplings. This fixed-order policy is
        separate from the upstream variable-order QNDF controller.
        """
        if result.history.context['constant_steps']<self.order+1:return h
        if 1.<=proposal/h<=1.5:return h
        return proposal

    def __call__(self,state,dt,f,*args,history=None,solver=None,restart_history=False,**kwargs):
        from .run import snapshot_state,assess_step
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
