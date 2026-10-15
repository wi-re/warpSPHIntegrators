"""Numerical stepping for variable_adams; configuration and coefficients are supplied separately."""
from dataclasses import fields

import math


import numpy as np

import torch

from warpSPHIntegrators.schemes.definitions.explicit import DormandPrince
from warpSPHIntegrators.drivers.runge_kutta import _weighted_update


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, state_difference, flatten_integrated

from warpSPHIntegrators.state.history import HistoryEntry, StepHistory

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step


from warpSPHIntegrators.schemes.coefficients.adams import adams_weights
from warpSPHIntegrators.drivers.history._adams_layout import _layout

def step(self,state,dt,rhs,*args,history=None,restart_history=False,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:
        raise ValueError('finite nonzero dt required')
    if type(restart_history) is not bool:raise ValueError('restart_history must be a bool')
    reject_prior_step(self.name,kwargs.pop('priorStep',None))
    needed=self.order-1
    history=StepHistory(needed) if history is None else snapshot_state(history)
    if not isinstance(history,StepHistory) or history.maxlen<needed:
        raise ValueError('StepHistory capacity must cover order-1 derivatives')
    base=snapshot_state(state);layout=_layout(base);ref=get_reference_state(base)
    uid=getattr(ref,'uid',None)
    context=dict(method='variable_adams.v1',layout=layout,uid=snapshot_state(uid))
    valid=not restart_history and bool(history.entries) and isinstance(history.context,dict)
    if valid:
        old=history.context
        valid=(old.get('method')==context['method'] and old.get('layout')==layout
               and (old.get('uid') is None)==(uid is None)
               and (uid is None or torch.equal(old['uid'],uid)))
        latest=history.latest
        ratio=dt/latest.dt if latest.dt else -1
        time_tolerance=8*max(math.ulp(float(base.t)),math.ulp(latest.t),math.ulp(latest.t+latest.dt))
        valid=valid and (1-1e-12)/self.max_step_ratio<=ratio<=self.max_step_ratio*(1+1e-12)
        valid=valid and math.isclose(latest.t+latest.dt,float(base.t),rel_tol=0,abs_tol=time_tolerance)
        if valid:
            previous=float(base.t)
            for entry in reversed(history.entries):
                if (previous-entry.t)*dt<=0 or entry.state is not None:valid=False;break
                previous=entry.t
                for field in fields(ref):
                    if field.name not in integrated_field_names(base):continue
                    rate=getattr(entry.update,field.metadata['update_key'],None)
                    value=getattr(ref,field.name)
                    if (not isinstance(rate,torch.Tensor) or rate.shape!=value.shape
                            or rate.dtype!=value.dtype or rate.device!=value.device):valid=False;break
    if not valid:history=StepHistory(history.maxlen,context=context)
    if len(history)<needed:
        started=DormandPrince(base,dt,rhs,*args,**kwargs)
        context=dict(method='variable_adams.v1',layout=_layout(base),
                     uid=snapshot_state(getattr(get_reference_state(base),'uid',None)))
        first=started.stages[0]
        partial=list(reversed(history.entries))
        nodes=[0.]+[(entry.t-float(base.t))/dt for entry in partial]
        compared=_weighted_update(base,[first.update]+[entry.update for entry in partial],
                                  adams_weights(nodes),dt,*args,**kwargs)
        startup_error=state_difference(started.state,compared);startup_error.t=float(base.t+dt)
        next_history=history
        if assess_step(started).successful:
            next_history=history.pushed(HistoryEntry(float(base.t),dt,first.update,first.aux),
                                        preserve_dt=True,context=context)
        return IntegrationResult(started.state,stages=started.stages,error=startup_error,
            history=next_history,solver_diagnostics=started.solver_diagnostics,estimate_order=len(nodes)+1)
    before=flatten_integrated(base).clone()
    initializeSystem(base,dt,*args,**kwargs)
    new_uid=getattr(get_reference_state(base),'uid',None)
    if (_layout(base)!=layout or not torch.equal(before,flatten_integrated(base))
            or (uid is None)!=(new_uid is None) or uid is not None and not torch.equal(context['uid'],new_uid)):
        raise ValueError('Warm Adams initialize must preserve integrated fields and particle IDs; restart history')
    start=base.initializeNewState(*args,**kwargs);start.t=float(base.t)
    current,aux=updateStep(base,start,dt,rhs,*args,**kwargs)
    past=list(reversed(history.entries))[:needed]
    updates=[current]+[entry.update for entry in past]
    auxes=[aux]+[entry.aux for entry in past]
    nodes=np.array([0.]+[(entry.t-float(base.t))/dt for entry in past])
    stages=[StageResult(aux,current)]
    if self.kind=='ab':
        weights=adams_weights(nodes)
        output=_weighted_update(base,updates,weights,dt,*args,**kwargs)
        compare=_weighted_update(base,updates[:-1],adams_weights(nodes[:-1]),dt,*args,**kwargs)
        used_updates,used_aux=updates,auxes
    else:
        prediction=_weighted_update(base,updates[:-1],adams_weights(nodes[:-1]),dt,*args,**kwargs)
        prediction.t=float(base.t+dt)
        predicted,predicted_aux=updateStep(base,prediction,dt,rhs,*args,**kwargs)
        stages.append(StageResult(predicted_aux,predicted))
        used_updates=[predicted]+updates[:-1];used_aux=[predicted_aux]+auxes[:-1]
        weights=adams_weights(np.r_[1.,nodes[:-1]])
        output=_weighted_update(base,used_updates,weights,dt,*args,**kwargs)
        compare=_weighted_update(base,[predicted]+updates,adams_weights(np.r_[1.,nodes]),dt,*args,**kwargs)
    output.t=float(base.t+dt)
    error=state_difference(output,compare);error.t=output.t
    endpoint=output.initializeNewState(*args,**kwargs);endpoint.t=output.t
    endpoint_update,endpoint_aux=updateStep(base,endpoint,dt,rhs,*args,**kwargs)
    stages.append(StageResult(endpoint_aux,endpoint_update))
    candidate=IntegrationResult(output,stages=stages,error=error,history=history,
                                estimate_order=self.estimate_order)
    if not assess_step(candidate).successful:return candidate
    finalizeSystem(output,base,dt,used_aux,used_updates,weights,*args,lastStageSystem=endpoint,**kwargs)
    next_history=history.pushed(HistoryEntry(float(base.t),dt,current,aux),preserve_dt=True,context=context)
    return IntegrationResult(output,stages=stages,error=error,history=next_history,
                             estimate_order=self.estimate_order)
