"""Numerical stepping for orthogonal; configuration and coefficients are supplied separately."""
from dataclasses import fields

import math


import torch

from warpSPHIntegrators.schemes.coefficients.orthogonal import orthogonal_coefficients, recurrence_program


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, flatten_integrated, unflatten_integrated

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step


def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    stages=self.stages_for(dt);coeff=orthogonal_coefficients(self.method,stages)
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    base=snapshot_state(state)
    if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
    initializeSystem(base,dt,*args,**kwargs)
    if not math.isfinite(float(base.t)):raise ValueError('finite initialized time required')
    ref=get_reference_state(base);names=tuple(integrated_field_names(base));declared={field.name:field for field in fields(ref)}
    y=flatten_integrated(base)
    if not y.numel() or any(not isinstance(getattr(ref,name),torch.Tensor) or not getattr(ref,name).is_floating_point()
            or getattr(ref,name).dtype!=y.dtype or getattr(ref,name).device!=y.device for name in names):
        raise ValueError('Nonempty floating fields with a shared dtype and device required')
    results=[];endpoint_state=None
    def evaluate(offset,*,endpoint=False):
        nonlocal endpoint_state
        index=len(results);stage=base.initializeNewState(*args,**kwargs)
        stage.t=float(base.t+dt*(1. if endpoint else coeff.c[index]))
        current=get_reference_state(stage);flat=y+offset;start=0
        for name in names:
            template=getattr(ref,name);size=template.numel()
            setattr(current,name,flat[start:start+size].reshape(template.shape));start+=size
        update,aux=updateStep(base,stage,dt,f,*args,**kwargs)
        rates=[]
        for name in names:
            rate=getattr(update,declared[name].metadata['update_key']);template=getattr(ref,name)
            if rate.shape!=template.shape or rate.dtype!=template.dtype or rate.device!=template.device:
                raise ValueError('Recurrence rate must match integrated field layout')
            rates.append(rate.reshape(-1))
        results.append(StageResult(aux,update))
        if endpoint:endpoint_state=stage
        return dt*torch.cat(rates)
    offset,error=recurrence_program(self.method,stages,coeff.parameters,evaluate,torch.zeros_like(y))
    output=unflatten_integrated(y+offset,base);output.t=float(base.t+dt)
    error_state=unflatten_integrated(error,base);error_state.t=output.t
    result=IntegrationResult(output,stages=results,error=error_state,
        estimate_order=self.estimate_order,quadrature=coeff.weights)
    if assess_step(result).successful:
        finalizeSystem(output,base,dt,[s.aux for s in results],[s.update for s in results],coeff.weights,
            *args,lastStageSystem=endpoint_state,**kwargs)
    return result
