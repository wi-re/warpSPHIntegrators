"""Numerical stepping for extrapolation; configuration and coefficients are supplied separately."""
from dataclasses import fields


import math


import torch


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, flatten_integrated, unflatten_integrated

from warpSPHIntegrators.state.protocol import BaseIntegrationSystem

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step


from warpSPHIntegrators.schemes.coefficients.extrapolation import extrapolation_coefficients
from warpSPHIntegrators.schemes.families.extrapolation import ExtrapolationQuadrature

def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    base=snapshot_state(state);start_time=float(base.t)
    if not math.isfinite(start_time):raise ValueError('finite initial time required')
    hook=getattr(base.finalize,'__func__',None)
    if not kwargs.get('_composition_substep',False) and hook is not BaseIntegrationSystem.finalize and not getattr(base,'extrapolation_compatible',False):
        raise TypeError('Custom finalize requires extrapolation_compatible=True and ExtrapolationQuadrature support')
    initializeSystem(base,dt,*args,**kwargs)
    if float(base.t)!=start_time:raise ValueError('Extrapolation initialize must preserve time')
    anchor=flatten_integrated(base).clone();ref=get_reference_state(base)
    names=integrated_field_names(base);specs=[field for field in fields(ref) if field.name in names]
    uid=snapshot_state(getattr(ref,'uid',None))
    ns,main,companion=extrapolation_coefficients(self.base,self.order)
    updates=[];auxes=[];rates=[];columns=[]
    def evaluate(value,time):
        stage=unflatten_integrated(value,base);stage.t=float(time)
        update,aux=updateStep(base,stage,dt,f,*args,**kwargs)
        rate=torch.cat([getattr(update,field.metadata['update_key']).reshape(-1) for field in specs])
        rates.append(rate);updates.append(update);auxes.append(aux)
        return len(rates)-1,stage
    first,_=evaluate(anchor,start_time)
    weights=[0.];comparison=[0.]
    for n,a,b in zip(ns,main,companion):
        indices=[first];h=dt/n
        previous=anchor;current=anchor+h*rates[first]
        for j in range(1,n):
            index,_=evaluate(current,start_time+j*h);indices.append(index)
            weights.append(0.);comparison.append(0.)
            following=current+h*rates[index] if self.base=='euler' else previous+2*h*rates[index]
            previous,current=current,following
        columns.append(tuple(indices))
        for j,index in enumerate(indices):
            local=1/n if self.base=='euler' else 2/n if j%2 else 0.
            weights[index]+=a*local;comparison[index]+=b*local
    if self.base=='midpoint' and self.order==2:comparison[first]=1.
    def increment(ws):
        value=torch.zeros_like(anchor)
        for w,rate in zip(ws,rates):
            if w:value=value+dt*w*rate
        return value
    output=unflatten_integrated(anchor+increment(weights),base);output.t=float(start_time+dt)
    error=unflatten_integrated(increment(tuple(a-b for a,b in zip(weights,comparison))),base);error.t=output.t
    result=IntegrationResult(output,stages=[StageResult(a,k) for a,k in zip(auxes,updates)],error=error,
        estimate_order=self.estimate_order)
    if not assess_step(result).successful:return result
    _,endpoint=evaluate(flatten_integrated(output),output.t);weights.append(0.);comparison.append(0.)
    quadrature=ExtrapolationQuadrature(tuple(weights),tuple(comparison),ns,tuple(columns))
    result=IntegrationResult(output,stages=[StageResult(a,k) for a,k in zip(auxes,updates)],error=error,
        estimate_order=self.estimate_order,quadrature=quadrature)
    if not assess_step(result).successful:return result
    if not torch.equal(anchor,flatten_integrated(base)):
        raise ValueError('Extrapolation RHS/lifecycle must preserve the initial integrated anchor')
    before=flatten_integrated(output).clone()
    finalizeSystem(output,base,dt,auxes,updates,quadrature,*args,lastStageSystem=endpoint,**kwargs)
    end_uid=getattr(get_reference_state(output),'uid',None)
    if (float(output.t)!=float(start_time+dt) or not torch.equal(before,flatten_integrated(output))
        or (uid is None)!=(end_uid is None) or uid is not None and not torch.equal(uid,end_uid)):
        raise ValueError('Extrapolation finalize must preserve integrated values and particle IDs')
    return result
