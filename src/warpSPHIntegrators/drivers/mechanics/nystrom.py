"""Numerical stepping for nystrom; configuration and coefficients are supplied separately."""
from dataclasses import fields


import math


import torch


from warpSPHIntegrators.state.fields import find_tagged_field, get_reference_state, integrated_field_names

from warpSPHIntegrators.state.protocol import BaseIntegrationSystem

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step


from warpSPHIntegrators.schemes.families.nystrom import NystromWeights
from warpSPHIntegrators.drivers.mechanics._nystrom import _sum

def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    base=snapshot_state(state)
    if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
    hook=getattr(base.finalize,'__func__',None)
    if not kwargs.get('_composition_substep',False) and hook is not BaseIntegrationSystem.finalize and not getattr(base,'nystrom_compatible',False):
        raise TypeError('Custom finalize requires nystrom_compatible=True and NystromWeights support')
    initializeSystem(base,dt,*args,**kwargs)
    ref=get_reference_state(base)
    try:
        position=find_tagged_field(ref,tag='position');velocity=find_tagged_field(ref,tag='velocity')
    except (LookupError,TypeError) as error:
        raise ValueError('Nyström requires position/velocity-tagged fields') from error
    q0,v0=getattr(ref,position.name),getattr(ref,velocity.name)
    if (not isinstance(q0,torch.Tensor) or not isinstance(v0,torch.Tensor) or not q0.is_floating_point()
            or q0.shape!=v0.shape or q0.dtype!=v0.dtype or q0.device!=v0.device):
        raise ValueError('Matching floating position and velocity tensors required')
    tab=self.tableau;vi=tab.abar is None
    names=integrated_field_names(base)
    extra=[field for field in fields(ref) if field.name in names and field.name not in (position.name,velocity.name)]
    if vi and self.structure=='probe':
        from warpSPHIntegrators.wrappers.splitting import classify_separable
        def prepared_rhs(current,h,*a,**kw):return updateStep(base,current,h,f,*a,**kw)
        verdict=classify_separable(prepared_rhs,base,dt=dt,atol=self.structure_atol,rhs_args=args,rhs_kwargs=kwargs)
        if not verdict.separable and 'nonfinite' not in verdict.reason:
            raise ValueError('Velocity-independent Nyström requires independent acceleration and passive extra fields: '+verdict.reason)
    updates=[];auxes=[]
    def evaluate(stage):
        update,aux=updateStep(base,stage,dt,f,*args,**kwargs)
        current=get_reference_state(stage)
        rate=getattr(update,position.metadata['update_key'])
        rates=[rate,getattr(update,velocity.metadata['update_key'])]
        rates.extend(getattr(update,field.metadata['update_key']) for field in extra)
        if any(not torch.isfinite(value).all() for value in rates) or not torch.isfinite(getattr(current,velocity.name)).all():
            return update,aux  # numerical failure is assessed, not a structural exception
        if not torch.allclose(rate,getattr(current,velocity.name),atol=self.structure_atol,rtol=0):
            raise ValueError("Nyström requires q'=v at every stage")
        if vi:
            for field in extra:
                derivative=getattr(update,field.metadata['update_key'])
                if not torch.allclose(derivative,torch.zeros_like(derivative),atol=self.structure_atol,rtol=0):
                    raise ValueError('Velocity-independent Nyström requires passive extra fields; use a velocity-dependent tableau')
        return update,aux
    for i,c in enumerate(tab.c):
        stage=base.initializeNewState(*args,**kwargs);stage.t=float(base.t+c*dt)
        current=get_reference_state(stage)
        setattr(current,position.name,_sum(q0+c*dt*v0,updates,tab.a[i],velocity.metadata['update_key'],dt*dt))
        if not vi:
            setattr(current,velocity.name,_sum(v0,updates,tab.abar[i],velocity.metadata['update_key'],dt))
            for field in extra:
                setattr(current,field.name,_sum(getattr(ref,field.name),updates,tab.abar[i],field.metadata['update_key'],dt))
        update,aux=evaluate(stage);updates.append(update);auxes.append(aux)
    output=base.initializeNewState(*args,**kwargs);output.t=float(base.t+dt)
    current=get_reference_state(output)
    setattr(current,position.name,_sum(q0+dt*v0,updates,tab.b,velocity.metadata['update_key'],dt*dt))
    setattr(current,velocity.name,_sum(v0,updates,tab.bp,velocity.metadata['update_key'],dt))
    for field in extra:
        value=getattr(ref,field.name)
        setattr(current,field.name,value.clone() if vi else _sum(value,updates,tab.bp,field.metadata['update_key'],dt))
    error=None
    if self.has_error_estimate:
        error=base.initializeNewState(*args,**kwargs);error.t=output.t;error_ref=get_reference_state(error)
        setattr(error_ref,position.name,_sum(torch.zeros_like(q0),updates,tab.btilde,velocity.metadata['update_key'],dt*dt))
        setattr(error_ref,velocity.name,_sum(torch.zeros_like(v0),updates,tab.bptilde,velocity.metadata['update_key'],dt))
        for field in extra:
            zero=torch.zeros_like(getattr(ref,field.name))
            setattr(error_ref,field.name,zero if vi else _sum(zero,updates,tab.bptilde,field.metadata['update_key'],dt))
    endpoint=output.initializeNewState(*args,**kwargs);endpoint.t=output.t
    update,aux=evaluate(endpoint);updates.append(update);auxes.append(aux)
    quadrature=NystromWeights((*tab.b,0.),(*tab.bp,0.),tuple(0. for _ in updates) if vi else (*tab.bp,0.))
    result=IntegrationResult(output,stages=[StageResult(a,k) for a,k in zip(auxes,updates)],
        error=error,estimate_order=self.estimate_order,quadrature=quadrature)
    if assess_step(result).successful:
        finalizeSystem(output,base,dt,auxes,updates,quadrature,*args,lastStageSystem=endpoint,**kwargs)
    return result
