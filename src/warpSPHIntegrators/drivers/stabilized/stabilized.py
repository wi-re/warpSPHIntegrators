"""Numerical stepping for stabilized; configuration and coefficients are supplied separately."""



import math


from warpSPHIntegrators.state.fields import get_reference_state, state_difference, integrated_field_names


from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, updateStateEuler, reject_prior_step


from warpSPHIntegrators.schemes.coefficients.stabilized import stabilized_coefficients

def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    base=snapshot_state(state)
    if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
    initializeSystem(base,dt,*args,**kwargs)
    s=self.stage_count(dt);coeff=stabilized_coefficients(self.family,s)
    y0=base.initializeNewState(*args,**kwargs);y0.t=float(base.t)
    f0,r0=updateStep(base,y0,dt,f,*args,**kwargs)
    updates=[f0];auxes=[r0]
    previous2=y0
    previous=updateStateEuler(base.initializeNewState(*args,**kwargs),f0,dt*coeff.mtilde[1],copyState=False,**kwargs)
    previous.t=float(base.t+dt*coeff.c[1])
    for j in range(2,s+1):
        derivative,aux=updateStep(base,previous,dt,f,*args,**kwargs)
        updates.append(derivative);auxes.append(aux)
        current=base.initializeNewState(*args,**kwargs)
        ref=get_reference_state(current)
        base_ref=get_reference_state(y0)
        previous_ref=get_reference_state(previous);previous2_ref=get_reference_state(previous2)
        # Difference form preserves zero-rate components exactly and avoids
        # cancellation between large common offsets in the three states.
        combined={name:getattr(base_ref,name)
            +coeff.mu[j]*(getattr(previous_ref,name)-getattr(base_ref,name))
            +coeff.nu[j]*(getattr(previous2_ref,name)-getattr(base_ref,name))
            for name in integrated_field_names(base)}
        for name,value in combined.items():setattr(ref,name,value)
        current=updateStateEuler(current,derivative,dt*coeff.mtilde[j],copyState=False,**kwargs)
        if coeff.gtilde[j]:current=updateStateEuler(current,f0,dt*coeff.gtilde[j],copyState=False,**kwargs)
        current.t=float(base.t+dt*coeff.c[j]);previous2,previous=previous,current
    output=previous;output.t=float(base.t+dt)
    endpoint=output.initializeNewState(*args,**kwargs);endpoint.t=output.t
    derivative,aux=updateStep(base,endpoint,dt,f,*args,**kwargs)
    updates.append(derivative);auxes.append(aux)
    euler=updateStateEuler(base.initializeNewState(*args,**kwargs),f0,dt,copyState=False,**kwargs)
    error=state_difference(output,euler)
    stages=[StageResult(aux,update) for aux,update in zip(auxes,updates)]
    candidate=IntegrationResult(output,stages=stages,error=error,estimate_order=2)
    if not assess_step(candidate).successful:return candidate
    finalizeSystem(output,base,dt,auxes,updates,[*coeff.b,0.],*args,lastStageSystem=endpoint,**kwargs)
    return candidate
