"""Numerical stepping for function_map; configuration and coefficients are supplied separately."""
from warpSPHIntegrators.adapters.maps import StateMap


import math


import torch


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, flatten_integrated

from warpSPHIntegrators.state.protocol import BaseIntegrationSystem

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import initializeSystem, preprocessSystem, postprocessSystem, finalizeSystem, reject_prior_step, split_return


from warpSPHIntegrators.drivers.maps._layout import _layout, _uid_equal
from warpSPHIntegrators.schemes.families.function_map import MapTransition

def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if not isinstance(f,StateMap):raise TypeError('FunctionMap requires StateMap/tensor_map, not an ODE RHS')
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
    base=snapshot_state(state);start=float(base.t)
    if not math.isfinite(start):raise ValueError('finite initial time required')
    if not integrated_field_names(base):raise ValueError('FunctionMap needs integrated tensor fields')
    hook=getattr(base.finalize,'__func__',None)
    if hook is not BaseIntegrationSystem.finalize and not getattr(base,'map_compatible',False):
        raise TypeError('Custom finalize requires map_compatible=True and MapTransition support')
    initializeSystem(base,dt,*args,**kwargs)
    if float(base.t)!=start:raise ValueError('FunctionMap initialize must preserve time')
    anchor=flatten_integrated(base).clone();layout=_layout(base)
    uid=snapshot_state(getattr(get_reference_state(base),'uid',None))
    stage=base.initializeNewState(*args,**kwargs);stage.t=float(start+dt)
    preprocessSystem(stage,base,dt,*args,**kwargs)
    def compatible(value):
        return (float(value.t)==float(start+dt) and _layout(value)==layout
            and _uid_equal(uid,getattr(get_reference_state(value),'uid',None)))
    if not compatible(stage) or not torch.equal(anchor,flatten_integrated(stage)):
        raise ValueError('Map input preprocess must preserve integrated values, time, layout and IDs')
    output,aux=split_return(f(stage,dt,*args,**kwargs))
    if not hasattr(output,'t') or not callable(getattr(output,'initializeNewState',None)):
        raise TypeError('StateMap must return an integration system')
    output=snapshot_state(output)
    if not compatible(output):raise ValueError('StateMap must preserve endpoint time, layout, dtype/device and particle IDs')
    result=IntegrationResult(output,stages=[StageResult(aux,None)])
    if not assess_step(result).successful:return result
    mapped=flatten_integrated(output).clone()
    # Recompute copied endpoint fields without applying the map a second time.
    preprocessSystem(output,base,dt,*args,**kwargs)
    postprocessSystem(base,output,dt,aux,*args,**kwargs)
    if not torch.equal(anchor,flatten_integrated(base)):
        raise ValueError('Map lifecycle must preserve its initial integrated anchor')
    if not compatible(output) or not torch.equal(mapped,flatten_integrated(output)):
        raise ValueError('Map endpoint lifecycle must preserve mapped integrated values')
    transition=MapTransition(snapshot_state(base),float(start+dt),snapshot_state(output))
    finalizeSystem(output,base,dt,[aux],[None],transition,*args,lastStageSystem=output,**kwargs)
    if not compatible(output) or not torch.equal(mapped,flatten_integrated(output)):
        raise ValueError('Map finalize must preserve mapped integrated values, time and IDs')
    return IntegrationResult(output,stages=[StageResult(aux,None)],transition=transition)
