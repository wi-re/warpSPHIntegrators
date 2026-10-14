"""Explicit discrete state maps with endpoint clock and owned lifecycle."""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import torch
from .catalog import SchemeMetadata
from .fields import get_reference_state,integrated_field_names,flatten_integrated
from .protocol import BaseIntegrationSystem
from .specs import IntegrationResult,StageResult
from .util import (initializeSystem,preprocessSystem,postprocessSystem,finalizeSystem,
    reject_prior_step,split_return)


@dataclass(frozen=True)
class StateMap:
    """Declare function(owned_state_at_t_plus_h, h, ...)->mapped_state[, aux].

    Integrated inputs are the initialized old values. The returned state must
    preserve integrated layout/dtype/device, endpoint time and particle IDs.
    Numeric transitions belong here; lifecycle hooks preserve mapped values.
    """
    function: object
    def __post_init__(self):
        if not callable(self.function):raise TypeError('StateMap requires a callable')
    def __call__(self,state,dt,*args,**kwargs):return self.function(state,dt,*args,**kwargs)


def tensor_map(function,/,**parameters):
    """Adapt map(t_next,y_old,h,*args,**parameters)->y_next, not an ODE rate.

    Step options are not forwarded. Shape/dtype/device remain unchanged.
    The mapped tensor remains differentiable; lifecycle buffers are owned.
    """
    if not callable(function):raise TypeError('tensor_map requires a callable')
    def evaluate(system,dt,*args,**kwargs):
        if not getattr(system,'_tensor_adapter',False):raise TypeError('tensor_map requires TensorState')
        value=function(system.t,system.y,dt,*args,**parameters)
        if not isinstance(value,torch.Tensor):raise TypeError('Tensor map must return a torch.Tensor')
        if value.shape!=system.y.shape or value.dtype!=system.y.dtype or value.device!=system.y.device:
            raise ValueError('Tensor map must preserve shape, dtype and device')
        output=system.initializeNewState();output.state.y=value;output.t=system.t
        return output
    return StateMap(evaluate)


def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),getattr(ref,name).dtype,getattr(ref,name).device)
        for name in integrated_field_names(state))


def _uid_equal(a,b):
    return (a is None and b is None or isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
        and a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b))


@dataclass(frozen=True)
class MapTransition:
    """A discrete transition, not derivative quadrature or an error estimate."""
    input_state: object
    evaluation_time: float
    mapped_state: object


@dataclass(frozen=True)
class FunctionMap:
    """Apply one declared StateMap per fixed step at the source's t+h clock.

    h advances the clock and is available to the map. This has no continuous
    ODE order or error estimate. Source scale_by_time behavior is expressible
    as tensor_map(lambda t,y,h:y+h*rate(t,y)); no overloaded RHS flag is needed.
    """
    @property
    def name(self):return 'Function map'
    @property
    def identifier(self):return SimpleNamespace(name='function_map_v1')
    @property
    def order(self):return None
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def estimate_order(self):return None
    @property
    def metadata(self):return SchemeMetadata(stage_count=1)

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state,assess_step
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
