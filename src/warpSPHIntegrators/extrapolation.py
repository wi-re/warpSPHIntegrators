"""Fixed-target explicit Euler/midpoint extrapolation over full-state fields."""
from dataclasses import dataclass,fields
from fractions import Fraction
from functools import lru_cache
import math
from types import SimpleNamespace
import torch
from .catalog import SchemeMetadata
from .fields import get_reference_state,integrated_field_names,flatten_integrated,unflatten_integrated
from .protocol import BaseIntegrationSystem
from .specs import IntegrationResult,StageResult
from .util import initializeSystem,finalizeSystem,updateStep,reject_prior_step


@lru_cache(maxsize=32)
def extrapolation_coefficients(base,order):
    """Immutable sequence/main/comparison weights, derived in exact arithmetic.

    Euler uses Romberg subdivisions 1,2,4,...; midpoint uses even harmonic
    subdivisions 2,4,6,... and cancels even inverse-step powers. The comparison
    drops the coarsest column. At midpoint order two it is Euler instead.
    """
    if base not in ('euler','midpoint'):raise ValueError('base must be euler or midpoint')
    if type(order) is not int or (base=='euler' and not 2<=order<=8) or (base=='midpoint' and (not 2<=order<=10 or order%2)):
        raise ValueError('Euler orders 2–8 or even midpoint orders 2–10 required')
    sequence=tuple(2**i for i in range(order)) if base=='euler' else tuple(2*(i+1) for i in range(order//2))
    power=1 if base=='euler' else 2
    def weights(ns):
        nodes=[Fraction(1,n**power) for n in ns];out=[]
        for i,x in enumerate(nodes):
            value=Fraction(1)
            for j,y in enumerate(nodes):
                if i!=j:value*=-y/(x-y)
            out.append(float(value))
        return tuple(out)
    main=weights(sequence)
    comparison=(0.,)+weights(sequence[1:]) if len(sequence)>1 else (0.,)
    return sequence,main,comparison


@dataclass(frozen=True)
class ExtrapolationQuadrature:
    """Full affine weights over returned stage derivatives, including endpoint.

    Columns record the stage indices belonging to each subdivision sequence.
    The shared initial derivative can occur in several columns. Final weights
    already include all column coefficients and substep scales.
    """
    weights: tuple
    comparison_weights: tuple
    subdivisions: tuple
    columns: tuple


@dataclass(frozen=True)
class Extrapolation:
    """Fixed-target explicit extrapolation with adaptive timestep, not order.

    Main kernels match pinned AitkenNeville Euler / Deuflhard midpoint fixed
    columns. Midpoint is the unsmoothed even-subdivision source recurrence.
    Returned stages retain full tensors/graphs; this is not low-storage or
    internally threaded. Implicit/frozen-Jacobian extrapolation is separate.
    """
    base: str='midpoint'
    order: int=4
    def __post_init__(self):extrapolation_coefficients(self.base,self.order)
    @property
    def name(self):return f'{self.base.capitalize()} extrapolation (order {self.order})'
    @property
    def identifier(self):return SimpleNamespace(name=f'extrapolation_{self.base}_{self.order}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.order if self.base=='euler' or self.order==2 else self.order-1
    @property
    def metadata(self):
        ns,_,_=extrapolation_coefficients(self.base,self.order)
        return SchemeMetadata(stage_count=2+sum(n-1 for n in ns),
            embedded_order=self.estimate_order-1,estimate_order=self.estimate_order)

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state,assess_step
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
