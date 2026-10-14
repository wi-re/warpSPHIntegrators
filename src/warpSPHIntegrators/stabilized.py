"""Stage-bound Legendre/Gegenbauer stabilized explicit RK backends.

Recurrence stage times match their first-order consistency relation. The Euler
comparison is unscaled (the source controller divides its norm by stage count).
Every step refreshes the physical endpoint, costing s+1 RHS evaluations. Stored
derivatives and autograd graphs mean this is not a low-storage execution path.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real
from types import SimpleNamespace

from .catalog import SchemeMetadata
from .fields import get_reference_state, state_difference
from .rkc import _reference_lincomb, _rkl2_coeffs
from .specs import IntegrationResult, StageResult
from .util import initializeSystem, finalizeSystem, updateStep, updateStateEuler, reject_prior_step


_MINIMUM={'rkl1':3,'rkl2':3,'rkg1':2,'rkg2':3}


def stabilized_interval(family,stages):
    """Conservative negative-real sizing interval used by the pinned source.

    Gegenbauer sizing uses half its full polynomial interval. This is a supplied
    spectral-radius bound, not an internal eigenvalue or pseudospectrum estimate.
    """
    if family not in _MINIMUM:raise ValueError('unknown stabilized family')
    if type(stages) is not int or stages<_MINIMUM[family]:raise ValueError('invalid stage count')
    if family=='rkl1':return float(stages*(stages+1))
    if family=='rkl2':return float(stages*stages+stages-2)/2
    if family=='rkg1':return float(stages*(stages+3))/4
    return float((stages+4)*(stages-1))/6


@dataclass(frozen=True)
class StabilizedCoefficients:
    mu: tuple
    nu: tuple
    mtilde: tuple
    gtilde: tuple
    c: tuple
    b: tuple


@lru_cache(maxsize=64)
def stabilized_coefficients(family,stages):
    """Immutable three-term recurrence, internally consistent times and final weights."""
    stabilized_interval(family,stages)
    s=stages
    if family=='rkl2':
        _,mu,nu,mtilde,gtilde=_rkl2_coeffs(s)
    else:
        mu=[0.]*(s+1);nu=[0.]*(s+1);mtilde=[0.]*(s+1);gtilde=[0.]*(s+1)
        if family=='rkl1':
            w1=2/(s*(s+1));mtilde[1]=w1
            for j in range(2,s+1):
                mu[j]=(2*j-1)/j;nu[j]=-(j-1)/j;mtilde[j]=mu[j]*w1
        else:
            if family=='rkg1':
                w1=4/(s*(s+3));b=[2/((j+1)*(j+2)) for j in range(s+1)]
            else:
                w1=6/((s+4)*(s-1));b=[1.,1/3]+[
                    4*(j-1)*(j+4)/(3*j*(j+1)*(j+2)*(j+3)) for j in range(2,s+1)]
            mtilde[1]=w1
            for j in range(2,s+1):
                mu[j]=(2*j+1)/j*b[j]/b[j-1]
                nu[j]=-(j+1)/j*b[j]/b[j-2]
                mtilde[j]=mu[j]*w1
                if family=='rkg2':gtilde[j]=-mtilde[j]*(1-j*(j+1)/2*b[j-1])
    c=[0.,mtilde[1]]+[0.]*(s-1)
    for j in range(2,s+1):c[j]=mu[j]*c[j-1]+nu[j]*c[j-2]+mtilde[j]+gtilde[j]
    # Reverse propagation through the recurrence obtains all final derivative
    # weights in O(s), without expanding a dense Butcher tableau.
    adjoint=[0.]*(s+1);adjoint[s]=1.;weights=[0.]*s
    for j in range(s,1,-1):
        weights[j-1]+=adjoint[j]*mtilde[j];weights[0]+=adjoint[j]*gtilde[j]
        adjoint[j-1]+=adjoint[j]*mu[j];adjoint[j-2]+=adjoint[j]*nu[j]
    weights[0]+=adjoint[1]*mtilde[1]
    return StabilizedCoefficients(*(tuple(x) for x in (mu,nu,mtilde,gtilde,c,weights)))


@dataclass(frozen=True)
class StabilizedRK:
    """Bind RKL1/2 or RKG1/2 and fixed stages or a supplied spectral bound.

    Default: five stages. ``lambda_max=...`` selects stages per attempted h;
    exceeding max_stages raises instead of silently truncating the stability
    requirement. Fixed Legendre counts must be odd, matching source selection.
    No source spectral estimator, FSAL, dense output or low-storage claim.
    """
    family: str='rkg2'
    stages: int|None=None
    lambda_max: float|None=None
    max_stages: int=1000

    def __post_init__(self):
        if self.family not in _MINIMUM:raise ValueError('family must be rkl1, rkl2, rkg1 or rkg2')
        if type(self.max_stages) is not int or self.max_stages<_MINIMUM[self.family]:
            raise ValueError('max_stages must be an integer above the method minimum')
        if self.stages is None and self.lambda_max is None:object.__setattr__(self,'stages',5)
        if self.stages is not None:
            stabilized_interval(self.family,self.stages)
            if self.stages>self.max_stages:raise ValueError('stages exceeds max_stages')
            if self.family.startswith('rkl') and self.stages%2==0:raise ValueError('Legendre stages must be odd')
            if self.lambda_max is not None:raise ValueError('supply stages or lambda_max, not both')
        else:
            if isinstance(self.lambda_max,bool) or not isinstance(self.lambda_max,Real) or not math.isfinite(self.lambda_max) or self.lambda_max<0:
                raise ValueError('lambda_max must be finite and nonnegative')
            object.__setattr__(self,'lambda_max',float(self.lambda_max))

    def stage_count(self,dt):
        if isinstance(dt,bool) or not isinstance(dt,Real) or not math.isfinite(dt):
            raise ValueError('finite real dt required')
        if self.stages is not None:return self.stages
        magnitude=abs(dt)*self.lambda_max
        s=_MINIMUM[self.family]
        while stabilized_interval(self.family,s)<magnitude:
            s+=2 if self.family.startswith('rkl') else 1
            if s>self.max_stages:raise ValueError('spectral bound requires more than max_stages')
        return s

    @property
    def order(self):return int(self.family[-1])
    @property
    def name(self):return f'{self.family.upper()} (s={self.stages})' if self.stages is not None else f'{self.family.upper()} (lambda_max={self.lambda_max})'
    @property
    def identifier(self):return SimpleNamespace(name=f'stabilized_{self.family}_{self.stages}_{self.lambda_max}_{self.max_stages}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 2
    @property
    def metadata(self):
        return SchemeMetadata(stage_count=self.stages+1 if self.stages is not None else None,
            embedded_order=1,estimate_order=2)

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state, assess_step
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
            combined=_reference_lincomb(get_reference_state(y0),[
                (get_reference_state(y0),1-coeff.mu[j]-coeff.nu[j]),
                (get_reference_state(previous),coeff.mu[j]),(get_reference_state(previous2),coeff.nu[j])])
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
