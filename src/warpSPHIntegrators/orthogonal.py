"""Bound ROCK/SERK/RKMC recurrences with owned full-state increment execution."""
from dataclasses import dataclass, fields
import math
from numbers import Real
from types import SimpleNamespace

import torch

from ._orthogonal_program import (CAPS,MINIMUM,available_stages,sizing_bound,
    orthogonal_coefficients,recurrence_program)
from .catalog import SchemeMetadata
from .fields import get_reference_state, integrated_field_names, flatten_integrated, unflatten_integrated
from .specs import IntegrationResult, StageResult
from .util import initializeSystem, finalizeSystem, updateStep, reject_prior_step


@dataclass(frozen=True)
class OrthogonalRK:
    """ROCK2/4, SERK2 or RKMC2 with fixed stages or a supplied spectral bound.

    Fixed counts round up to a source-supported configuration. lambda_max is a
    bound on negative-real spectral magnitude; exceeding the cap raises rather
    than truncating stability needs. The source selector inversion is conservative
    and numerically qualified; no nonnormal/nonlinear stability certificate.
    Increment coordinates preserve zero-rate fields. Full stages/AD graphs are
    retained; no low-storage, source spectral-estimator or FSAL policy claim.
    """
    method: str='ROCK2'
    stages: int|None=None
    lambda_max: float|None=None
    max_stages: int|None=None

    def __post_init__(self):
        if self.method not in CAPS:raise ValueError(f'method must be one of {tuple(CAPS)}')
        cap=CAPS[self.method] if self.max_stages is None else self.max_stages
        if type(cap) is not int or not MINIMUM[self.method]<=cap<=CAPS[self.method]:raise ValueError('Invalid maximum stage count')
        object.__setattr__(self,'max_stages',cap)
        if self.stages is not None and self.lambda_max is not None:raise ValueError('Choose stages or lambda_max')
        if self.lambda_max is not None:
            if isinstance(self.lambda_max,bool) or not isinstance(self.lambda_max,Real) or not math.isfinite(self.lambda_max) or self.lambda_max<0:
                raise ValueError('lambda_max must be finite and nonnegative')
        else:
            target=10 if self.stages is None else self.stages
            if type(target) is not int or target<MINIMUM[self.method]:raise ValueError('Invalid fixed stage count')
            if self.method=='ROCK4' and target in (133,152):
                raise ValueError('Pinned ROCK4 stage counts 133/152 have unstable interior bands')
            actual=next((s for s in available_stages(self.method) if s>=target and s<=cap),None)
            if actual is None:raise ValueError('Requested stages exceed the available cap')
            object.__setattr__(self,'stages',actual)

    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name=f'orthogonal_{self.method}_{self.stages}_{None if self.lambda_max is None else float(self.lambda_max).hex()}_{self.max_stages}')
    @property
    def order(self):return 4 if self.method=='ROCK4' else 2
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 4 if self.method=='ROCK4' else 2
    @property
    def metadata(self):return SchemeMetadata(stage_count=None if self.stages is None else self.stages+1,
        embedded_order=3 if self.method=='ROCK4' else 1,estimate_order=self.estimate_order)

    def stages_for(self,dt):
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if self.stages is not None:return self.stages
        demand=abs(dt)*self.lambda_max
        for stages in available_stages(self.method):
            if stages>self.max_stages:break
            if sizing_bound(self.method,stages)>=demand:return stages
        raise ValueError('Spectral stability requirement exceeds max_stages; reduce dt')

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state, assess_step
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
