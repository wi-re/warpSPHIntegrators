"""Explicit generalized Runge–Kutta–Nyström for tagged mechanics systems.

Velocity-dependent tableaus lift extra first-order fields through their velocity
stage/final weights. Velocity-independent tableaus require passive extra fields.
Position and velocity quadratures are distinct and exposed to finalization.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
import math
from numbers import Real
from types import SimpleNamespace

import torch

from ._nystrom_data import TABLEAUS
from .catalog import SchemeMetadata
from .fields import find_tagged_field, get_reference_state, integrated_field_names
from .protocol import BaseIntegrationSystem
from .specs import IntegrationResult, StageResult
from .util import initializeSystem, finalizeSystem, updateStep, reject_prior_step


_ORDER={'DPRKN4':4,'DPRKN5':5,'DPRKN6':6,'DPRKN6FM':6,'DPRKN8':8,'DPRKN12':12,
    'ERKN4':4,'ERKN5':5,'ERKN7':7,'Nystrom4':4,'Nystrom4VelocityIndependent':4,
    'Nystrom5VelocityIndependent':5,'FineRKN4':4,'FineRKN5':5}
_ESTIMATE={'DPRKN4':4,'DPRKN5':5,'DPRKN6':5,'DPRKN6FM':5,'DPRKN8':7,'DPRKN12':11,
    'ERKN4':4,'ERKN5':5,'ERKN7':6,'FineRKN4':4,'FineRKN5':5}


@dataclass(frozen=True)
class NystromTableau:
    a: tuple
    abar: tuple|None
    b: tuple
    bp: tuple
    btilde: tuple
    bptilde: tuple
    c: tuple
    order: int
    estimate_order: int|None
    position_only_estimate: bool


@lru_cache(maxsize=14)
def nystrom_tableau(method):
    """Immutable pinned RKN coefficients; c includes the initial zero node."""
    if method=='RKN4':
        raise ValueError('Pinned RKN4 is only order two with velocity-dependent forces; use Nystrom4 or a velocity-independent method')
    if method not in TABLEAUS:raise ValueError(f'Unknown Nyström method; choose from {tuple(_ORDER)}')
    data=TABLEAUS[method]
    return NystromTableau(tuple(tuple(row) for row in data['a']),
        tuple(tuple(row) for row in data['abar']) if data['velocity_dependent'] else None,
        *(tuple(data[key]) for key in ('b','bp','btilde','bptilde')),
        (0.,*data['c']),_ORDER[method],_ESTIMATE.get(method),data['pos_only_error'])


@dataclass(frozen=True)
class NystromWeights:
    """Component quadratures paired with IntegrationResult.stages.

    Position = q0 + h*position_initial_velocity*v0 + h²*sum(position*a).
    Velocity/quantity increments use h*sum(velocity*a) / h*sum(quantity*g).
    The physical endpoint refresh has zero weight in each tuple.
    """
    position: tuple
    velocity: tuple
    quantity: tuple
    position_initial_velocity: float=1.


def _sum(value,updates,weights,key,scale):
    for update,weight in zip(updates,weights):
        if weight:value=value+(scale*weight)*getattr(update,key)
    return value


@dataclass(frozen=True)
class Nystrom:
    """Bind an explicit RKN method for q'=v and explicit constant-mass acceleration.

    Default VI structure='probe' checks local acceleration velocity-independence
    using AD. structure='declared' is an explicit caller assertion for that
    property. Both policies check q'=v and passive extra fields at actual stages.
    VD methods support coupled extra first-order fields via a rigorous velocity
    lifting. These factories are not claims of symplecticity or phase exactness.

    Custom finalize requires nystrom_compatible=True and consumes NystromWeights;
    grouped physical compositions suppress inner finalize and retain quadrature
    on each result. No FSAL, dense output, constraint or particle-resorting policy.
    """
    method: str='DPRKN6'
    structure: str='probe'
    structure_atol: float=1e-10

    def __post_init__(self):
        nystrom_tableau(self.method)
        if self.structure not in ('probe','declared'):raise ValueError('structure must be probe or declared')
        if isinstance(self.structure_atol,bool) or not isinstance(self.structure_atol,Real) or not math.isfinite(self.structure_atol) or self.structure_atol<0:
            raise ValueError('structure_atol must be finite and nonnegative')

    @property
    def tableau(self):return nystrom_tableau(self.method)
    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name=f'nystrom_{self.method}_{self.structure}_{float(self.structure_atol).hex()}')
    @property
    def order(self):return self.tableau.order
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return self.tableau.estimate_order is not None
    @property
    def estimate_order(self):return self.tableau.estimate_order
    @property
    def metadata(self):
        q=self.estimate_order
        return SchemeMetadata(stage_count=len(self.tableau.b)+1,
            embedded_order=q-1 if q is not None else None,estimate_order=q,
            state_kind='second_order',requires_separable=self.tableau.abar is None)

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state, assess_step
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
            from .splitting import classify_separable
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
