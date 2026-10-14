"""Improved explicit Nyström history with corrected, physically timed startup.

The warm recurrences follow the pinned IRKN3/4 mutable kernels. Startup uses
Nyström's fourth-order velocity-independent method, then seeds the old physical
acceleration and internal nodes explicitly; it does not copy the source seed bug.
"""
from dataclasses import dataclass, fields, replace
import math
from types import SimpleNamespace

import torch

from .catalog import SchemeMetadata
from .fields import (find_tagged_field, get_reference_state, integrated_field_names,
                     flatten_integrated)
from .history import HistoryEntry, StepHistory
from .nystrom import Nystrom
from .protocol import BaseIntegrationSystem
from .specs import IntegrationResult, StageResult
from .util import initializeSystem, finalizeSystem, updateStep, reject_prior_step
from .variable_adams import _layout


@dataclass(frozen=True)
class ImprovedNystromWeights:
    """Quadrature paired with stages, including the explicitly retained old rates.

    q1=q0+h*(initial_velocity*v0+previous_velocity_weight*previous_velocity)
       +h²*sum(position[i]*stages[i].update.acceleration).
    v1=v0+h*sum(velocity[i]*stages[i].update.acceleration).
    Extra integrated fields are passive. Endpoint refresh has zero weight.
    """
    position: tuple
    velocity: tuple
    quantity: tuple
    initial_velocity: float
    previous_velocity_weight: float
    previous_velocity: object = None


@dataclass(frozen=True)
class ImprovedNystrom:
    """Fixed-step IRKN3/4 for q'=v, velocity-independent force and passive extras.

    Corrected startup is deliberately distinct from the faulty pinned source
    startup. Feed accepted result.history to direct calls or let solve own it.
    State/time/layout/UID/step changes restart; changed RHS parameters require
    restart_history=True. Custom finalize must opt into irkn_compatible and
    preserve integrated fields. Supports in-memory mechanical checkpoints.
    """
    order: int = 4
    structure: str = 'probe'
    structure_atol: float = 1e-10

    def __post_init__(self):
        if type(self.order) is not int or self.order not in (3,4):
            raise ValueError('Improved Nyström order must be 3 or 4')
        Nystrom('Nystrom4VelocityIndependent',self.structure,self.structure_atol)

    @property
    def name(self):return f'Improved Nyström {self.order} (corrected startup)'
    @property
    def identifier(self):return SimpleNamespace(name=f'improved_nystrom_{self.order}_{self.structure}_{float(self.structure_atol).hex()}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def metadata(self):return SchemeMetadata(state_kind='second_order',requires_separable=True,history_length=1)

    def __call__(self,state,dt,f,*args,history=None,restart_history=False,**kwargs):
        from .run import snapshot_state, assess_step
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if type(restart_history) is not bool:raise ValueError('restart_history must be bool')
        reject_prior_step(self.name,kwargs.pop('priorStep',None))
        if kwargs.get('_composition_substep',False):raise ValueError('Improved Nyström history cannot be used as a physical subflow')
        if history is not None and not isinstance(history,StepHistory):raise TypeError('StepHistory required')
        accepted=snapshot_state(history) if history is not None else StepHistory(1)
        base=snapshot_state(state)
        if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
        hook=getattr(base.finalize,'__func__',None)
        if hook is not BaseIntegrationSystem.finalize and not getattr(base,'irkn_compatible',False):
            raise TypeError('Custom finalize requires irkn_compatible=True and ImprovedNystromWeights support')
        initializeSystem(base,dt,*args,**kwargs)
        ref=get_reference_state(base)
        try:
            position=find_tagged_field(ref,tag='position');velocity=find_tagged_field(ref,tag='velocity')
        except (LookupError,TypeError) as error:raise ValueError('Improved Nyström requires position/velocity tags') from error
        q0,v0=getattr(ref,position.name),getattr(ref,velocity.name)
        if (not isinstance(q0,torch.Tensor) or not isinstance(v0,torch.Tensor) or not q0.is_floating_point()
                or q0.shape!=v0.shape or q0.dtype!=v0.dtype or q0.device!=v0.device):
            raise ValueError('Matching floating position and velocity tensors required')
        names=integrated_field_names(base)
        extra=[field for field in fields(ref) if field.name in names and field.name not in (position.name,velocity.name)]
        if self.structure=='probe':
            from .splitting import classify_separable
            def prepared_rhs(current,h,*a,**kw):return updateStep(base,current,h,f,*a,**kw)
            verdict=classify_separable(prepared_rhs,base,dt=dt,atol=self.structure_atol,rhs_args=args,rhs_kwargs=kwargs)
            if not verdict.separable and 'nonfinite' not in verdict.reason:
                raise ValueError('Improved Nyström requires independent acceleration and passive extra fields: '+verdict.reason)
        def evaluate(q,v,t):
            stage=base.initializeNewState(*args,**kwargs);stage.t=float(t)
            current=get_reference_state(stage)
            setattr(current,position.name,q);setattr(current,velocity.name,v)
            update,aux=updateStep(base,stage,dt,f,*args,**kwargs)
            drift=getattr(update,position.metadata['update_key'])
            rates=[drift,getattr(update,velocity.metadata['update_key'])]
            rates.extend(getattr(update,field.metadata['update_key']) for field in extra)
            if all(torch.isfinite(rate).all() for rate in rates):
                if not torch.allclose(drift,getattr(current,velocity.name),rtol=0,atol=self.structure_atol):
                    raise ValueError("Improved Nyström requires q'=v at every stage")
                if any(not torch.allclose(rate,torch.zeros_like(rate),rtol=0,atol=self.structure_atol) for rate in rates[2:]):
                    raise ValueError('Improved Nyström requires passive extra fields')
            return StageResult(aux,update),stage
        def acc(stage):return getattr(stage.update,velocity.metadata['update_key'])
        flat=flatten_integrated(base);uid=getattr(ref,'uid',None);c=accepted.context
        warm=(not restart_history and isinstance(c,dict) and c.get('method')==self.identifier.name
            and c.get('t')==float(base.t) and c.get('layout')==_layout(base)
            and isinstance(c.get('current'),torch.Tensor) and c['current'].shape==flat.shape
            and c['current'].dtype==flat.dtype and c['current'].device==flat.device and torch.equal(c['current'],flat)
            and (uid is None)==(c.get('uid') is None) and (uid is None or torch.equal(uid,c['uid']))
            and accepted.latest is not None and math.isclose(accepted.latest.dt,dt,rel_tol=1e-12,abs_tol=0))
        t=float(base.t);h=dt
        c1,a21=(.5,1/8) if self.order==3 else (.25,1/32)
        if warm:
            initial=c['endpoint'];old=c['initial'];previous=c['velocity'];a0=acc(initial)
            second,_=evaluate(q0+c1*h*v0+h*h*a21*(acc(old) if self.order==3 else a0),v0,t+c1*h)
            stages=[initial,old,second,c['second']]
            pw=[0.,0.,5/12,-5/12] if self.order==3 else [0.,0.,7/24,-7/24]
            vw=[2/3,1/3,5/6,-5/6] if self.order==3 else [19/18,-1/18,-1/6,1/6]
            third=None
            if self.order==4:
                third,_=evaluate(q0+.75*h*v0+h*h*9/32*acc(second),v0,t+.75*h)
                stages.extend((third,c['third']));pw.extend((1/8,-1/8));vw.extend((11/18,-11/18))
            q=q0+h*(1.5*v0-.5*previous);v=v0
            for stage,pw_i,vw_i in zip(stages,pw,vw):
                if pw_i:q=q+h*h*pw_i*acc(stage)
                if vw_i:v=v+h*vw_i*acc(stage)
            initial_weight,previous_weight=1.5,-.5
        else:
            accepted=StepHistory(accepted.maxlen)
            initial,_=evaluate(q0,v0,t);a0=acc(initial)
            midpoint,_=evaluate(q0+.5*h*v0+h*h/8*a0,v0,t+.5*h)
            endstage,_=evaluate(q0+h*v0+h*h/2*acc(midpoint),v0,t+h)
            q=q0+h*v0+h*h*(a0+2*acc(midpoint))/6
            v=v0+h*(a0+4*acc(midpoint)+acc(endstage))/6
            stages=[initial,midpoint,endstage];pw=[1/6,1/3,0.];vw=[1/6,2/3,1/6]
            # Seed old nodes at their physical old times and positions.
            second,_=evaluate(q0+c1*h*v0+h*h*a21*a0,v0,t+c1*h)
            stages.append(second);pw.append(0.);vw.append(0.)
            third=None
            if self.order==4:
                third,_=evaluate(q0+.75*h*v0+h*h*9/32*acc(second),v0,t+.75*h)
                stages.append(third);pw.append(0.);vw.append(0.)
            previous=None;initial_weight,previous_weight=1.,0.
        output=base.initializeNewState(*args,**kwargs);output.t=float(t+h)
        out=get_reference_state(output);setattr(out,position.name,q);setattr(out,velocity.name,v)
        endpoint,endpoint_state=evaluate(q,v,t+h)
        stages.append(endpoint);pw.append(0.);vw.append(0.)
        weights=ImprovedNystromWeights(tuple(pw),tuple(vw),tuple(0. for _ in stages),
            initial_weight,previous_weight,snapshot_state(previous))
        result=IntegrationResult(output,stages=stages,history=accepted,quadrature=weights)
        if not assess_step(result).successful:return result
        before=flatten_integrated(output).clone();before_uid=snapshot_state(getattr(out,'uid',None))
        finalizeSystem(output,base,dt,[s.aux for s in stages],[s.update for s in stages],weights,
            *args,lastStageSystem=endpoint_state,**kwargs)
        final_uid=getattr(out,'uid',None)
        if (not torch.equal(flatten_integrated(output),before) or float(output.t)!=float(t+h)
                or (before_uid is None)!=(final_uid is None) or before_uid is not None and not torch.equal(before_uid,final_uid)):
            raise ValueError('Improved Nyström finalize must preserve integrated values, time and IDs')
        context=dict(method=self.identifier.name,t=float(output.t),layout=_layout(output),
            current=flatten_integrated(output).clone(),uid=snapshot_state(final_uid),
            velocity=v0.clone(),initial=initial,second=second,third=third,endpoint=endpoint)
        memory=accepted.pushed(HistoryEntry(t,h,endpoint.update,endpoint.aux),context=context)
        # Own all stored updates and aux values, including reused history.
        return replace(result,history=snapshot_state(memory))
