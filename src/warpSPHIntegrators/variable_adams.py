"""Variable-step fixed-order Adams interpolation with transactional history.

Normalized Lagrange integration is equivalent to the pinned Adams constant-cache
Newton/divided-difference coefficients. AB uses its lower-order companion; ABM
uses the source order-(p-1) predictor, order-p corrector and source difference
estimate, with the public error sign main minus comparison. DP5 startup compares
against the available-history Adams predictor and reports its ramped exponent.
"""
from dataclasses import dataclass, fields
import math
from numbers import Real
from types import SimpleNamespace

import numpy as np
import torch

from .butcher import DormandPrince, _weighted_update
from .catalog import SchemeMetadata
from .fields import get_reference_state, integrated_field_names, state_difference, flatten_integrated
from .history import HistoryEntry, StepHistory
from .specs import IntegrationResult, StageResult
from .util import initializeSystem, finalizeSystem, updateStep, reject_prior_step


def adams_weights(nodes):
    """Integrate cardinal polynomials at distinct normalized nodes over [0,1]."""
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=1 or not len(nodes) or not np.isfinite(nodes).all() or len(np.unique(nodes))!=len(nodes):
        raise ValueError('Finite distinct one-dimensional Adams nodes required')
    result=[]
    for j,node in enumerate(nodes):
        polynomial=np.polynomial.Polynomial([1.])
        for k,other in enumerate(nodes):
            if k!=j:polynomial*=np.polynomial.Polynomial([-other,1.])/(node-other)
        primitive=polynomial.integ()
        result.append(primitive(1)-primitive(0))
    return np.asarray(result)


def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),str(getattr(ref,name).dtype))
                 for name in integrated_field_names(state))


@dataclass(frozen=True)
class VariableAdams:
    """Bind variable-step AB/ABM of order 3–5, with fixed/adaptive runner support.

    ``kind='ab'`` uses order-p interpolation and an order-(p-1) companion.
    ``kind='abm'`` predicts at order p-1 and corrects at order p, using the
    source's same-order difference estimate. There is no implicit solve.
    Empty/invalidated history starts with DP5; abrupt step-ratio changes outside
    [1/max_step_ratio,max_step_ratio] safely restart history. Initial states and
    history are cloned. Thread accepted result.history in a direct stepping loop.
    Pass restart_history=True at RHS/parameter or derivative discontinuities.
    Nonstiff ODEs only: explicit Adams has no general stiff stability guarantee.
    """
    order: int = 3
    kind: str = 'abm'
    max_step_ratio: float = 4.

    def __post_init__(self):
        if type(self.order) is not int or not 3<=self.order<=5 or self.kind not in ('ab','abm'):
            raise ValueError("Adams order must be 3, 4 or 5; kind must be 'ab' or 'abm'")
        if (not isinstance(self.max_step_ratio,Real) or isinstance(self.max_step_ratio,bool)
                or not math.isfinite(self.max_step_ratio) or self.max_step_ratio<1):
            raise ValueError('max_step_ratio must be finite and at least one')

    @property
    def name(self):return f'Variable Adams-{self.kind.upper()} {self.order}'
    @property
    def identifier(self):return SimpleNamespace(name=f'variable_adams_{self.kind}_{self.order}_{float(self.max_step_ratio).hex()}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.metadata.estimate_order
    @property
    def metadata(self):
        embedded=self.order-1 if self.kind=='ab' else self.order
        return SchemeMetadata(embedded_order=embedded,estimate_order=embedded+1,
                              history_length=self.order-1,history_requires_constant_dt=False)

    def __call__(self,state,dt,rhs,*args,history=None,restart_history=False,**kwargs):
        from .run import snapshot_state, assess_step
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:
            raise ValueError('finite nonzero dt required')
        if type(restart_history) is not bool:raise ValueError('restart_history must be a bool')
        reject_prior_step(self.name,kwargs.pop('priorStep',None))
        needed=self.order-1
        history=StepHistory(needed) if history is None else snapshot_state(history)
        if not isinstance(history,StepHistory) or history.maxlen<needed:
            raise ValueError('StepHistory capacity must cover order-1 derivatives')
        base=snapshot_state(state);layout=_layout(base);ref=get_reference_state(base)
        uid=getattr(ref,'uid',None)
        context=dict(method='variable_adams.v1',layout=layout,uid=snapshot_state(uid))
        valid=not restart_history and bool(history.entries) and isinstance(history.context,dict)
        if valid:
            old=history.context
            valid=(old.get('method')==context['method'] and old.get('layout')==layout
                   and (old.get('uid') is None)==(uid is None)
                   and (uid is None or torch.equal(old['uid'],uid)))
            latest=history.latest
            ratio=dt/latest.dt if latest.dt else -1
            time_tolerance=8*max(math.ulp(float(base.t)),math.ulp(latest.t),math.ulp(latest.t+latest.dt))
            valid=valid and (1-1e-12)/self.max_step_ratio<=ratio<=self.max_step_ratio*(1+1e-12)
            valid=valid and math.isclose(latest.t+latest.dt,float(base.t),rel_tol=0,abs_tol=time_tolerance)
            if valid:
                previous=float(base.t)
                for entry in reversed(history.entries):
                    if (previous-entry.t)*dt<=0 or entry.state is not None:valid=False;break
                    previous=entry.t
                    for field in fields(ref):
                        if field.name not in integrated_field_names(base):continue
                        rate=getattr(entry.update,field.metadata['update_key'],None)
                        value=getattr(ref,field.name)
                        if (not isinstance(rate,torch.Tensor) or rate.shape!=value.shape
                                or rate.dtype!=value.dtype or rate.device!=value.device):valid=False;break
        if not valid:history=StepHistory(history.maxlen,context=context)
        if len(history)<needed:
            started=DormandPrince(base,dt,rhs,*args,**kwargs)
            context=dict(method='variable_adams.v1',layout=_layout(base),
                         uid=snapshot_state(getattr(get_reference_state(base),'uid',None)))
            first=started.stages[0]
            partial=list(reversed(history.entries))
            nodes=[0.]+[(entry.t-float(base.t))/dt for entry in partial]
            compared=_weighted_update(base,[first.update]+[entry.update for entry in partial],
                                      adams_weights(nodes),dt,*args,**kwargs)
            startup_error=state_difference(started.state,compared);startup_error.t=float(base.t+dt)
            next_history=history
            if assess_step(started).successful:
                next_history=history.pushed(HistoryEntry(float(base.t),dt,first.update,first.aux),
                                            preserve_dt=True,context=context)
            return IntegrationResult(started.state,stages=started.stages,error=startup_error,
                history=next_history,solver_diagnostics=started.solver_diagnostics,estimate_order=len(nodes)+1)
        before=flatten_integrated(base).clone()
        initializeSystem(base,dt,*args,**kwargs)
        new_uid=getattr(get_reference_state(base),'uid',None)
        if (_layout(base)!=layout or not torch.equal(before,flatten_integrated(base))
                or (uid is None)!=(new_uid is None) or uid is not None and not torch.equal(context['uid'],new_uid)):
            raise ValueError('Warm Adams initialize must preserve integrated fields and particle IDs; restart history')
        start=base.initializeNewState(*args,**kwargs);start.t=float(base.t)
        current,aux=updateStep(base,start,dt,rhs,*args,**kwargs)
        past=list(reversed(history.entries))[:needed]
        updates=[current]+[entry.update for entry in past]
        auxes=[aux]+[entry.aux for entry in past]
        nodes=np.array([0.]+[(entry.t-float(base.t))/dt for entry in past])
        stages=[StageResult(aux,current)]
        if self.kind=='ab':
            weights=adams_weights(nodes)
            output=_weighted_update(base,updates,weights,dt,*args,**kwargs)
            compare=_weighted_update(base,updates[:-1],adams_weights(nodes[:-1]),dt,*args,**kwargs)
            used_updates,used_aux=updates,auxes
        else:
            prediction=_weighted_update(base,updates[:-1],adams_weights(nodes[:-1]),dt,*args,**kwargs)
            prediction.t=float(base.t+dt)
            predicted,predicted_aux=updateStep(base,prediction,dt,rhs,*args,**kwargs)
            stages.append(StageResult(predicted_aux,predicted))
            used_updates=[predicted]+updates[:-1];used_aux=[predicted_aux]+auxes[:-1]
            weights=adams_weights(np.r_[1.,nodes[:-1]])
            output=_weighted_update(base,used_updates,weights,dt,*args,**kwargs)
            compare=_weighted_update(base,[predicted]+updates,adams_weights(np.r_[1.,nodes]),dt,*args,**kwargs)
        output.t=float(base.t+dt)
        error=state_difference(output,compare);error.t=output.t
        endpoint=output.initializeNewState(*args,**kwargs);endpoint.t=output.t
        endpoint_update,endpoint_aux=updateStep(base,endpoint,dt,rhs,*args,**kwargs)
        stages.append(StageResult(endpoint_aux,endpoint_update))
        candidate=IntegrationResult(output,stages=stages,error=error,history=history,
                                    estimate_order=self.estimate_order)
        if not assess_step(candidate).successful:return candidate
        finalizeSystem(output,base,dt,used_aux,used_updates,weights,*args,lastStageSystem=endpoint,**kwargs)
        next_history=history.pushed(HistoryEntry(float(base.t),dt,current,aux),preserve_dt=True,context=context)
        return IntegrationResult(output,stages=stages,error=error,history=next_history,
                                 estimate_order=self.estimate_order)
