"""Accepted-order Adams control over the owned variable-step PECE backend."""
from dataclasses import dataclass, fields
import math
from types import SimpleNamespace
import numpy as np
import torch
from .catalog import SchemeMetadata
from .fields import get_reference_state, integrated_field_names, flatten_integrated, unflatten_integrated
from .history import StepHistory
from .variable_adams import VariableAdams, adams_weights, _layout


def adams_order_choice(order, error, lower_errors=None, upper_error=None, *, max_order=5):
    """Mature source VCABM neighbor comparison, bounded to orders 3–max_order.

    Lower defects are for p-2 and p-1; both must be no larger than the current
    normalized error before lowering. A smaller p+1 defect permits promotion.
    Package startup, maturity, zero-error and timestep policies are separate.
    """
    if type(max_order) is not int or not 3<=max_order<=5:raise ValueError('max_order must be 3–5')
    if type(order) is not int or not 3<=order<=max_order:raise ValueError('order outside configured range')
    def valid(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('finite nonnegative normalized errors required')
    valid(error)
    if lower_errors is not None:
        if not isinstance(lower_errors,(tuple,list)) or len(lower_errors)!=2:raise ValueError('two lower errors required')
        for value in lower_errors:valid(value)
    if upper_error is not None:valid(upper_error)
    if error==0:return order
    if lower_errors is not None and max(lower_errors)<=error:return max(3,order-1)
    if upper_error is not None and upper_error<error:return min(max_order,order+1)
    return order


def _defect_weights(nodes, order):
    """Source AM(q+1)-AM(q) difference on one shared endpoint derivative."""
    points=np.r_[1.,nodes];result=np.zeros(len(points))
    result[:order+1]=adams_weights(points[:order+1])
    result[:order]-=adams_weights(points[:order])
    return result


def _upper_weights(nodes, order):
    """Source dt*gamma_star[p+2]*phi_endpoint[p+2], normalized by outer dt."""
    points=np.r_[1.,np.asarray(nodes)[:order+1]]
    if len(points)!=order+2:raise ValueError('one extra history derivative required')
    gamma=(1.,-.5,-1/12,-1/24,-19/720,-3/160,-863/60480)
    scale=gamma[order+1]*np.prod(1-points[1:])
    return np.array([scale/np.prod(point-np.delete(points,i)) for i,point in enumerate(points)])


def _defect(template, updates, weights, dt):
    # Work on declared derivative fields directly: no RHS/lifecycle calls and no
    # subtraction of large state offsets in these neighboring-order defects.
    ref=get_reference_state(template);names=integrated_field_names(template);parts=[]
    for field in fields(ref):
        if field.name not in names:continue
        total=torch.zeros_like(getattr(ref,field.name))
        for update,weight in zip(updates,weights):
            if weight:total=total+float(weight)*getattr(update,field.metadata['update_key'])
        parts.append((dt*total).reshape(-1))
    return torch.cat(parts) if parts else flatten_integrated(template)*0


@dataclass(frozen=True)
class AdaptiveAdams:
    """Variable-step PECE with accepted target orders 3–5 (default maximum 5).

    Starts with target three and DP5 bootstrap, preserving VariableAdams ramped
    estimate exponents. Neighbor decisions use source VCABM defects and the
    runner's field norm, after p+1 mature steps. Order changes hold h once and
    retain compatible derivative history. Direct calls/fixed runs retain target
    order. This is not the full source 1–12 controller or its startup/FSAL policy.
    """
    max_order: int=5
    max_step_ratio: float=4.

    def __post_init__(self):
        if type(self.max_order) is not int or not 3<=self.max_order<=5:raise ValueError('max_order must be 3–5')
        VariableAdams(3,'abm',self.max_step_ratio)
    @property
    def name(self):return f'Adaptive Adams PECE (orders 3–{self.max_order})'
    @property
    def identifier(self):return SimpleNamespace(name=f'adaptive_adams_{self.max_order}_{float(self.max_step_ratio).hex()}')
    @property
    def order(self):return None
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 4
    @property
    def metadata(self):return SchemeMetadata(estimate_order=4,history_length=self.max_order,history_requires_constant_dt=False)

    def __call__(self,state,dt,f,*args,history=None,restart_history=False,**kwargs):
        from .run import snapshot_state
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if type(restart_history) is not bool:raise ValueError('restart_history must be bool')
        if history is not None and not isinstance(history,StepHistory):raise TypeError('StepHistory required')
        if history is not None and history.maxlen<self.max_order:raise ValueError('StepHistory capacity must cover max_order')
        accepted=snapshot_state(history) if history is not None else StepHistory(self.max_order)
        context=accepted.context;current=flatten_integrated(state)
        uid=getattr(get_reference_state(state),'uid',None)
        compatible=(not restart_history and isinstance(context,dict)
            and context.get('adaptive_adams')==self.identifier.name
            and type(context.get('order')) is int and 3<=context['order']<=self.max_order
            and context.get('layout')==_layout(state)
            and isinstance(context.get('current'),torch.Tensor)
            and context['current'].shape==current.shape and context['current'].dtype==current.dtype
            and context['current'].device==current.device and torch.equal(context['current'],current)
            and (uid is None)==(context.get('uid') is None)
            and (uid is None or torch.equal(uid,context['uid']))
            and math.isclose(context.get('t',float('inf')),float(state.t),rel_tol=0,
                abs_tol=8*math.ulp(float(state.t))))
        if compatible:
            ratio=dt/accepted.latest.dt if accepted.latest is not None else -1
            compatible=(1-1e-12)/self.max_step_ratio<=ratio<=self.max_step_ratio*(1+1e-12)
        order=context['order'] if compatible else 3
        age=context.get('age',0) if compatible else 0
        if not compatible:accepted=StepHistory(accepted.maxlen)
        result=VariableAdams(order,'abm',self.max_step_ratio)(state,dt,f,*args,history=accepted,**kwargs)
        if not result.step_status.successful:return result
        warm=result.estimate_order==order+1
        c=result.history.context
        c.update(adaptive_adams=self.identifier.name,order=order,last_order=order,
            age=age+1 if warm else 0,current=flatten_integrated(result.state).clone(),t=float(result.state.t))
        if warm:
            entries=list(reversed(result.history.entries))
            nodes=np.array([(entry.t-float(state.t))/dt for entry in entries])
            old=[entry.update for entry in entries];predicted=result.stages[1].update
            defects={f'lower_{q}':_defect(result.state,[predicted]+old,_defect_weights(nodes,q),dt)
                     for q in (order-2,order-1)}
            if order<self.max_order and len(entries)>=order+1:
                defects['upper']=_defect(result.state,[result.stages[-1].update]+old[:order+1],_upper_weights(nodes,order),dt)
            c['order_defects']=defects
        return result

    def adapt_step(self,result,h,proposal,*,error_norm,norm_estimate):
        c=result.history.context;order=c['order'];defects=c.pop('order_defects',{})
        if c['age']<order+1 or not defects:return proposal
        normalized={key:norm_estimate(unflatten_integrated(value,result.state)) for key,value in defects.items()}
        selected=adams_order_choice(order,float(error_norm),
            [normalized[f'lower_{q}'] for q in (order-2,order-1)],normalized.get('upper'),max_order=self.max_order)
        if selected!=order:
            c.update(order=selected,age=0)
            return h
        return proposal
