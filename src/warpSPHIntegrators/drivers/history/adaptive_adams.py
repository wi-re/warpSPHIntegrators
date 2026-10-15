"""Numerical stepping for adaptive_adams; configuration and coefficients are supplied separately."""


import math


import numpy as np

import torch


from warpSPHIntegrators.state.fields import get_reference_state, flatten_integrated

from warpSPHIntegrators.state.history import StepHistory

from warpSPHIntegrators.schemes.families.variable_adams import VariableAdams
from warpSPHIntegrators.drivers.history._adams_layout import _layout


from warpSPHIntegrators.schemes.coefficients.adams_control import _defect_weights, _upper_weights
from warpSPHIntegrators.drivers.history._adams import _defect

def step(self,state,dt,f,*args,history=None,restart_history=False,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state
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
