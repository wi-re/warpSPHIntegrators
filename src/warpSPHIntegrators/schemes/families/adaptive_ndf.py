"""Accepted-step adaptive order selection over the owned NDF backend."""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import torch
from warpSPHIntegrators.schemes.metadata import SchemeMetadata
from warpSPHIntegrators.state.fields import unflatten_integrated
from warpSPHIntegrators.schemes.families.ndf import NDF


def ndf_order_proposals(order,error,lower_error=None,upper_error=None):
    """Source QNDF accepted candidate ratios, before hold/deadband/order policy.

    Main/lower/upper safety factors are 1.2/1.3/1.4. The upstream special zero
    main error bypass is separate. None omits an unavailable neighboring order.
    """
    if type(order) is not int or not 1<=order<=5:raise ValueError('order must be from one through five')
    def proposal(value,q,safety,neighbor=False):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('finite nonnegative scalar errors required')
        z=safety*value**(1/(q+1))
        return 10. if z<=.1 else 0. if neighbor and z>safety else 1/z
    values=[(order,proposal(error,order,1.2))]
    if order>1 and lower_error is not None:values.append((order-1,proposal(lower_error,order-1,1.3,True)))
    if order<5 and upper_error is not None:values.append((order+1,proposal(upper_error,order+1,1.4,True)))
    return tuple(values)


@dataclass(frozen=True)
class AdaptiveNDF:
    """Variable accepted order 1–max_order with quasi-constant timestep blocks.

    The driver starts at order one. It compares accepted neighboring difference
    defects using the runner's field tolerances, and selects one order at a time.
    Use solve(...,adaptive=True) for order/step adaptation. Direct calls and fixed
    runs retain their current order. This is not the complete QNDF controller:
    startup is implicit RK, order changes wait for mature differences, retries
    use the runner's shrink policy and NDF's safe history restart.
    """
    max_order: int=5
    max_step_ratio: float=2.

    def __post_init__(self):
        if type(self.max_order) is not int or not 1<=self.max_order<=5:raise ValueError('max_order must be from one through five')
        NDF(1,self.max_step_ratio)
    @property
    def name(self):return f'Adaptive NDF (orders 1–{self.max_order})'
    @property
    def identifier(self):return SimpleNamespace(name=f'adaptive_ndf_{self.max_order}_{float(self.max_step_ratio).hex()}')
    @property
    def order(self):return None  # dynamic order, not a fixed-order catalog claim
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 2
    @property
    def metadata(self):return SchemeMetadata(estimate_order=2,history_length=self.max_order,history_requires_constant_dt=False)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.history.adaptive_ndf import step
        return step(self,*args,**kwargs)

    def adapt_step(self,result,h,proposal,*,error_norm,norm_estimate):
        """Select the next accepted order on owned trial memory, before commit."""
        context=result.history.context;order=context['order']
        if context['phase']!='warm' or context['constant_steps']<order+2:return h
        d=context['differences']
        def norm(q,index):
            error=unflatten_integrated(NDF(q).error_constant*d[index],result.state)
            error.t=result.state.t
            return norm_estimate(error)
        candidates=ndf_order_proposals(order,float(error_norm),
            norm(order-1,order-1) if order>1 else None,
            norm(order+1,order+1) if order<self.max_order else None)
        selected,ratio=max(candidates,key=lambda candidate:candidate[1])
        # A zero defect provides no evidence for raising order.
        if error_norm==0:selected=order;ratio=10.
        if selected!=order:
            differences=d[:selected+2]
            if selected>order:
                differences=torch.cat((d,torch.zeros_like(d[:1])),dim=0)
            context.update(order=selected,differences=differences,constant_steps=1)
            return h  # mature the new order before resampling its history
        return h if 1<=ratio<=1.5 else min(h*self.max_step_ratio,h*ratio)
