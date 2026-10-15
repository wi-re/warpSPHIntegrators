

"""Accepted-order Adams control over the owned variable-step PECE backend."""
from dataclasses import dataclass
import math
from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata
from warpSPHIntegrators.state.fields import unflatten_integrated
from warpSPHIntegrators.schemes.families.variable_adams import VariableAdams


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

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.history.adaptive_adams import step
        return step(self,*args,**kwargs)

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
