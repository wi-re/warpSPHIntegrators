

"""Variable-step fixed-order Adams interpolation with transactional history.

Normalized Lagrange integration is equivalent to the pinned Adams constant-cache
Newton/divided-difference coefficients. AB uses its lower-order companion; ABM
uses the source order-(p-1) predictor, order-p corrector and source difference
estimate, with the public error sign main minus comparison. DP5 startup compares
against the available-history Adams predictor and reports its ramped exponent.
"""
from dataclasses import dataclass
import math
from numbers import Real
from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


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

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.history.variable_adams import step
        return step(self,*args,**kwargs)
