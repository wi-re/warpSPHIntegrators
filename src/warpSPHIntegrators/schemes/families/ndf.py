"""Fixed-target-order NDF with quasi-constant backward-difference history.

The warm kernel and R*U timestep transform match the pinned QNDF source. Startup
uses order-preserving L-stable DIRK, not the source's adaptive-order warmup.
Upstream order selection, dense interpolation and mass matrices are separate.
"""


from warpSPHIntegrators.schemes.coefficients.ndf import _KAPPA, _HARMONIC
from dataclasses import dataclass

import math
from numbers import Real
from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class NDFQuadrature:
    """Full-state affine step: y_new = known_state + h*sum(weights*f_i).

    Bootstrap uses ordinary DIRK weights, the initialized starting state and a
    zero-weight endpoint refresh. A warm step uses its known difference-history
    combination and one endpoint derivative with weight beta.
    """
    known_state: object
    weights: tuple
    bootstrap: bool


@dataclass(frozen=True)
class NDF:
    """Bind order 1–5 with owned variable-h differences and local defect control.

    Thread result.history on direct calls; solve owns trial rollback and portable
    TensorState continuation. Ratios outside the declared bound restart before
    use. Changed state values/layout/UID/time invalidate memory; changed RHS
    parameters require restart_history=True. Warm initialize and all finalize
    hooks preserve integrated values/IDs. Custom finalize explicitly opts into
    ndf_compatible=True and NDFQuadrature. No adaptive order, dense output or DAE.
    """
    order: int=3
    max_step_ratio: float=2.

    def __post_init__(self):
        if type(self.order) is not int or not 1<=self.order<=5:raise ValueError('NDF order must be from 1 through 5')
        if (isinstance(self.max_step_ratio,bool) or not isinstance(self.max_step_ratio,Real)
            or not math.isfinite(self.max_step_ratio) or not 1<=self.max_step_ratio<=2):
            raise ValueError('max_step_ratio must lie between one and two')
    @property
    def name(self):return f'NDF {self.order} (fixed target order)'
    @property
    def identifier(self):return SimpleNamespace(name=f'ndf_{self.order}_{float(self.max_step_ratio).hex()}')
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.order+1
    @property
    def metadata(self):
        return SchemeMetadata(estimate_order=self.estimate_order,history_length=self.order,
            history_requires_constant_dt=False)
    @property
    def kappa(self):return _KAPPA[self.order-1]
    @property
    def beta(self):return 1/((1-self.kappa)*_HARMONIC[self.order-1])
    @property
    def error_constant(self):return self.kappa*_HARMONIC[self.order-1]+1/(self.order+1)

    def next_step_size(self,result,h,proposal):
        """Hold accepted h through a quasi-constant block; reject trials may shrink.

        A deadband avoids repeated small resamplings. This fixed-order policy is
        separate from the upstream variable-order QNDF controller.
        """
        if result.history.context['constant_steps']<self.order+1:return h
        if 1.<=proposal/h<=1.5:return h
        return proposal

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.history.ndf import step
        return step(self,*args,**kwargs)
