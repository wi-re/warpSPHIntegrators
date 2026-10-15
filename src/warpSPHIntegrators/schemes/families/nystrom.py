"""Explicit generalized Runge–Kutta–Nyström for tagged mechanics systems.

Velocity-dependent tableaus lift extra first-order fields through their velocity
stage/final weights. Velocity-independent tableaus require passive extra fields.
Position and velocity quadratures are distinct and exposed to finalization.
"""


from warpSPHIntegrators.schemes.coefficients.nystrom import nystrom_tableau
from dataclasses import dataclass

import math
from numbers import Real
from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


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

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.mechanics.nystrom import step
        return step(self,*args,**kwargs)
