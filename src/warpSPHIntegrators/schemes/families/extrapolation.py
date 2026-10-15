"""Fixed-target explicit Euler/midpoint extrapolation over full-state fields."""

from warpSPHIntegrators.schemes.coefficients.extrapolation import extrapolation_coefficients
from dataclasses import dataclass


from types import SimpleNamespace

from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class ExtrapolationQuadrature:
    """Full affine weights over returned stage derivatives, including endpoint.

    Columns record the stage indices belonging to each subdivision sequence.
    The shared initial derivative can occur in several columns. Final weights
    already include all column coefficients and substep scales.
    """
    weights: tuple
    comparison_weights: tuple
    subdivisions: tuple
    columns: tuple


@dataclass(frozen=True)
class Extrapolation:
    """Fixed-target explicit extrapolation with adaptive timestep, not order.

    Main kernels match pinned AitkenNeville Euler / Deuflhard midpoint fixed
    columns. Midpoint is the unsmoothed even-subdivision source recurrence.
    Returned stages retain full tensors/graphs; this is not low-storage or
    internally threaded. Implicit/frozen-Jacobian extrapolation is separate.
    """
    base: str='midpoint'
    order: int=4
    def __post_init__(self):extrapolation_coefficients(self.base,self.order)
    @property
    def name(self):return f'{self.base.capitalize()} extrapolation (order {self.order})'
    @property
    def identifier(self):return SimpleNamespace(name=f'extrapolation_{self.base}_{self.order}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.order if self.base=='euler' or self.order==2 else self.order-1
    @property
    def metadata(self):
        ns,_,_=extrapolation_coefficients(self.base,self.order)
        return SchemeMetadata(stage_count=2+sum(n-1 for n in ns),
            embedded_order=self.estimate_order-1,estimate_order=self.estimate_order)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.extrapolation.extrapolation import step
        return step(self,*args,**kwargs)
