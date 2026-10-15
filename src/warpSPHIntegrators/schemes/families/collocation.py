"""Stage-parameterized Gauss-Legendre/Radau IIA backend over the block solver.

The Legendre-basis construction integrates the cardinal polynomials without
forming a high-degree monomial inverse. Gauss uses the pinned FIRK constructor's
same-node order-(s-1) companion. Radau uses a derived unsmoothed order-s null-stage
companion; it does not reproduce the upstream filtered controller or order changes.
"""

from warpSPHIntegrators.schemes.coefficients.collocation import collocation_tableau
from dataclasses import dataclass
from functools import cached_property

from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class Collocation:
    """Bind family/stage count for fixed or estimate-controlled block stepping.

    ``Collocation('gauss', 4)`` has order 8 and a source order-3 companion.
    ``Collocation('radau', 5)`` has order 9 and a derived order-5 companion.
    The existing fixed catalog entries retain their own coefficients/estimators.
    Supply a backend preconditioner through solver_opts for stiff spatial systems.
    No order switching, dense output, source filtering or constraint/DAE contract.
    """
    family: str = 'gauss'
    stages: int = 4

    def __post_init__(self):
        collocation_tableau(self.family, self.stages)

    @cached_property
    def tableau(self):return collocation_tableau(self.family, self.stages)
    @property
    def order(self):return 2*self.stages if self.family == 'gauss' else 2*self.stages-1
    @property
    def name(self):return f'{"Gauss-Legendre" if self.family == "gauss" else "Radau IIA"} s={self.stages}'
    @property
    def identifier(self):return SimpleNamespace(name=f'collocation_{self.family}_{self.stages}')
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return self.metadata.estimate_order
    @property
    def metadata(self):
        companion_order = self.stages-1 if self.family == 'gauss' else self.stages
        return SchemeMetadata(stage_count=self.stages, embedded_order=companion_order,
                              estimate_order=companion_order+1,
                              estimate_extra_stages=int(self.family == 'radau'))

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.collocation import step
        return step(self,*args,**kwargs)
