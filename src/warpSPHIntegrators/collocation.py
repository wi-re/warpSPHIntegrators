"""Stage-parameterized Gauss-Legendre/Radau IIA backend over the block solver.

The Legendre-basis construction integrates the cardinal polynomials without
forming a high-degree monomial inverse. Gauss uses the pinned FIRK constructor's
same-node order-(s-1) companion. Radau uses a derived unsmoothed order-s null-stage
companion; it does not reproduce the upstream filtered controller or order changes.
"""
from dataclasses import dataclass
from functools import cached_property
import math
from types import SimpleNamespace

import numpy as np

from .catalog import SchemeMetadata
from .fullyimplicit import BlockTableau, fullyImplicitBlock


def collocation_tableau(family, stages):
    """Build float64 collocation coefficients, with stages in [2, 9]."""
    if family not in ('gauss', 'radau'):
        raise ValueError("family must be 'gauss' or 'radau'")
    if type(stages) is not int or not 2 <= stages <= 9:
        raise ValueError('stages must be an integer from 2 through 9')
    leg = np.polynomial.legendre
    if family == 'gauss':
        x, quadrature = leg.leggauss(stages)
    else:
        # Right Radau nodes are the roots of P_s - P_{s-1}, including +1.
        polynomial = np.zeros(stages+1); polynomial[-1] = 1; polynomial[-2] = -1
        roots = leg.legroots(polynomial)
        if np.max(np.abs(roots.imag)) > 1e-12:
            raise ValueError('Radau node calculation produced nonreal roots')
        x = np.sort(roots.real); x[-1] = 1.
    c = (x+1)/2
    v = leg.legvander(x, stages-1)
    cardinal = np.linalg.solve(v, np.eye(stages))
    integrals = np.empty((stages, stages))
    for k in range(stages):
        polynomial = np.zeros(stages); polynomial[k] = 1
        primitive = leg.legint(polynomial)
        integrals[:, k] = (leg.legval(x, primitive)-leg.legval(-1., primitive))/2
    a = integrals@cardinal
    b = quadrature/2 if family == 'gauss' else a[-1].copy()
    if family == 'gauss':
        # Source sets the s-th monomial moment to zero. Express the resulting
        # moment functional in a Legendre basis to avoid a Vandermonde inverse.
        moments = []
        for k in range(stages):
            p = leg.Legendre.basis(k, domain=[0., 1.]).convert(kind=np.polynomial.Polynomial)
            moments.append(sum(p.coef[j]/(j+1) for j in range(min(k+1, stages-1))))
        embedded = np.linalg.solve(v.T, moments)
        companion = np.concatenate(([0.], embedded))
    else:
        # Interpolate f(0) from the s stage nodes. The defect is O(h^s),
        # so adding h*(sum L_j(0)*f_j-f(0)) gives a companion of order s.
        at_zero = np.array([(-1.)**k for k in range(stages)])@cardinal
        companion = np.concatenate(([-1.], b+at_zero))
    for array in (a, b, c, companion):array.setflags(write=False)
    return BlockTableau(a, b, c, companion)


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

    def __call__(self, state, dt, rhs, *args, **kwargs):
        from .run import snapshot_state
        if isinstance(dt, bool) or not math.isfinite(dt) or dt == 0:
            raise ValueError('finite nonzero dt required')
        return fullyImplicitBlock(snapshot_state(state), dt, rhs, self.tableau, *args,
                                  name=self.name, **kwargs)
