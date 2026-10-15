"""Validated coupled RK coefficient definitions; no numerical driver dependency."""
from typing import NamedTuple, Optional
import numpy as np

class BlockTableau(NamedTuple):
    """A coupled (non-SDIRK) fully implicit RK tableau.

    ``a`` is s-by-s with a nonzero off-diagonal entry (the coupling that makes
    the stage equations a single block system), ``c`` the stage times, ``b`` the
    propagated-solution weights, and ``companion_b`` an optional (s+1)-vector of
    weights over the stages *plus an optional null first stage* ``k0 = f(t^n, y^n)`` at
    ``c0 = 0``. Companion order is recorded in scheme metadata. A zero first
    weight needs no extra evaluation. The 2-stage-only order-2 pair is degenerate for both legacy
    tableaus (it is ``b`` itself), so the estimate needs the null stage:
    ``companion_b[0] != 0`` costs one extra RHS evaluation per step.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    companion_b: Optional[np.ndarray]


def getBlockTableau(scheme: str) -> BlockTableau:
    if scheme == 'gaussLegendre3':
        # OrdinaryDiffEq 7.8.1 monorepo tree bae9b8e5f1d00e51c554648e5094e215164a19f8,
        # lib/OrdinaryDiffEqFIRK (2.8.1)/src/firk_tableaus.jl,
        # generateGaussLegendreTableau(..., 3). Exact collocation form;
        # source same-node order-2 companion bhat, no null RHS call.
        r = np.sqrt(15.)
        return BlockTableau(
            np.array([[5/36, 2/9-r/15, 5/36-r/30],
                      [5/36+r/24, 2/9, 5/36-r/24],
                      [5/36+r/30, 2/9+r/15, 5/36]]),
            np.array([5/18, 4/9, 5/18]),
            np.array([.5-r/10, .5, .5+r/10]),
            np.array([0., -5/6, 8/3, -5/6]))
    elif scheme == 'radauIia3':
        # Same pinned source: RadauIIA5Tableau (order 5, three stages).
        # A reconstructed from its inverse-stage transform matches these exact
        # collocation coefficients. Source e acts on stage increments, not RHS
        # weights. Our UNSMOOTHED order-3 solution companion is
        # bhat = (-1, b - e @ A); its error is h*(f0 + e @ A @ k).
        # This omits upstream's shifted-Jacobian smoothing/re-estimation and
        # must not be described as reproducing its stiff adaptive controller.
        r = np.sqrt(6.)
        a = np.array([[(88-7*r)/360, (296-169*r)/1800, (-2+3*r)/225],
                      [(296+169*r)/1800, (88+7*r)/360, (-2-3*r)/225],
                      [(16-r)/36, (16+r)/36, 1/9]])
        return BlockTableau(a, a[-1].copy(),
            np.array([(4-r)/10, (4+r)/10, 1.]),
            np.array([-1., 7/9+17*r/36, 7/9-17*r/36, 4/9]))
    elif scheme == 'radauIia2':
        # Radau IIA s=2: the collocation method at the Radau points {1/3, 1} --
        # the roots of 3x^2 - 2x - 1 on [0, 1], right endpoint pinned, so the
        # s=1 limit is exactly backward Euler. Order 3, A- and L-stable, stiffly
        # accurate (b = last row of a, c_2 = 1), coupled (a_12 != 0 -- a DIRK
        # driver cannot take it). Verified against the order conditions, the
        # collocation identity (A c)_i = c_i^2/2, and the stability function
        # R(z) = (1 + z/3) / (1 - 2z/3 + z^2/6) (L-stable: R(-100) ~= 0.019) by
        # hand; no Radau IIA tableau is published in SUNDIALS ARKODE v7.9.0, so
        # the derivation stands in for a citable source (NOTES.md S3.18).
        # Companion: min-norm order-2 quadrature over (k0, k1, k2), k0 the null
        # stage at c0 = 0; the 2-stage-only order-2 pair is b itself (degenerate).
        return BlockTableau(
            a=np.array([[5.0 / 12.0, -1.0 / 12.0],
                        [3.0 / 4.0, 1.0 / 4.0]]),
            b=np.array([3.0 / 4.0, 1.0 / 4.0]),
            c=np.array([1.0 / 3.0, 1.0]),
            companion_b=np.array([2.0 / 7.0, 9.0 / 28.0, 11.0 / 28.0]),
        )
    elif scheme == 'gaussLegendre2':
        # Gauss-Legendre 2: the collocation method at the Gauss points
        # 1/2 +- sqrt(3)/6. Order 4, A-stable with |R(iy)| = 1, symplectic for
        # separable Hamiltonians (pinned by measurement in
        # tests/test_hamiltonian.py; the tableau itself is NOT symmetric --
        # a_12 = (3 - 2 sqrt(3))/12 != a_21 = (3 + 2 sqrt(3))/12), NOT L-stable
        # (R(-100) ~= 0.887, and R(z) -> 1 as z -> -inf) and not stiffly
        # accurate (c_2 != 1). Stability function R(z) =
        # (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12), verified by hand as for
        # Radau IIA s=2 (NOTES.md S3.18). Companion as for radauIia2: min-norm
        # order-2 quadrature over (k0, k1, k2).
        sqrt3 = np.sqrt(3.0)
        c1 = 0.5 - sqrt3 / 6.0
        c2 = 0.5 + sqrt3 / 6.0
        return BlockTableau(
            a=np.array([[0.25, 0.25 - sqrt3 / 6.0],
                        [0.25 + sqrt3 / 6.0, 0.25]]),
            b=np.array([0.5, 0.5]),
            c=np.array([c1, c2]),
            companion_b=np.array([1.0 / 6.0, (5.0 - sqrt3) / 12.0, (5.0 + sqrt3) / 12.0]),
        )
    else:
        raise ValueError(f"Unknown block scheme {scheme}")
