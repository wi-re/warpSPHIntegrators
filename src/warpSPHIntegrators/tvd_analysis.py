"""TVD / SSP (convex-combination) analysis for the registered schemes (NOTES S3.11).

`stability.py` answers the parabolic question -- "does this scheme stay bounded
on `y' = lambda y`?"; this module answers the hyperbolic one: "does one step of
this scheme, applied to the model upwind-advection semi-discretisation,
increase total variation?" That is the TVD property the TVD/SSP-named schemes
advertise, and it is measurable for every scheme that integrates a first-order
system.

Model problem
-------------
1D periodic first-order upwind advection ``u_t + c u_x = 0`` with
``(A u)_i = c (u_{i-1} - u_i) / h`` (``testing.advection_problem``), so the
scaled one-step operator is ``mu (S - I)`` with ``mu = c dt / h`` the CFL and
``S`` the periodic backward shift. Every Fourier mode ``v^{(m)}_i =
e^{2 pi i m i / n}`` is an exact eigenmode with eigenvalue
``lambda_m(mu) = mu (e^{-2 pi i m / n} - 1)``, so a Runge-Kutta step applied to
this problem is a circulant map on the grid whose mode-space values are found
by the stage recurrence

    Y_1 = y,                                   k_1 = mu (S - I) Y_1
    Y_j = y + sum_{i < j} a_ji k_i             (explicit stage)
    Y_j = (y + sum_{i < j} a_ji k_i) / (1 - a_jj mu (e^{-i theta} - 1))
                                                 (implicit stage, a_jj != 0)
    y^{n+1} = y + sum_j b_j k_j,               k_j = mu (S - I) Y_j

evaluated on all ``n`` modes at once (``stage_and_final_maps``).

Two measurements
---------------
1. ``convex_combination_cfl``: the SSP coefficient ``r`` -- the largest CFL
   such that every stage-value map *and* the final map is a convex combination
   of periodic shifts of the input (elementwise non-negative circulant first
   row, row sum 1) for all ``mu' <= mu``. A convex-combination step is TVD by
   construction (total variation is convex and shift-invariant), so ``r`` is a
   *rigorous* TVD CFL for every scheme that exposes its tableau.
2. ``measure_tvd_cfl``: the measured per-step TVD CFL -- run the registered
   driver on a step initial condition (``TV = 4``) and require the per-step
   total-variation increase to stay within tolerance in a post-burn-in window
   (the burn-in covers the multistep cold start, which is shared by all
   self-starting multistep schemes and is not a property of the step map).
   This one sees the actual driver, so it also classifies the schemes that
   expose no tableau (BDF/Adams family).

TVD is strictly broader than SSP: a scheme can have a TVD final map while its
internal stages leave the convex hull -- classical RK3 at CFL 1, whose stage 3
map is ``1 + zeta + zeta^2`` with a negative circulant entry ``mu (1 - 2 mu)
= -1`` while its final map (the shared third-order stability function) is a
convex combination up to CFL 1. ``classify_tvd`` therefore reports both
numbers: the rigorous SSP coefficient and the measured TVD CFL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .butcher import butcherTableau

#: The TVD RK schemes are hand-written Shu-Osher forms with no tableau
#: attribute (``tvd.py``). These Butcher tableaus are algebraically the same
#: methods: TVD RK2's stages are ``Y_2 = y + dt k_1`` and
#: ``y^{n+1} = y + dt (1/2 k_1 + 1/2 k_2)`` (Heun's method), and TVD RK3's are
#: ``Y_2 = y + dt k_1``, ``Y_3 = y + dt (1/4 k_1 + 1/4 k_2)`` and
#: ``y^{n+1} = y + dt (1/6 k_1 + 1/6 k_2 + 2/3 k_3)`` -- the latter is the
#: published SSP(2,3) tableau, which ``butcher.SSPRK3`` carries. The
#: equivalence is pinned to the actual registered drivers by
#: ``tests/test_tvd.py`` (a driver cross-check on the eigenmode initialisation),
#: so the classifier never rests on the name alone.
TVD_RK2_TABLEAU = butcherTableau(
    a=np.array([[0.0, 0.0], [1.0, 0.0]]),
    b=np.array([0.5, 0.5]),
    c=np.array([0.0, 1.0]),
)
TVD_RK3_TABLEAU = butcherTableau(
    a=np.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.25, 0.0]]),
    b=np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    c=np.array([0.0, 1.0, 0.5]),
)


# --------------------------------------------------------------------------- #
# Stage maps on the upwind model                                               #
# --------------------------------------------------------------------------- #

def upwind_eigenvalues(mu: float, n: int = 128) -> np.ndarray:
    """``lambda_m(mu) = mu (e^{-2 pi i m / n} - 1)``, m = 0..n-1."""
    theta = 2.0 * np.pi * np.arange(n) / n
    return mu * (np.exp(-1j * theta) - 1.0)


def _first_row(eigenvalues: np.ndarray) -> np.ndarray:
    """First row of the circulant whose mode eigenvalues are given.

    For the circulant ``C`` with ``C_{ij} = r_{(j - i) mod n}`` the eigenvalue
    on ``v^{(m)}_i = e^{2 pi i m i / n}`` is
    ``lambda_m = sum_k r_k e^{+2 pi i m k / n}``, so ``r = fft(lambda) / n``
    (checked against ``S``: ``(S u)_i = u_{i-1}`` gives ``r_{n-1} = 1``).
    """
    return np.fft.fft(eigenvalues) / len(eigenvalues)


def stage_and_final_maps(tableau, mu: float, n: int = 128):
    """First rows of the stage-value and final-value maps for one step of the
    (possibly implicit) Runge-Kutta method ``tableau`` applied to ``mu (S - I)``.

    Returns ``(stage_rows, final_row)``; ``stage_rows[j]`` is the map
    ``y^n -> Y_{j+1}``. The embedded-pair convention is honoured: for a
    ``butcherTableau`` whose ``b`` is a tuple, the propagated (first) weight
    vector is used -- the final map of the method that advances the solution.
    """
    a = np.asarray(tableau.a, dtype=float)
    b = tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b
    b = np.asarray(b, dtype=float)
    s = len(tableau.c)
    # zeta is the eigenvalue of the *scaled* operator mu (S - I), so the stage
    # derivatives k = zeta * Y already carry the CFL factor: the Butcher
    # update Y_j = y + sum_i a_ji k_i (and y^{n+1} = y + sum_j b_j k_j)
    # takes no further factor of mu.
    zeta = upwind_eigenvalues(mu, n)
    y = np.ones(n, dtype=complex)
    ks = []
    rows = []
    for j in range(s):
        explicit_part = y
        for i in range(j):
            if a[j, i] != 0.0:
                explicit_part = explicit_part + a[j, i] * ks[i]
        if a[j, j] != 0.0:
            Y = explicit_part / (1.0 - a[j, j] * zeta)
        else:
            Y = explicit_part
        ks.append(zeta * Y)
        rows.append(_first_row(Y))
    Y_final = y
    for j in range(s):
        if b[j] != 0.0:
            Y_final = Y_final + b[j] * ks[j]
    return rows, _first_row(Y_final)


def is_convex_row(row: np.ndarray, tol: float = 1e-10) -> bool:
    """True if the circulant first row is a convex combination: elementwise
    non-negative (real, within ``tol``) and row sum 1."""
    r = np.real(row)
    return bool(r.min() >= -tol and abs(r.sum() - 1.0) <= 100.0 * tol)


def _min_row_entry(tableau, mu: float, n: int = 128) -> float:
    """Smallest (real) circulant entry over all stage rows and the final row."""
    rows, final = stage_and_final_maps(tableau, mu, n)
    return float(min(np.real(r).min() for r in rows + [final]))


def convex_combination_cfl(tableau, mu_max: float = 4.0, n: int = 128,
                           scan: int = 400, tol: float = 1e-10) -> float:
    """The SSP coefficient: the largest ``mu`` such that every stage-value map
    and the final map is a convex combination of periodic shifts for all
    ``mu' in [0, mu]``.

    Returns ``mu_max`` (read as "convex up to at least ``mu_max``") when no
    failure is found on ``[0, mu_max]``. A scan from 0 finds the first
    failure and a bisection refines the boundary; the passing set is an
    initial interval for every registered tableau (the row entries are
    low-degree polynomials -- or rational functions with positive poles -- in
    ``mu``, with ``r_0(0) = 1``).

    A reported boundary is only a boundary when a row entry genuinely
    *crosses* zero there. An entry that is negative at every numerically
    resolvable ``mu`` merely has its magnitude cross ``tol`` somewhere, which
    the scan would otherwise report as the coefficient (Nystrom 5th order:
    its stage-5 entry at circulant offset 3 is ``7.995e-18 mu^3 -
    0.1185 mu^4`` -- negative for every ``mu`` above ``~1e-16``, checked by
    exact rational arithmetic, so its true coefficient is 0). The refinement
    probes ``lo/2`` and ``lo/4``: at a genuine crossing the entry's magnitude
    grows as the probe moves away from the root, while an everywhere-negative
    entry shrinks at least as fast as ``(1/2)^d`` per halving (``d`` its
    polynomial degree in ``mu``). ``0.0`` is returned in the latter case.
    """
    def passes(mu: float) -> bool:
        if mu <= 0.0:
            return True
        rows, final = stage_and_final_maps(tableau, mu, n)
        return all(is_convex_row(r, tol) for r in rows + [final])

    floor = 1e-13  # above the ~1e-16 FFT round-off floor at n = 128
    grid = np.linspace(0.0, mu_max, scan)
    for i in range(1, scan):
        if not passes(grid[i]):
            lo, hi = grid[i - 1], grid[i]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if passes(mid):
                    lo = mid
                else:
                    hi = mid
            lo = float(lo)
            m2 = _min_row_entry(tableau, 0.5 * lo, n)
            if m2 <= -floor:
                m4 = _min_row_entry(tableau, 0.25 * lo, n)
                if m4 <= -floor and abs(m4) < 0.5 * abs(m2):
                    return 0.0
            return lo
    return float(mu_max)


# --------------------------------------------------------------------------- #
# Trajectory total variation                                                   #
# --------------------------------------------------------------------------- #

def total_variation(u: torch.Tensor) -> float:
    """Periodic total variation ``sum_i |u_{i+1} - u_i|``."""
    return float((torch.roll(u, -1) - u).abs().sum())


def tv_per_step_increases(scheme, problem, dt: float, n_steps: int,
                          burn_in: int = 0) -> list:
    """Per-step total-variation increases ``TV_{n+1} - TV_n`` of a fixed-step
    run of the registered driver ``scheme`` on ``problem`` from its initial
    condition, dropping the first ``burn_in`` steps (the multistep cold start
    is shared by all self-starting multistep schemes; it is not a property of
    the step map itself)."""
    from .fields import get_reference_state

    system = problem.initial()
    prev = total_variation(get_reference_state(system).x)
    increases = []
    for _ in range(n_steps):
        system = scheme(system, dt=dt, f=problem.rhs).state
        cur = total_variation(get_reference_state(system).x)
        increases.append(cur - prev)
        prev = cur
    return increases[burn_in:]


@dataclass(frozen=True)
class TVDMeasurement:
    """Outcome of the trajectory sweep for one scheme.

    ``tvd_cfl`` is the largest CFL of the sweep at which the per-step TV
    increases stayed within tolerance; ``unconditional`` flags a pass at the
    top of the sweep (a lower bound, not a proof). ``worst_increase`` is the
    largest post-burn-in per-step increase at the top CFL that was tried, as a
    fraction of the initial TV.
    """
    tvd_cfl: float
    unconditional: bool
    worst_increase: float
    detail: str


def measure_tvd_cfl(scheme, problem, cfls, dt_scale: float,
                    n_steps: int = 50, burn_in: int = 10,
                    tol: float = 1e-8) -> TVDMeasurement:
    """Measured per-step TVD CFL of the registered driver ``scheme``.

    ``cfls`` is an ascending CFL grid; ``dt = mu * dt_scale`` (``dt_scale =
    h / c`` for advection, ``h / alpha`` for Burgers). A CFL passes when every
    post-burn-in per-step TV increase is within ``tol`` of the initial TV.
    The run is stopped at the first failing CFL, so the cost grows only with
    the CFL the scheme actually holds.
    """
    tv0 = total_variation(problem.initial().state.x)
    passed = 0.0
    worst = float('inf')
    for mu in cfls:
        increases = tv_per_step_increases(
            scheme, problem, dt=float(mu) * dt_scale,
            n_steps=n_steps, burn_in=burn_in)
        rel = max((inc / tv0 for inc in increases), default=0.0)
        if rel <= tol:
            passed = float(mu)
            worst = rel
        else:
            return TVDMeasurement(
                tvd_cfl=passed, unconditional=False, worst_increase=rel,
                detail=f'per-step TV increase {rel:.3e} * TV0 at CFL {mu:g}')
    return TVDMeasurement(
        tvd_cfl=passed, unconditional=True, worst_increase=worst,
        detail=f'no TV increase above {tol:g} * TV0 up to CFL {cfls[-1]:g}')


# --------------------------------------------------------------------------- #
# The classifier                                                               #
# --------------------------------------------------------------------------- #

#: Schemes that integrate second-order systems ``x'' = f(x, x')``: running them
#: on a first-order advection problem is meaningless (the driver would treat
#: the field as a position with zero velocity), so the TVD question does not
#: apply to them.
NOT_APPLICABLE_IDENTIFIERS = {
    'leapFrog', 'symplecticEuler', 'velocityVerlet', 'pefrl', 'vefrl',
    'newmark',
}

#: Parabolic super-timestepping schemes (RKC1 / RKC2 / RKL2, NOTES.md S3.13).
#: They take a per-step stage count and are built to stabilise real negative
#: (parabolic) eigenvalues, so they cannot be run on the advection problem as-is
#: and the hyperbolic TVD / SSP question does not apply to them either.
PARABOLIC_ONLY_IDENTIFIERS = {'rkc1', 'rkc2', 'rkl2'}

#: Structured-semilinear-only schemes (NOTES.md S3.15). These integrate the
#: linear part ``L`` exactly and therefore require the ``linear`` accessor of a
#: ``SemilinearRHS`` (a plain callable carries only the combined ``f``). The
#: model advection problem is registered as a plain callable, so such a scheme
#: cannot be run on it as registered -- the hyperbolic TVD / SSP measurement
#: does not apply to it here. (Rosenbrock-W is *not* in this set: it falls back
#: to the combined ``f``'s Jacobian for its frozen operator, so it runs on a
#: plain callable.)
SEMILINEAR_ONLY_IDENTIFIERS = {'etd2rk'}


@dataclass(frozen=True)
class TVDVerdict:
    """The TVD classification of one registered scheme.

    ``ssp_cfl`` is the rigorous convex-combination (SSP) coefficient of the
    scheme's stage maps, or None when the scheme exposes no tableau.
    ``tvd_cfl`` is the measured per-step TVD CFL from the trajectory sweep, or
    None when the TVD question does not apply. ``unconditional`` flags a pass
    at the top of the sweep grid (a lower bound).
    """
    scheme: str
    order: int
    ssp_cfl: Optional[float]
    tvd_cfl: Optional[float]
    unconditional: bool
    verdict: str
    detail: str


def scheme_tableau(scheme):
    """The (implicit-)Runge-Kutta tableau a scheme exposes, or None.

    Explicit RK schemes carry ``.butcherTableau``; DIRK schemes
    ``.dirkTableau``; ARK schemes ``.arkTableau`` (for an ordinary RHS the
    driver runs the pure-implicit limit, which is the ``a_implicit`` /
    ``b_implicit`` half, so that half is what gets measured). TVD RK2 / TVD
    RK3 expose nothing (hand-written Shu-Osher forms, ``tvd.py``); their
    algebraically identical Butcher equivalents are returned instead (see
    ``TVD_RK2_TABLEAU`` / ``TVD_RK3_TABLEAU``).
    """
    fn = scheme.function
    t = getattr(fn, 'butcherTableau', None)
    if t is not None:
        return t
    t = getattr(fn, 'dirkTableau', None)
    if t is not None:
        return t
    t = getattr(fn, 'arkTableau', None)
    if t is not None:
        return butcherTableau(a=np.asarray(t.a_implicit, dtype=float),
                              b=np.asarray(t.b_implicit, dtype=float),
                              c=np.asarray(t.c, dtype=float))
    if scheme.name == 'TVD RK2':
        return TVD_RK2_TABLEAU
    if scheme.name == 'TVD RK3':
        return TVD_RK3_TABLEAU
    return None


def classify_tvd(scheme, problem, cfls, dt_scale: float,
                 mu_max: float = 4.0, n: int = 128,
                 n_steps: int = 50, burn_in: int = 10,
                 tol: float = 1e-8) -> TVDVerdict:
    """Classify one registered scheme: rigorous SSP coefficient from its
    stage maps (when it exposes a tableau) plus the measured per-step TVD CFL
    from the trajectory sweep on ``problem`` (a step initial condition)."""
    if scheme.identifier.name in NOT_APPLICABLE_IDENTIFIERS:
        return TVDVerdict(
            scheme=scheme.name, order=scheme.order, ssp_cfl=None,
            tvd_cfl=None, unconditional=False,
            verdict='not applicable (second-order system scheme)',
            detail='integrates x\'\' = f, not a first-order system')

    if scheme.identifier.name in PARABOLIC_ONLY_IDENTIFIERS:
        return TVDVerdict(
            scheme=scheme.name, order=scheme.order, ssp_cfl=None,
            tvd_cfl=None, unconditional=False,
            verdict='not applicable (parabolic super-timestepping)',
            detail='per-step stage count (s=), no tableau; stabilises parabolic '
                   'terms, the hyperbolic TVD/SSP question does not apply')

    if scheme.identifier.name in SEMILINEAR_ONLY_IDENTIFIERS:
        return TVDVerdict(
            scheme=scheme.name, order=scheme.order, ssp_cfl=None,
            tvd_cfl=None, unconditional=False,
            verdict='not applicable (structured semilinear RHS required)',
            detail='integrates the linear part exactly via phi_k(hL), so it '
                   'needs the `linear` accessor of a SemilinearRHS; the model '
                   'advection problem is a plain callable, so the step cannot '
                   'be run on it as registered')

    ssp_cfl = None
    tableau = scheme_tableau(scheme)
    if tableau is not None:
        ssp_cfl = convex_combination_cfl(tableau, mu_max=mu_max, n=n)

    measurement = measure_tvd_cfl(scheme, problem, cfls, dt_scale,
                                  n_steps=n_steps, burn_in=burn_in, tol=tol)
    tvd_cfl = measurement.tvd_cfl
    unconditional = measurement.unconditional
    detail = measurement.detail
    if ssp_cfl is not None and ssp_cfl < mu_max - 1e-12:
        detail += f'; stage maps stop being convex combinations at CFL {ssp_cfl:.4g}'

    if tvd_cfl <= 0.0:
        verdict = f'not TVD ({detail})'
    elif ssp_cfl is not None and ssp_cfl <= 0.0:
        # A stage map leaves the convex hull immediately (a row entry is
        # negative at every numerically resolvable CFL), so no convex-
        # combination (SSP) CFL exists; only the measured per-step number
        # below applies.
        verdict = (f'not an SSP step (stage maps leave the convex hull for any '
                   f'resolvable CFL); measured TVD up to CFL {tvd_cfl:g}')
    elif ssp_cfl is not None:
        # The scheme exposes a tableau, so the stage-map coefficient is the
        # rigorous number; the trajectory sweep confirms it.
        if ssp_cfl >= mu_max - 1e-12:
            verdict = (f'unconditionally TVD: every stage map and the final map '
                       f'is a convex combination of shifts up to CFL {mu_max:g} '
                       f'(trajectory sweep agrees to CFL {tvd_cfl:g})')
        else:
            verdict = (f'SSP(r = {ssp_cfl:.4g}): a convex-combination step up to '
                       f'CFL {ssp_cfl:.4g}, hence TVD there (trajectory sweep '
                       f'agrees to CFL {tvd_cfl:g})')
    elif unconditional:
        verdict = f'unconditionally TVD (measured; tested to CFL {cfls[-1]:g})'
    else:
        verdict = f'TVD up to CFL {tvd_cfl:g} (measured; not a convex-combination step)'
    return TVDVerdict(
        scheme=scheme.name, order=scheme.order, ssp_cfl=ssp_cfl,
        tvd_cfl=tvd_cfl, unconditional=unconditional, verdict=verdict,
        detail=detail)


def classify_all(schemes, problem, cfls, dt_scale: float, **kwargs) -> list:
    """``classify_tvd`` over a list of schemes, in the given order."""
    return [classify_tvd(s, problem, cfls, dt_scale, **kwargs) for s in schemes]
