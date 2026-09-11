"""RKC1 / RKC2 / RKL2 -- relaxed Chebyshev / Lobatto super-timestepping (NOTES S3.13, Phase 12).

The three schemes are stabilised *explicit* methods for stiff parabolic
right-hand sides: ``s`` matrix-free right-hand-side evaluations per step buy a
real-axis stability interval that grows like ``O(s^2)``. They are deliberately
excluded from the shared ``scheme`` fixture (tests/conftest.py) because the
stage count ``s`` is a per-step parameter the generic one-step-callable tests do
not provide, and because their real-axis stability analysis does not apply to the
generic oscillator (imaginary eigenvalues). This file is where they are exercised
on the problem they are built for -- the semi-discrete diffusion in
``testing`` -- plus the stage-count rule and the driver contract.

The recurrence was derived and validated independently (scalar stability
polynomial to round-off, intermediate-stage boundedness, and order on the
semi-linear diffusion) before the driver was written; these tests pin that
behaviour through the real machinery.
"""

import math

import pytest
import torch

from conftest import ORDER_TOLERANCE
from warpSPHIntegrators import (
    RKC1,
    RKC2,
    RKL2,
    StageResult,
    get_reference_state,
    stage_count,
    testing,
)

#: (driver, family key, registered order) for each of the three schemes.
SCHEMES = [
    (RKC1, 'rkc1', 1),
    (RKC2, 'rkc2', 2),
    (RKL2, 'rkl2', 2),
]


def _K(family, s):
    """Real-axis stability interval: stable for ``dt * |lambda_max| <= K(s)``."""
    if family == 'rkc1':
        return 2.0 * s * s
    if family == 'rkc2':
        return 2.0 * (s * s - 1) / 3.0
    return (s * s + s - 2) / 2.0


_SMIN = {'rkc1': 1, 'rkc2': 3, 'rkl2': 3}
#: A comfortably valid stage count for the one-step driver-contract tests.
_VALID_S = {'rkc1': 5, 'rkc2': 5, 'rkl2': 5}


def _diffusion(n=32):
    return testing.PROBLEMS['diffusion'](n=n)


def _diffusion_lam_max(n, D=1.0, L=1.0):
    """Magnitude of the stiffest (highest-mode) Dirichlet Laplacian eigenvalue."""
    h = L / (n + 1)
    return (4.0 * D / h ** 2) * math.sin(n * math.pi * h / (2.0 * L)) ** 2


def _integrate(scheme, prob, system, dt, T, s_stages):
    """Integrate from ``system`` (not ``prob.initial()``) so a custom IC survives."""
    for _ in range(int(round(T / dt))):
        system = scheme(system, dt=dt, f=prob.rhs, s=s_stages).state
    return system


# --------------------------------------------------------------------------- #
# Convergence order on the semi-discrete diffusion                            #
# --------------------------------------------------------------------------- #

T_DIFF = 0.02
DTS_DIFF = [T_DIFF / m for m in (4, 8, 16, 32)]


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rkc_achieves_its_registered_order_on_diffusion(scheme, family, order):
    """RKC1 is order 1; RKC2 and RKL2 are order 2, measured on the parabolic
    problem they are built for. ``s`` is held fixed (the order is independent of
    the stage count; ``s`` only sets the stability interval)."""
    prob = _diffusion(n=32)
    # s = 12 keeps every dt in DTS_DIFF well inside K(s); the order is s-independent.
    measured, errors = testing.convergence(scheme, prob, DTS_DIFF, T_DIFF, s=12)
    assert measured is not None, f'{family}: no usable errors: {errors}'
    assert measured >= order - ORDER_TOLERANCE, (
        f'{family}: registered order {order}, measured {measured:.2f}. '
        f'Errors: {[f"{e:.3e}" for e in errors]}')


# --------------------------------------------------------------------------- #
# Stability: bounded iff dt * |lambda_max| <= K(s)                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_stability_boundary_sits_at_K(scheme, family, order):
    """A multi-mode (random) IC excites the stiffest mode, so the run is bounded
    just inside ``dt * |lambda_max| = K(s)`` and blows up just outside it. The
    margins (0.8 / 1.5) clear the ~5-10% the measured boundary sits from the
    theoretical ``K``."""
    n = 16
    s = 8
    prob = _diffusion(n=n)
    lam_max = _diffusion_lam_max(n)
    system = prob.initial()
    rng = torch.Generator().manual_seed(0)
    get_reference_state(system).x = torch.randn(n, generator=rng, dtype=torch.float64)
    u0_max = get_reference_state(system).x.abs().max().item()
    T = 0.5

    dt_in = 0.8 * _K(family, s) / lam_max
    in_max = get_reference_state(_integrate(scheme, prob, system, dt_in, T, s)).x.abs().max().item()
    assert in_max < 1.5 * u0_max, (
        f'{family}: unstable inside the boundary (dt*|lam| = {dt_in * lam_max:.2f} < '
        f'K = {_K(family, s):.1f}); max grew to {in_max / u0_max:.2f}x')

    dt_out = 1.5 * _K(family, s) / lam_max
    out_max = get_reference_state(_integrate(scheme, prob, system, dt_out, T, s)).x.abs().max().item()
    assert out_max > 10.0 * u0_max, (
        f'{family}: still bounded outside the boundary (dt*|lam| = {dt_out * lam_max:.2f} > '
        f'K = {_K(family, s):.1f}); max only {out_max / u0_max:.2f}x')


# --------------------------------------------------------------------------- #
# The stage-count rule                                                       #
# --------------------------------------------------------------------------- #

def test_stage_count_is_the_smallest_admissible():
    for family in ('rkc1', 'rkc2', 'rkl2'):
        smin = _SMIN[family]
        # Below the method's own K(smin) the count clamps to the minimum.
        assert stage_count(0.0, family) == smin
        assert stage_count(1e-12, family) == smin
        assert stage_count(_K(family, smin), family) == smin
        # Tightness: an argument just above K(s-1) and one at K(s) both give s.
        for s in range(smin + 1, smin + 40):
            assert stage_count(_K(family, s - 1) * (1.0 + 1e-9), family) == s
            assert stage_count(_K(family, s), family) == s


def test_stage_count_grows_like_s_squared():
    """Doubling ``s`` quadruples ``K``; so 4x the argument needs ~2x the count."""
    for family in ('rkc1', 'rkc2', 'rkl2'):
        arg = 1000.0
        s1 = stage_count(arg, family)
        s4 = stage_count(4.0 * arg, family)
        assert 1.5 < s4 / s1 < 2.5, (
            f'{family}: 4x the argument took the stage count from {s1} to {s4}, '
            f'not ~2x')


# --------------------------------------------------------------------------- #
# s= and lambda_max= are two views of the same choice                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_lambda_max_selects_the_same_stage_count_as_s(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    dt, lam = 0.01, 100.0
    s = stage_count(dt * lam, family)
    via_lambda = scheme(system, dt=dt, f=prob.rhs, lambda_max=lam).state
    via_s = scheme(system, dt=dt, f=prob.rhs, s=s).state
    assert torch.equal(get_reference_state(via_lambda).x, get_reference_state(via_s).x)


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_stage_count_below_the_minimum_is_clamped(scheme, family, order):
    """Passing ``s`` under the method's minimum is silently raised to it."""
    prob = _diffusion(n=16)
    system = prob.initial()
    low = scheme(system, dt=0.01, f=prob.rhs, s=1).state
    min = scheme(system, dt=0.01, f=prob.rhs, s=_SMIN[family]).state
    assert torch.equal(get_reference_state(low).x, get_reference_state(min).x)


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rkc_requires_a_stage_count(scheme, family, order):
    """Without ``s=`` (or ``lambda_max=``) there is no way to size the step."""
    prob = _diffusion(n=16)
    system = prob.initial()
    with pytest.raises(ValueError, match='pass s='):
        scheme(system, dt=0.01, f=prob.rhs)


# --------------------------------------------------------------------------- #
# The driver contract                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rkc_rejects_prior_step_with_a_warning(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    prior = StageResult(aux=None, update=None)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        scheme(system, dt=0.01, f=prob.rhs, s=_VALID_S[family], priorStep=prior)


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rhs_is_evaluated_s_times_per_step(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    counts = []

    def counting_rhs(state, dt, **kwargs):
        counts.append(1)
        return prob.rhs(state, dt, **kwargs)

    s = _VALID_S[family]
    scheme(system, dt=0.01, f=counting_rhs, s=s)
    assert len(counts) == s, (
        f'{family}: expected exactly {s} right-hand-side evaluations per step, got {len(counts)}')


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_stage_times_stay_within_the_step_and_open_at_t(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    system.t = 1.0
    dt, s = 0.1, _VALID_S[family]
    times = []

    def recording_rhs(state, dt_, **kwargs):
        times.append(float(state.t))
        return prob.rhs(state, dt_, **kwargs)

    scheme(system, dt=dt, f=recording_rhs, s=s)
    assert len(times) == s, f'{family}: {len(times)} evaluations, expected {s}'
    assert times[0] == pytest.approx(1.0), (
        f'{family}: first evaluation at t={times[0]}, expected t={1.0}')
    assert all(1.0 - 1e-12 <= t <= 1.0 + dt + 1e-12 for t in times), (
        f'{family}: a stage was evaluated outside [t, t+dt]: {times}')


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_final_time_advances_by_dt_and_stays_a_float(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    system.t = 0.75
    result = scheme(system, dt=0.1, f=prob.rhs, s=_VALID_S[family])
    assert result.state.t == pytest.approx(0.85)
    assert type(result.state.t) is float, (
        f'{family}: t became {type(result.state.t)}, expected a python float')


@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rkc_does_not_mutate_the_caller_state(scheme, family, order):
    prob = _diffusion(n=16)
    system = prob.initial()
    before_x = system.state.x.clone()
    before_t = system.t
    result = scheme(system, dt=0.01, f=prob.rhs, s=_VALID_S[family])
    assert torch.equal(system.state.x, before_x), f'{family} mutated the caller field'
    assert system.t == before_t, f'{family} mutated the caller time'
    assert get_reference_state(result.state).x is not system.state.x, (
        f'{family}: the final state aliases the caller buffer instead of owning one')


# --------------------------------------------------------------------------- #
# It is a black-box explicit method: any right-hand side works                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('scheme,family,order', SCHEMES, ids=['RKC1', 'RKC2', 'RKL2'])
def test_rkc_runs_on_a_semilinear_rhs_and_preserves_mass(scheme, family, order):
    """On the semilinear viscous-Burgers problem (a ``SemilinearRHS``), RKC drives
    the *combined* right-hand side as a black box. Both the linear and nonlinear
    halves have zero periodic mean, so the discrete mass is conserved to
    round-off by any consistent explicit step -- a clean smoke that the driver
    works with the structured RHS, not just a bare callable."""
    prob = testing.PROBLEMS['viscousBurgers'](n=64, nu=0.01)
    system = prob.initial()
    m0 = get_reference_state(system).x.sum().item()
    for _ in range(10):
        system = scheme(system, dt=1e-3, f=prob.rhs, s=8).state
    x = get_reference_state(system).x
    assert torch.isfinite(x).all(), f'{family}: non-finite values on viscous Burgers'
    assert abs(x.sum().item() - m0) < 1e-12, (
        f'{family}: mass not conserved on viscous Burgers ({x.sum().item() - m0:.3e})')
