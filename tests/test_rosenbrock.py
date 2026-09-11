"""ROS3P -- 3-stage, order-3, A-stable (not L-stable) Rosenbrock-W (NOTES S3.14,
Phase 7).

The canonical *linearly-implicit* method of Phase 7's "alternative stiff
families": every stage is a single ``gmres`` solve against a **frozen** operator
``W`` (no outer Newton loop, no line search), with the coefficients from Lang &
Verwer, BIT 41(4) (2001) 731-738. This file exercises the driver on the problem
it is built for -- the semi-discrete viscous Burgers -- plus the
Rosenbrock-W-specific behaviours the generic ``scheme`` fixture does not cover:

* order 3 on the canonical (semilinear) problem, for both ``w='jvp'`` and
  ``w='fd'``;
* the ``w='linear'`` option -- **first order** on a genuinely semilinear problem
  (the honest finding), but **order 3** on a problem that is linear in the state
  (``J_N = 0``, the case it is designed for);
* the non-autonomous ``f_t`` term -- ``f_t='fd'`` retains order 3 on the forced
  oscillator, ``f_t='none'`` drops to first order (pinning the requirement);
* A-stability without L-stability: a stiff mode is damped to ``|1 - sqrt(3)|``
  per step, not killed;
* the built-in order-2 embedded estimator, propagated;
* the driver contract: stage 3 reuses stage 2's right-hand-side result, the step
  touches two distinct stage times (three with the ``f_t`` probe), ``priorStep``
  is rejected, the caller state is not mutated, and the adjoint gradient is exact
  on a linear problem.

The generic ``scheme`` fixture already pins order 3 on the oscillator / forced /
kepler problems; this file adds the semilinear-canonical and Rosenbrock-W-specific
tests on top.
"""

import math

import pytest
import torch

from conftest import ORDER_TOLERANCE
from warpSPHIntegrators import (
    ROS3P,
    SemilinearRHS,
    StageResult,
    getIntegrator,
    flatten_integrated,
    get_reference_state,
    unflatten_integrated,
)
from warpSPHIntegrators import testing

#: The canonical semi-discrete viscous-Burgers accuracy/cost setting (NOTES S3.12).
T_BURGERS = 0.4
NU = 0.01
N_BURGERS = 64


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

_REF_CACHE: dict = {}


def _burgers_ref(n: int = N_BURGERS, nu: float = NU, T: float = T_BURGERS) -> torch.Tensor:
    """Fine-``dt`` RK4 reference for viscous Burgers (cached per setting)."""
    key = (n, nu, T)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = testing.viscous_burgers_reference(n=n, nu=nu, dt=2e-3, T=T)
    return _REF_CACHE[key]


def _run_burgers(dt: float, T: float, w: str, n: int = N_BURGERS, nu: float = NU):
    prob = testing.PROBLEMS['viscousBurgers'](n=n, nu=nu)
    system = prob.initial()
    for _ in range(int(round(T / dt))):
        system = ROS3P(system, dt=dt, f=prob.rhs, w=w).state
    return get_reference_state(system).x


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.linalg.norm(x - ref) / torch.linalg.norm(ref))


def _linear_diffusion_problem(n: int = 32, nu: float = 0.01, L: float = 1.0):
    """A **purely linear** ``SemilinearRHS`` (diffusion only, ``N = 0``).

    The frozen operator ``W = L`` is then the *exact* Jacobian (``J_N = 0``), so
    ``w='linear'`` is not a low-order approximation here: ROS3P keeps its full
    order 3. This is the case the ``w='linear'`` option is designed for, and the
    contrast with the semilinear case below is the point.
    """
    h = L / n
    xc, sigma = L / 2, 0.08

    def linear(state, dt, *args, **kwargs):
        u = get_reference_state(state).x
        u_xx = (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / (h * h)
        return testing.ParticleUpdate(dxdt=nu * u_xx,
                                      dudt=torch.zeros_like(u),
                                      dedt=torch.zeros_like(u)), None

    def nonlinear(state, dt, *args, **kwargs):
        u = get_reference_state(state).x
        return testing.ParticleUpdate(dxdt=torch.zeros_like(u),
                                      dudt=torch.zeros_like(u),
                                      dedt=torch.zeros_like(u)), None

    def initial():
        u0 = [math.exp(-((i * h - xc) ** 2) / (2 * sigma ** 2)) for i in range(n)]
        return testing.ParticleSystem(state=testing.ParticleState(
            x=torch.tensor(u0, dtype=torch.float64),
            u=torch.zeros(n, dtype=torch.float64),
            e=torch.zeros(n, dtype=torch.float64),
            m=torch.ones(n, dtype=torch.float64)), t=0.0)

    return testing.Problem(
        name='linearDiffusion',
        description=f'pure linear diffusion (N=0), n={n}, nu={nu}; J_N=0 so W=L is exact',
        rhs=SemilinearRHS(linear=linear, nonlinear=nonlinear),
        initial=initial,
        exact=lambda t: (_ for _ in ()).throw(NotImplementedError),
        energy=None, autonomous=True)


def _stiff_problem(rate: float):
    """Scalar stiff relaxation ``x' = -rate * x`` (``u``, ``e`` identically zero)
    as a plain-callable RHS, for the A-stable-not-L stability test."""

    def rhs(state, dt, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=-rate * s.x,
                                      dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    def initial():
        return testing.ParticleSystem(state=testing.ParticleState(
            x=torch.ones(1, dtype=torch.float64),
            u=torch.zeros(1, dtype=torch.float64),
            e=torch.zeros(1, dtype=torch.float64),
            m=torch.ones(1, dtype=torch.float64)), t=0.0)

    return testing.Problem(
        name='stiffRelaxation',
        description=f'scalar stiff relaxation, rate={rate}',
        rhs=rhs, initial=initial,
        exact=lambda t: (_ for _ in ()).throw(NotImplementedError),
        energy=None, autonomous=True)


def _grad_parity_error(prob, w: str, dt: float = 0.02, n_fd: int = 6, f_t: str = 'fd') -> float:
    """Max relative ``|autograd - FD| / |FD|`` of the step map's output ``x`` w.r.t.
    the input ``x``, over the first ``n_fd`` components. ``FD`` is the central
    finite-difference of the same step; on a linear problem the adjoint is exact
    so this is at round-off, on a semilinear one it carries the frozen-operator
    (``dW/dy`` Hessian) approximation error."""
    base = get_reference_state(prob.initial()).x.clone()
    weights = torch.randn_like(base)

    def run_scalar(base_x):
        system = prob.initial()
        get_reference_state(system).x = base_x.clone()
        result = ROS3P(system, dt=dt, f=prob.rhs, w=w, f_t=f_t)
        return float(get_reference_state(result.state).x @ weights)

    x0 = base.clone().requires_grad_(True)
    system = prob.initial()
    get_reference_state(system).x = x0
    result = ROS3P(system, dt=dt, f=prob.rhs, w=w, f_t=f_t)
    s = get_reference_state(result.state).x @ weights
    g_auto = torch.autograd.grad(s, x0)[0]

    eps = 1e-7
    worst = 0.0
    for i in range(n_fd):
        xp = base.clone(); xp[i] += eps
        xm = base.clone(); xm[i] -= eps
        g_fd = (run_scalar(xp) - run_scalar(xm)) / (2 * eps)
        worst = max(worst, abs(g_auto[i].item() - g_fd) / max(abs(g_fd), 1e-30))
    return worst


# --------------------------------------------------------------------------- #
# Order on the canonical semilinear problem                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('w', ['jvp', 'fd'])
def test_ros3p_achieves_order_3_on_viscous_burgers(w):
    """Order 3, measured on the problem ROS3P is built for, against the fine-``dt``
    RK4 reference -- for both the exact (``'jvp'``) and finite-difference (``'fd'``)
    frozen operators."""
    ref = _burgers_ref()
    # Four halvings: the finest pair (0.01 -> 0.005) is where the order-3
    # asymptotic regime is reached on this (pre-asymptotic at coarser dts).
    dts = [0.04, 0.02, 0.01, 0.005]
    errors = [_rel_l2(_run_burgers(dt, T_BURGERS, w), ref) for dt in dts]
    order = testing.measured_order(errors, dts)
    assert order is not None, f'w={w}: no usable errors: {errors}'
    assert order >= 3 - ORDER_TOLERANCE, (
        f'w={w}: registered order 3, measured {order:.2f}. '
        f'Errors: {[f"{e:.3e}" for e in errors]}')


def test_w_selector_jvp_and_fd_agree():
    """The exact (``'jvp'``) and finite-difference (``'fd'``) frozen operators give
    the same solution to the order-3 accuracy (the FD Jacobian error is below the
    method's own temporal error at these steps)."""
    dt = 0.02
    x_jvp = _run_burgers(dt, T_BURGERS, 'jvp')
    x_fd = _run_burgers(dt, T_BURGERS, 'fd')
    rel = _rel_l2(x_jvp, x_fd)
    assert rel < 1e-3, f"w='jvp' vs w='fd' differ by {rel:.2e} relative L2 (expected < 1e-3)"


# --------------------------------------------------------------------------- #
# The w='linear' option: low-order on semilinear, exact on linear             #
# --------------------------------------------------------------------------- #

def test_w_linear_is_first_order_on_semilinear():
    """On the genuinely *semilinear* viscous-Burgers problem, ``w='linear'`` drops
    the nonlinear Jacobian ``J_N`` from the frozen operator and the asymptotic
    order falls from 3 to ~1. This pins the honest limitation (the option is a
    cheap low-order method, not a stand-in for the full Jacobian)."""
    ref = _burgers_ref()
    dts = [0.04, 0.02, 0.01]
    errors = [_rel_l2(_run_burgers(dt, T_BURGERS, 'linear'), ref) for dt in dts]
    order = testing.measured_order(errors, dts)
    assert order is not None, f'w=linear: no usable errors: {errors}'
    assert order < 2.0, (
        f"w='linear' on viscous Burgers: order {order:.2f}, expected ~1 (NOT 3). "
        f'Errors: {[f"{e:.3e}" for e in errors]}')


@pytest.mark.parametrize('w', ['linear', 'jvp'])
def test_w_linear_is_order_3_on_a_linear_problem(w):
    """On a problem that is **linear in the state** (``J_N = 0``), the frozen
    operator ``W = L`` *is* the exact Jacobian, so ``w='linear'`` keeps the full
    order 3 -- and agrees with the exact ``w='jvp'``. This is the case the option
    is designed for."""
    prob = _linear_diffusion_problem(n=32, nu=0.01)
    T = 0.4
    ref = get_reference_state(testing.run(getIntegrator('RK4'), prob, dt=2e-3, T=T)).x
    dts = [0.04, 0.02, 0.01]
    errors = []
    for dt in dts:
        system = prob.initial()
        for _ in range(int(round(T / dt))):
            system = ROS3P(system, dt=dt, f=prob.rhs, w=w).state
        errors.append(_rel_l2(get_reference_state(system).x, ref))
    order = testing.measured_order(errors, dts)
    assert order is not None, f'w={w} on linear problem: no usable errors: {errors}'
    assert order >= 3 - ORDER_TOLERANCE, (
        f'w={w} on a linear problem: order {order:.2f}, expected 3 (J_N=0). '
        f'Errors: {[f"{e:.3e}" for e in errors]}')


# --------------------------------------------------------------------------- #
# The non-autonomous f_t term                                                 #
# --------------------------------------------------------------------------- #

def test_ft_fd_retains_order_on_forced():
    """On the genuinely non-autonomous forced oscillator (``x'' = cos t``), the
    finite-difference time derivative (``f_t='fd'``) retains the full order 3."""
    prob = testing.PROBLEMS['forced']()
    dts = [0.1, 0.05, 0.025, 0.0125]
    order, errors = testing.convergence(ROS3P, prob, dts, 1.0, w='jvp', f_t='fd')
    assert order is not None, f'f_t="fd": no usable errors: {errors}'
    assert order >= 3 - ORDER_TOLERANCE, f'f_t="fd": order {order:.2f}, expected 3'


def test_ft_none_drops_order_on_forced():
    """Dropping the time-derivative term (``f_t='none'``) on a genuinely
    non-autonomous problem loses order: the forced oscillator falls to ~first
    order. This pins that ``f_t='fd'`` (the default) is required for non-autonomous
    dynamics, not an optional refinement."""
    prob = testing.PROBLEMS['forced']()
    dts = [0.1, 0.05, 0.025, 0.0125]
    order, errors = testing.convergence(ROS3P, prob, dts, 1.0, w='jvp', f_t='none')
    assert order is not None, f'f_t="none": no usable errors: {errors}'
    assert order < 2.0, f'f_t="none" on forced: order {order:.2f}, expected ~1 (NOT 3)'


# --------------------------------------------------------------------------- #
# A-stable, not L-stable                                                      #
# --------------------------------------------------------------------------- #

def test_a_stable_not_l_stable():
    """A stiff mode with ``rate*dt >> 1`` is damped by the stability function
    ``R(-rate*dt) -> R(-inf) = 1 - sqrt(3)`` per step: the amplitude is cut to
    ``|1 - sqrt(3)| ~= 0.732``, *not* to zero. That is the signature of
    A-stability without L-stability -- stiff modes decay geometrically at a
    fixed rate rather than being killed outright."""
    rate, dt = 1000.0, 1.0          # rate * dt = 1000 >> 1
    prob = _stiff_problem(rate)
    system = prob.initial()
    x0 = get_reference_state(system).x.clone()
    result = ROS3P(system, dt=dt, f=prob.rhs, w='jvp')
    amp = float(get_reference_state(result.state).x.abs() / x0.abs())
    expected = abs(1.0 - math.sqrt(3.0))
    assert amp == pytest.approx(expected, abs=0.01), (
        f'A-stable-not-L: amplification {amp:.4f}, expected |1-sqrt(3)| = {expected:.4f}')
    assert amp > 0.5, (
        f'stiff mode killed (amplification {amp:.4f}); A-stable-not-L expects ~{expected:.3f}')


# --------------------------------------------------------------------------- #
# The built-in embedded estimator                                             #
# --------------------------------------------------------------------------- #

def test_embedded_estimator_is_order_2():
    """Propagating the embedded (order-2) solution ``y_hat = y^{n+1} - (y-y_hat)``
    -- where ``y - y_hat = tau (K1 - K2) / 3`` is the driver's error estimate --
    converges at order 2 on the forced oscillator (closed-form reference)."""
    prob = testing.PROBLEMS['forced']()
    T = 1.0
    ex_x, ex_u = 2.0 - math.cos(T), math.sin(T)

    def propagate_embedded(dt):
        system = prob.initial()
        for _ in range(int(round(T / dt))):
            result = ROS3P(system, dt=dt, f=prob.rhs, w='jvp')
            y_hat_flat = flatten_integrated(result.state) - flatten_integrated(result.error)
            system = unflatten_integrated(y_hat_flat, system)
            system.t += dt
        s = get_reference_state(system)
        return math.hypot(s.x[0].item() - ex_x, s.u[0].item() - ex_u)

    dts = [0.1, 0.05, 0.025, 0.0125]
    errors = [propagate_embedded(dt) for dt in dts]
    order = testing.measured_order(errors, dts)
    assert order is not None, f'embedded: no usable errors: {errors}'
    assert order >= 2 - ORDER_TOLERANCE, (
        f'embedded estimator: order {order:.2f}, expected 2. Errors: {[f"{e:.3e}" for e in errors]}')


# --------------------------------------------------------------------------- #
# The driver contract                                                         #
# --------------------------------------------------------------------------- #

def test_stage_three_reuses_stage_two_rhs():
    """ROS3P's stage 3 is evaluated at the *same* (t, z) point as stage 2
    (``a31 = 1, a32 = 0`` and ``c3 = c2 = 1``), so the driver reuses stage 2's
    right-hand-side result: the third ``StageResult`` carries the same ``aux`` and
    ``update`` objects as the second."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    result = ROS3P(system, dt=0.01, f=prob.rhs, w='jvp')
    assert len(result.stages) == 3, f'expected 3 stages, got {len(result.stages)}'
    assert result.stages[1].aux is result.stages[2].aux, (
        "stage 3 should reuse stage 2's aux (same (t, z) point)")
    assert result.stages[1].update is result.stages[2].update, (
        "stage 3 should reuse stage 2's update (same (t, z) point)")


def test_evaluates_f_at_two_distinct_stage_times():
    """A step evaluates the right-hand side at two distinct *stage* times --
    ``(tn, un)`` and ``(tn + dt, un + dt K1)`` -- because stage 3 reuses stage 2's
    state and time (the Krylov ``W``-actions are also evaluated at the step start,
    so they share the first time). The ``f_t='fd'`` probe adds a third time."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)

    def count_stage_times(f_t):
        system = prob.initial()
        times = set()

        def recording_rhs(state, dt_, **kw):
            times.add(round(float(state.t), 10))
            return prob.rhs(state, dt_, **kw)

        ROS3P(system, dt=0.01, f=recording_rhs, w='jvp', f_t=f_t)
        return times

    assert len(count_stage_times('none')) == 2, 'expected 2 stage times with f_t=none'
    assert len(count_stage_times('fd')) == 3, 'expected 3 stage times with f_t=fd (the probe)'


def test_rejects_prior_step_with_a_warning():
    """ROS3P is not stiffly accurate (its last stage is not the step), so it
    refuses first-stage reuse with a warning rather than silently mis-using it."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    prior = StageResult(aux=None, update=None)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        ROS3P(system, dt=0.01, f=prob.rhs, w='jvp', priorStep=prior)


def test_does_not_mutate_the_caller_state():
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    before_x = system.state.x.clone()
    before_t = system.t
    result = ROS3P(system, dt=0.01, f=prob.rhs, w='jvp')
    assert torch.equal(system.state.x, before_x), 'ROS3P mutated the caller field'
    assert system.t == before_t, 'ROS3P mutated the caller time'
    assert get_reference_state(result.state).x is not system.state.x, (
        'the final state aliases the caller buffer instead of owning one')


# --------------------------------------------------------------------------- #
# Adjoint gradient                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('w', ['jvp', 'fd'])
def test_gradient_parity_exact_on_linear_problem(w):
    """On a problem linear in the state no ``dW/dy`` (Hessian) term is dropped, so
    the re-attached adjoint gradient is exact: with the exact (``'jvp'``) frozen
    operator it matches a central finite-difference reference to round-off, and
    with the FD (``'fd'``) operator only to the FD frozen-operator precision
    (``O(eps_fd / dt)``) -- the limit is the operator, not the adjoint."""
    prob = _linear_diffusion_problem(n=16, nu=0.01)
    err = _grad_parity_error(prob, w, n_fd=6)
    tol = 1e-4 if w == 'jvp' else 1e-3
    assert err < tol, (
        f'w={w} on a linear problem: gradient parity error {err:.2e} (bound {tol:g})')


@pytest.mark.parametrize('w', ['jvp', 'fd'])
def test_gradient_parity_fd_limited_on_semilinear(w):
    """On the semilinear viscous-Burgers problem the adjoint holds the frozen
    operator constant (its ``dW/dy`` is a Hessian term, dropped), so the gradient
    carries an ``O(dt)`` per-step error. It stays small (well under 25% relative)
    but is not machine-exact, unlike the linear case above."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    err = _grad_parity_error(prob, w, n_fd=4)
    assert err < 0.25, (
        f'w={w} on viscous Burgers: gradient parity error {err:.2e} (expected O(dt) ~ a few %)')
