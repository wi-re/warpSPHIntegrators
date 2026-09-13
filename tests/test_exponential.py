"""ETD2RK -- 2-stage, order-2, L-stable exponential integrator (NOTES S3.15,
Phase 7).

The exponential family of Phase 7's "alternative stiff families": the semilinear
split ``f = L·y + N`` is integrated with the *linear* part ``L`` carried exactly
through the matrix exponential and the entire ``phi`` functions -- applied
**matrix-free** as ``phi_k(hL)v`` via a Krylov (Arnoldi) approximation, consuming
the Phase 14 ``linear`` accessor and forming no dense matrix -- while the nonlinear
remainder ``N`` is quadratured (two evaluations per step). The scheme is the base,
unsplit ETD2RK of Sarumi (arXiv:2601.06849, eqs. (7)-(8)); the ``phi``-function
definition is Caliari & Ostermann, *Appl. Numer. Math.* 59 (2009) 568-581, eq.
(2.4).

This file exercises the driver on the problem it is built for -- the semi-discrete
viscous Burgers -- plus the exponential-specific behaviours the generic ``scheme``
fixture does not cover:

* order 2 on the canonical (semilinear) problem, against the fine-``dt`` RK4
  reference;
* **exact** integration of a purely linear problem (``N = 0`` gives
  ``y^{n+1} = exp(hL) y^n`` up to the Krylov tolerance, so the error is at the
  Krylov tolerance, far below any ``O(h^2)`` order error);
* L-stability: a stiff linear mode (``rate*h >> 1``) is damped by ``exp(-rate*h)``
  per step, *killed* rather than merely damped (the Rosenbrock analogue,
  A-stable-not-L, damps the same regime to ``|1 - sqrt(3)| ~= 0.732``);
* the driver contract: two stages, two ``N`` evaluations per step, the step
  touches two distinct stage times, ``priorStep`` is rejected, a plain callable
  (no ``linear`` part) is rejected *before the solve*, the caller state is not
  mutated, and ``error is None``;
* the adjoint gradient is exact up to the Krylov tolerance on *both* the linear
  and the semilinear problem (``L`` is constant, so no Hessian term is dropped --
  in contrast to the Rosenbrock frozen-``W``, which is ``O(h)`` on semilinear).

EXPRB32 (NOTES S3.16): Hochbrueck, Ostermann & Schweitzer, *SIAM J. Numer. Anal.*
47 (2009) 786-803, sections 2.3 / 6.5, in the frozen-``Jn`` reformulation with
``F_n = f(tn, yn)`` and ``Jn = D_u f(tn, yn)`` --

* ``U2 = yn + h phi_1(hJn) Fn`` (``+ h^2 phi_2(hJn) f_t`` when ``f_t`` is
  supplied); ``D2 = f(tn+h, U2) - Fn - Jn (U2 - yn)``;
* ``u^{n+1} = U2 + 2 h phi_3(hJn) D2``, with the embedded order-2 estimate
  ``uhat = U2`` (exp-Rosenbrock-Euler), so the estimate is the corrector's
  correction itself.

The frozen operator is the *full* right-hand-side Jacobian (forward-mode AD via
``w='jvp'``, or finite differences via ``w='fd'``), applied matrix-free through
the same Krylov ``phi_k`` builds. The inverse of ETD2RK's contract follows: a
plain callable is *accepted* (there is no ``linear`` part to demand) and a
semilinear split is *not* needed. The generic ``scheme`` fixture already pins
order 3 on the forced / kepler problems (the linear autonomous problems are
skipped: the frozen operator is constant there and the step is the exact
exponential flow); this file adds the semilinear-canonical and
exponential-specific tests on top:

* order 3 on the canonical problem, against the fine-``dt`` RK4 reference;
* **exact** integration of a purely linear problem (the nonlinear remainder
  ``D2`` vanishes, the step is ``exp(hJn)`` applied through the Krylov
  ``phi_1`` build, and the corrector correction is skipped by the zero guard);
* L-stability on a stiff linear mode (killed, not merely damped);
* the driver contract: two stages, the full-right-hand-side evaluation count
  per step (``f_n``, the ``f_t`` probe, ``f(tn+h, U2)``, and the frozen-``Jn``
  application), the Krylov builds counted separately, ``priorStep`` rejected,
  the caller state not mutated, and ``error`` carrying the embedded estimate;
* the embedded estimate is third order locally (the order-2 branch's local
  error), so it shrinks ~8x when ``dt`` halves;
* the adjoint gradient: exact up to the Krylov tolerance on the linear problem
  (``Jn`` is constant) and ``O(dt^2)`` on the semilinear one (every
  frozen-operator occurrence carries a factor of ``dt``) -- better than the
  Rosenbrock frozen-``W`` adjoint, which is ``O(dt)`` on semilinear.
"""

import math

import pytest
import torch

from conftest import ORDER_TOLERANCE
from warpSPHIntegrators import (
    ETD2RK,
    EXPRB32,
    SemilinearRHS,
    StageResult,
    getIntegrator,
    get_reference_state,
)
from warpSPHIntegrators import testing
from warpSPHIntegrators.rhs import resolve

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


def _run_burgers(dt: float, T: float, n: int = N_BURGERS, nu: float = NU,
                 scheme=ETD2RK, **kw):
    prob = testing.PROBLEMS['viscousBurgers'](n=n, nu=nu)
    system = prob.initial()
    for _ in range(int(round(T / dt))):
        system = scheme(system, dt=dt, f=prob.rhs, **kw).state
    return get_reference_state(system).x


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.linalg.norm(x - ref) / torch.linalg.norm(ref))


def _linear_diffusion_problem(n: int = 32, nu: float = 0.01, L: float = 1.0):
    """A **purely linear** ``SemilinearRHS`` (diffusion only, ``N = 0``).

    ETD2RK integrates the linear part *exactly* (``y^{n+1} = exp(hL) y^n`` up to
    the Krylov tolerance), so this is the case where the method is exact rather
    than merely order-limited -- the contrast with the semilinear case is the
    point.
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
        description=f'pure linear diffusion (N=0), n={n}, nu={nu}',
        rhs=SemilinearRHS(linear=linear, nonlinear=nonlinear),
        initial=initial,
        exact=lambda t: (_ for _ in ()).throw(NotImplementedError),
        energy=None, autonomous=True)


def _stiff_linear_problem(rate: float):
    """Scalar stiff relaxation ``x' = -rate * x`` as a **SemilinearRHS** with
    ``N = 0`` (the linear part carries the stiffness), for the L-stability test.

    A SemilinearRHS (not a plain callable) because ETD2RK needs the ``linear``
    part; with ``N = 0`` the step is ``y^{n+1} = exp(-rate*h) y^n`` exactly (up to
    the Krylov tolerance), so the amplification is the exponential ``exp(-rate*h)``.
    """

    def linear(state, dt, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=-rate * s.x,
                                      dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    def nonlinear(state, dt, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=torch.zeros_like(s.x),
                                      dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    def initial():
        return testing.ParticleSystem(state=testing.ParticleState(
            x=torch.ones(1, dtype=torch.float64),
            u=torch.zeros(1, dtype=torch.float64),
            e=torch.zeros(1, dtype=torch.float64),
            m=torch.ones(1, dtype=torch.float64)), t=0.0)

    return testing.Problem(
        name='stiffLinear',
        description=f'scalar stiff relaxation (N=0), rate={rate}',
        rhs=SemilinearRHS(linear=linear, nonlinear=nonlinear),
        initial=initial,
        exact=lambda t: (_ for _ in ()).throw(NotImplementedError),
        energy=None, autonomous=True)


def _grad_parity_error(prob, dt: float = 0.02, n_fd: int = 6, scheme=ETD2RK) -> float:
    """Max relative ``|autograd - FD| / |FD|`` of the step map's output ``x`` w.r.t.
    the input ``x``, over the first ``n_fd`` components. ETD2RK's ``L`` is constant,
    so its re-attached adjoint drops *no* structural term: the error is the
    transpose-``phi`` Krylov tolerance plus the FD reference error, on *both* a
    linear and a semilinear problem. EXPRB32 freezes the full Jacobian ``Jn``; its
    re-attached adjoint is exact on a linear problem (``Jn`` is constant) and
    ``O(dt^2)`` on a semilinear one (every frozen-operator occurrence carries a
    factor of ``dt``), still far below the FD reference error here -- unlike the
    Rosenbrock frozen-``W``, which is ``O(dt)`` on a semilinear one."""
    base = get_reference_state(prob.initial()).x.clone()
    weights = torch.randn_like(base)

    def run_scalar(base_x):
        system = prob.initial()
        get_reference_state(system).x = base_x.clone()
        result = scheme(system, dt=dt, f=prob.rhs)
        return float(get_reference_state(result.state).x @ weights)

    x0 = base.clone().requires_grad_(True)
    system = prob.initial()
    get_reference_state(system).x = x0
    result = scheme(system, dt=dt, f=prob.rhs)
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

def test_etd2rk_achieves_order_2_on_viscous_burgers():
    """Order 2, measured on the problem ETD2RK is built for, against the fine-``dt``
    RK4 reference."""
    ref = _burgers_ref()
    # Four halvings: the finest pair (0.01 -> 0.005) is where the order-2
    # asymptotic regime is reached on this problem.
    dts = [0.04, 0.02, 0.01, 0.005]
    errors = [_rel_l2(_run_burgers(dt, T_BURGERS), ref) for dt in dts]
    order = testing.measured_order(errors, dts)
    assert order is not None, f'no usable errors: {errors}'
    assert order >= 2 - ORDER_TOLERANCE, (
        f'registered order 2, measured {order:.2f}. '
        f'Errors: {[f"{e:.3e}" for e in errors]}')


# --------------------------------------------------------------------------- #
# Exact integration of the linear part (N = 0)                                #
# --------------------------------------------------------------------------- #

def test_exact_on_a_purely_linear_problem():
    """With ``N = 0``, ETD2RK gives ``y^{n+1} = exp(hL) y^n`` up to the Krylov
    tolerance, so a purely linear problem is solved to the Krylov tolerance --
    far below any ``O(h^2)`` order error. This pins that the linear part is
    integrated *exactly*, not just stably."""
    prob = _linear_diffusion_problem(n=32, nu=0.01)
    T = 0.4
    ref = get_reference_state(testing.run(getIntegrator('RK4'), prob, dt=2e-3, T=T)).x
    # A coarse step (h = 0.1, only 4 steps) would give an O(h^2) ~ 1e-2 error for
    # a merely second-order method; the Krylov-tolerance error is ~1e-10, so the
    # method is essentially exact here.
    system = prob.initial()
    for _ in range(int(round(T / 0.1))):
        system = ETD2RK(system, dt=0.1, f=prob.rhs).state
    err = _rel_l2(get_reference_state(system).x, ref)
    assert err < 1e-6, (
        f'linear problem (N=0): rel L2 {err:.2e} with h=0.1, expected ~Krylov tol '
        f'(<< the O(h^2) ~ 1e-2 an order-2 method would give)')


def test_l_stable_kills_a_stiff_linear_mode():
    """A stiff linear mode with ``rate*h >> 1`` is damped by ``exp(-rate*h)`` per
    step: with ``rate*h = 10`` the amplification is ``e^-10 ~= 4.5e-5`` -- the mode
    is *killed*, not merely damped. That is L-stability (the exponential
    signature); the Rosenbrock analogue (A-stable, not L) damps the same regime to
    ``|1 - sqrt(3)| ~= 0.732`` instead. The stiffness is kept in the range the
    ``phi_k`` Taylor series evaluates accurately (see ``exponential._phi_series``).
    """
    rate, dt = 10.0, 1.0          # rate * dt = 10 >> 1
    prob = _stiff_linear_problem(rate)
    system = prob.initial()
    x0 = get_reference_state(system).x.clone()
    result = ETD2RK(system, dt=dt, f=prob.rhs)
    amp = float(get_reference_state(result.state).x.abs() / x0.abs())
    expected = math.exp(-rate * dt)
    assert amp == pytest.approx(expected, rel=1e-3), (
        f'L-stable: amplification {amp:.4e}, expected exp(-rate*dt) = {expected:.4e}')
    assert amp < 1e-3, (
        f'stiff mode only damped (amplification {amp:.3e}); L-stability expects '
        f'exp(-rate*dt) = {expected:.3e} ~= 0 (killed), not the Rosenbrock ~0.732')


# --------------------------------------------------------------------------- #
# The driver contract                                                         #
# --------------------------------------------------------------------------- #

def test_two_stages_two_nonlinear_evals_per_step():
    """A step is two stages (the exp-Euler predictor and the corrector) and costs
    exactly two nonlinear (``N``) evaluations -- the two stage ``N`` values. The
    three matrix-function actions are cheap ``L``-matvec Krylov builds, reported
    in the ``gmres_iterations`` field, and are *not* counted as full right-hand-
    side evaluations."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    result = ETD2RK(system, dt=0.01, f=prob.rhs)
    assert len(result.stages) == 2, f'expected 2 stages, got {len(result.stages)}'
    rhs = sum(d.rhs_evaluations for d in result.solver_diagnostics)
    krylov = sum(d.gmres_iterations for d in result.solver_diagnostics)
    assert rhs == 2, f'expected 2 N evaluations per step, got {rhs}'
    assert krylov > 0, 'expected the linear Krylov builds to be counted'
    assert all(d.rhs_evaluations == 1 for d in result.solver_diagnostics), (
        'each stage should account for exactly one N evaluation')
    assert result.error is None, 'ETD2RK has no embedded estimator (error must be None)'


def test_evaluates_f_at_two_distinct_stage_times():
    """A step evaluates the nonlinear part at two distinct *stage* times --
    ``(tn, yn)`` for the predictor's ``N_n`` and ``(tn + dt, w)`` for the
    corrector's ``N_w`` -- because the two stages are at different (t, z) points.
    (The ``linear`` part is kept intact and recorded separately: the Krylov
    ``L``-matvecs are cheap operator applications, not stage ``N`` evaluations.)"""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    parts = resolve(prob.rhs, scheme_name='test', need_linear=True)
    times = set()

    def recording_nonlinear(state, dt_, **kw):
        times.add(round(float(state.t), 10))
        return parts.nonlinear(state, dt_, **kw)

    ETD2RK(system, dt=0.01,
           f=SemilinearRHS(linear=parts.linear, nonlinear=recording_nonlinear))
    assert len(times) == 2, f'expected 2 stage times, got {sorted(times)}'


def test_rejects_prior_step_with_a_warning():
    """ETD2RK is not stiffly accurate (its last stage is not the step), so it
    refuses first-stage reuse with a warning rather than silently mis-using it."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    prior = StageResult(aux=None, update=None)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        ETD2RK(system, dt=0.01, f=prob.rhs, priorStep=prior)


def test_rejects_a_plain_callable_before_the_solve():
    """ETD2RK integrates the *linear* part exactly, so it needs the ``linear``
    accessor of the semilinear split. A plain callable carries only the combined
    ``f`` (no ``linear`` part) and is rejected *before the solve* with a
    capability ``TypeError``."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()

    def plain_rhs(state, dt, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=-s.x,
                                      dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    with pytest.raises(TypeError, match='needs a "linear" accessor'):
        ETD2RK(system, dt=0.01, f=plain_rhs)


def test_does_not_mutate_the_caller_state():
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    before_x = system.state.x.clone()
    before_t = system.t
    result = ETD2RK(system, dt=0.01, f=prob.rhs)
    assert torch.equal(system.state.x, before_x), 'ETD2RK mutated the caller field'
    assert system.t == before_t, 'ETD2RK mutated the caller time'
    assert get_reference_state(result.state).x is not system.state.x, (
        'the final state aliases the caller buffer instead of owning one')


# --------------------------------------------------------------------------- #
# Adjoint gradient                                                            #
# --------------------------------------------------------------------------- #

def test_gradient_parity_exact_on_linear_problem():
    """On a problem linear in the state, ``L`` is the whole dynamics and the
    re-attached adjoint (constant ``L``, transposed ``phi``-actions) is exact up
    to the Krylov tolerance: it matches a central finite-difference reference to
    well under 1e-4."""
    prob = _linear_diffusion_problem(n=16, nu=0.01)
    err = _grad_parity_error(prob, n_fd=6)
    assert err < 1e-4, (
        f'linear problem: gradient parity error {err:.2e} (bound 1e-4)')


def test_gradient_parity_exact_on_semilinear_problem():
    """On the semilinear viscous-Burgers problem the adjoint is *still* exact up
    to the Krylov tolerance (the nonlinearity enters only through the two ``N``
    evaluations, whose VJPs are ordinary autograd VJPs, and ``L`` is constant so
    no Hessian term is dropped). This is markedly better than the Rosenbrock
    frozen-``W`` adjoint, which is ``O(dt)`` on a semilinear problem -- the
    difference is the point of the exponential integrator's gradient."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    err = _grad_parity_error(prob, n_fd=4)
    assert err < 1e-4, (
        f'viscous Burgers: gradient parity error {err:.2e} (bound 1e-4; the '
        f'Rosenbrock frozen-W is O(dt) ~ a few % here, not this small)')


# --------------------------------------------------------------------------- #
# EXPRB32 (NOTES S3.16)                                                       #
# --------------------------------------------------------------------------- #

def test_exprb32_achieves_order_3_on_viscous_burgers():
    """Order 3, measured on the canonical semilinear problem against the
    fine-``dt`` RK4 reference -- the exponential family's cost gate is assessed
    at this error level (scripts/exponential_benchmark.py, NOTES S3.16)."""
    ref = _burgers_ref()
    # Four halvings: the finest pair (0.01 -> 0.005) is where the order-3
    # asymptotic regime is reached on this problem.
    dts = [0.04, 0.02, 0.01, 0.005]
    errors = [_rel_l2(_run_burgers(dt, T_BURGERS, scheme=EXPRB32), ref) for dt in dts]
    order = testing.measured_order(errors, dts)
    assert order is not None, f'no usable errors: {errors}'
    assert order >= 3 - ORDER_TOLERANCE, (
        f'registered order 3, measured {order:.2f}. '
        f'Errors: {[f"{e:.3e}" for e in errors]}')


def test_exprb32_exact_on_a_purely_linear_problem():
    """On a linear right-hand side the frozen ``Jn`` is constant, the nonlinear
    remainder ``D2 = f - Fn - Jn (u - un)`` vanishes, and the step reduces to
    ``exp(hJn) un`` applied through the Krylov ``phi_1`` build (the corrector
    correction is skipped by the zero guard). The error is at the Krylov
    tolerance -- far below any ``O(h^3)`` order error."""
    prob = _linear_diffusion_problem(n=32, nu=0.01)
    T = 0.4
    ref = get_reference_state(testing.run(getIntegrator('RK4'), prob, dt=2e-3, T=T)).x
    # A coarse step (h = 0.1, only 4 steps) would give an O(h^3) ~ 1e-3 error
    # for a merely third-order method; the Krylov-tolerance error is ~1e-10.
    system = prob.initial()
    for _ in range(int(round(T / 0.1))):
        system = EXPRB32(system, dt=0.1, f=prob.rhs).state
    err = _rel_l2(get_reference_state(system).x, ref)
    assert err < 1e-6, (
        f'linear problem: rel L2 {err:.2e} with h=0.1, expected ~Krylov tol '
        f'(<< the O(h^3) ~ 1e-3 an order-3 method would give)')


def test_exprb32_l_stable_kills_a_stiff_linear_mode():
    """A stiff linear mode with ``rate*h >> 1`` is damped by ``exp(-rate*h)`` per
    step: with ``rate*h = 10`` the amplification is ``e^-10 ~= 4.5e-5`` -- the
    mode is *killed*, not merely damped. The frozen ``Jn`` is the exact constant
    Jacobian here, so the step is the scalar exponential flow to the Krylov
    tolerance."""
    rate, dt = 10.0, 1.0          # rate * dt = 10 >> 1
    prob = _stiff_linear_problem(rate)
    system = prob.initial()
    x0 = get_reference_state(system).x.clone()
    result = EXPRB32(system, dt=dt, f=prob.rhs)
    amp = float(get_reference_state(result.state).x.abs() / x0.abs())
    expected = math.exp(-rate * dt)
    assert amp == pytest.approx(expected, rel=1e-3), (
        f'L-stable: amplification {amp:.4e}, expected exp(-rate*dt) = {expected:.4e}')
    assert amp < 1e-3, (
        f'stiff mode only damped (amplification {amp:.3e}); L-stability expects '
        f'exp(-rate*dt) = {expected:.3e} ~= 0 (killed), not the Rosenbrock ~0.732')


def test_exprb32_runs_on_a_plain_callable():
    """The inverse of ETD2RK's contract: EXPRB32 freezes the *full* Jacobian, so
    there is no semilinear split to demand. A plain callable (no ``linear``
    part) runs as-is -- here integrated to the L-stable scalar exponential."""
    rate = 10.0

    def plain_rhs(state, dt, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=-rate * s.x,
                                      dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    system = _stiff_linear_problem(rate).initial()
    x0 = get_reference_state(system).x.clone()
    result = EXPRB32(system, dt=1.0, f=plain_rhs)
    amp = float(get_reference_state(result.state).x.abs() / x0.abs())
    assert amp == pytest.approx(math.exp(-rate), rel=1e-3), (
        f'plain callable: amplification {amp:.4e}, expected '
        f'exp(-rate) = {math.exp(-rate):.4e}')


def test_exprb32_driver_contract():
    """A step is two stages and costs full right-hand-side evaluations only:
    with the default ``f_t='fd'`` that is ``f_n``, the ``f_t`` probe,
    ``f(tn + h, U2)``, and the frozen-``Jn`` application ``Jn (U2 - un)`` --
    one per stage 1 and two per stage 2. The ``phi_k(hJn)`` Krylov builds are
    counted separately (``gmres_iterations``). And, unlike ETD2RK, the step
    carries the embedded estimate (the corrector correction), so ``error`` is
    not None."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    result = EXPRB32(system, dt=0.01, f=prob.rhs)
    assert len(result.stages) == 2, f'expected 2 stages, got {len(result.stages)}'
    rhs = [d.rhs_evaluations for d in result.solver_diagnostics]
    assert rhs == [2, 2], f'expected per-stage [2, 2] with f_t="fd", got {rhs}'
    krylov = sum(d.gmres_iterations for d in result.solver_diagnostics)
    assert krylov > 0, 'expected the phi Krylov builds to be counted'
    assert result.error is not None, (
        'EXPRB32 carries the embedded estimate (the corrector correction); '
        'error must not be None')

    system = prob.initial()
    result = EXPRB32(system, dt=0.01, f=prob.rhs, f_t='none')
    rhs = [d.rhs_evaluations for d in result.solver_diagnostics]
    assert rhs == [1, 2], (
        f'expected per-stage [1, 2] with f_t="none" (no probe), got {rhs}')


def test_exprb32_embedded_estimate_is_third_order_locally():
    """The embedded estimate is the corrector correction ``u^{n+1} - U2``, the
    local error of the embedded order-2 exp-Rosenbrock-Euler branch: it is
    ``O(h^3)``, so halving ``dt`` shrinks it by ~8 (a second-order local error
    would give 4)."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)

    def estimate_norm(dt):
        system = prob.initial()
        result = EXPRB32(system, dt=dt, f=prob.rhs)
        return float(torch.linalg.norm(get_reference_state(result.error).x))

    coarse, fine = estimate_norm(0.02), estimate_norm(0.01)
    ratio = coarse / fine
    assert 5.0 < ratio < 12.0, (
        f'embedded estimate shrinks {ratio:.2f}x under dt -> dt/2; the order-2 '
        f'local error expects ~8 (a 4x ratio would mean order 2, ~27x order 4)')


def test_exprb32_rejects_prior_step_with_a_warning():
    """EXPRB32 is not stiffly accurate (its last stage is not the step), so it
    refuses first-stage reuse with a warning rather than silently mis-using it."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    prior = StageResult(aux=None, update=None)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        EXPRB32(system, dt=0.01, f=prob.rhs, priorStep=prior)


def test_exprb32_does_not_mutate_the_caller_state():
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    system = prob.initial()
    before_x = system.state.x.clone()
    before_t = system.t
    result = EXPRB32(system, dt=0.01, f=prob.rhs)
    assert torch.equal(system.state.x, before_x), 'EXPRB32 mutated the caller field'
    assert system.t == before_t, 'EXPRB32 mutated the caller time'
    assert get_reference_state(result.state).x is not system.state.x, (
        'the final state aliases the caller buffer instead of owning one')


def test_exprb32_gradient_parity_on_linear_problem():
    """On a problem linear in the state the frozen ``Jn`` is the whole dynamics
    and is constant, so the re-attached adjoint (transposed frozen operator,
    transposed ``phi`` actions) is exact up to the Krylov tolerance: it matches
    a central finite-difference reference to well under 1e-4."""
    prob = _linear_diffusion_problem(n=16, nu=0.01)
    err = _grad_parity_error(prob, scheme=EXPRB32, n_fd=6)
    assert err < 1e-4, (
        f'linear problem: gradient parity error {err:.2e} (bound 1e-4)')


def test_exprb32_gradient_parity_on_semilinear_problem():
    """On the semilinear viscous-Burgers problem the frozen-``Jn`` adjoint is
    ``O(dt^2)``: every occurrence of the frozen operator in the step map
    (``Jn (U2 - un)`` inside ``D2``, and the ``phi_k(hJn)`` actions, whose inputs
    are themselves ``O(dt)``) carries a factor of ``dt``, so the error in the
    dropped ``dJn/dun`` terms is ``O(dt^2) .||dJ/du||`` -- one power of ``dt``
    better than the Rosenbrock frozen-``W`` adjoint (``O(dt)`` on semilinear).
    At ``dt = 0.02`` on this problem that is at the finite-difference floor, so
    the parity check passes by a wide margin."""
    prob = testing.PROBLEMS['viscousBurgers'](n=16, nu=0.01)
    err = _grad_parity_error(prob, scheme=EXPRB32, n_fd=4)
    assert err < 1e-4, (
        f'viscous Burgers: gradient parity error {err:.2e} (bound 1e-4; the '
        f'Rosenbrock frozen-W is O(dt) ~ a few % here, not this small)')
