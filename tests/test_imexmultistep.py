"""SBDF2/3 and CNAB2: IMEX multistep with BDF / trapezoidal backbones (Phase 12).

The schemes are in ``warpSPHIntegrators.imexmultistep``: a BDFp (SBDF2/3) or
trapezoidal (CNAB2) implicit backbone for the stiff part, explicit endpoint
extrapolation for the smooth part, the BDF derivative weight scaling the whole
endpoint derivative. Pinned here:

* the Lagrange extrapolation weights (exact for polynomials of degree < p);
* the measured orders on the split viscous-Burgers problem (threaded history);
* the plain-callable pure-implicit limits (bit-exact against BDF2/BDF3,
  solver-tolerance-close against the trapezoidal rule) and the CNAB2
  pure-explicit limit (bit-exact against AB2);
* the pure-explicit-limit stability: zero-stability, the exact negative-real-
  axis boundaries (4/3, 20/21, 1), and the weak (O(y^4) per-step growth)
  imaginary-axis behaviour -- the practical restriction on the explicit half;
* the driver contract: cold start, history snapshots, dt change, priorStep
  refusal, and the split-evaluation diagnostics count.
"""

import math

import numpy as np
import pytest
import torch

from warpSPHIntegrators import getIntegrator, testing, IMEXRHS, resolve
from warpSPHIntegrators.history import StepHistory
from warpSPHIntegrators.imexmultistep import _extrapolation_weights

# Exact (autograd) Jacobian, tight tolerance: the order runs must not be
# limited by the linear solver (same rationale as tests/test_bdf.py).
_ORDER_SOLVER_OPTS = {'matvec': 'jvp', 'tol': 1e-12, 'newton_tol': 1e-12}


def _run(scheme, problem, dt, T, rhs, solver_opts=None):
    system = problem.initial()
    history = StepHistory(maxlen=max(1, scheme.steps))
    kwargs = {'solver_opts': solver_opts} if solver_opts is not None else {}
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=rhs, history=history, **kwargs)
        system, history = result.state, result.history
    return system


def _burgers_split(n=64, length=1.0, nu=0.01):
    """The canonical split: diffusion L (stiff) implicit, convection N explicit."""
    problem = testing.viscous_burgers_problem(n=n, L=length, nu=nu)
    parts = resolve(problem.rhs, scheme_name='test')
    return problem, IMEXRHS(explicit=parts.nonlinear, implicit=parts.linear)


def _burgers_reference(dt_ref=2e-3, T=0.4, n=64, length=1.0, nu=0.01):
    problem, imex_rhs = _burgers_split(n, length, nu)
    scheme = getIntegrator('RK4')
    return testing.get_reference_state(_run(scheme, problem, dt_ref, T, imex_rhs)).x


# --------------------------------------------------------------------------- #
# Extrapolation weights                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('order', [2, 3])
def test_extrapolation_weights_are_the_lagrange_endpoint_weights(order):
    """The weights are exact for every polynomial of degree < p -- which
    characterizes them uniquely (p weights, p conditions) -- and the degree-p
    error is O(h^p) with a constant leading coefficient."""
    weights = _extrapolation_weights(order)
    assert len(weights) == order
    t, h = 1.0, 0.1
    points = [t - j * h for j in range(order)]  # newest first
    for degree in range(order):
        extrapolated = sum(w * point ** degree for w, point in zip(weights, points))
        assert extrapolated == pytest.approx((t + h) ** degree, abs=1e-12), \
            f'order {order}: extrapolation not exact for x^{degree}'
    # Degree p: the error must be O(h^p) with a constant leading coefficient.
    ratio = []
    for h in (0.1, 0.01):
        points = [t - j * h for j in range(order)]
        extrapolated = sum(w * point ** order for w, point in zip(weights, points))
        ratio.append((extrapolated - (t + h) ** order) / h ** order)
    assert ratio[0] != 0.0
    assert ratio[1] == pytest.approx(ratio[0], rel=1e-4), \
        f'order {order}: degree-{order} error is not O(h^{order})'


def test_extrapolation_weights_reject_unsupported_orders():
    with pytest.raises(ValueError):
        _extrapolation_weights(1)
    with pytest.raises(ValueError):
        _extrapolation_weights(4)


# --------------------------------------------------------------------------- #
# Measured orders                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(('name', 'expected_order', 'dts'), [
    ('SBDF2', 2, (0.004, 0.002, 0.001)),
    ('SBDF3', 3, (0.002, 0.001, 0.0005)),
    ('CNAB2', 2, (0.004, 0.002, 0.001)),
])
def test_imex_multistep_reach_claimed_order_on_split_burgers(name, expected_order, dts):
    """Order on the canonical split problem (diffusion implicit, convection
    explicit), with history threaded and the exact-Jacobian solver. The dt grid
    sits inside the explicit half's practical stability range (see the
    stability tests below): at the coarsest dt the convective |z| is 0.256
    (SBDF2/CNAB2) and 0.128 (SBDF3)."""
    scheme = getIntegrator(name)
    problem, imex_rhs = _burgers_split()
    ref = _burgers_reference()
    errors = []
    for dt in dts:
        state = testing.get_reference_state(
            _run(scheme, problem, dt, T=0.4, rhs=imex_rhs, solver_opts=_ORDER_SOLVER_OPTS))
        errors.append(np.linalg.norm(state.x.numpy() - ref.numpy())
                      / np.linalg.norm(ref.numpy()))
    order = testing.measured_order(errors, list(dts))
    assert order == pytest.approx(expected_order, abs=0.15), \
        f'{name}: measured {order:.2f}, errors {[f"{e:.3e}" for e in errors]}'


@pytest.mark.parametrize(('name', 'expected_order'), [
    ('SBDF2', 2), ('SBDF3', 3), ('CNAB2', 2),
])
def test_plain_callable_limits_reach_claimed_order(name, expected_order):
    """An ordinary RHS (no split) runs the pure-implicit backbone on a
    NON-AUTONOMOUS problem, pinning the grid-time handling through the driver
    (the same regression class NOTES 2.2/2.3 caught in the RK schemes)."""
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS['forced']()
    dts = testing.default_step_sizes(0.1, 5)
    errors = []
    for dt in dts:
        state = _run(scheme, problem, dt, T=2.0, rhs=problem.rhs,
                     solver_opts=_ORDER_SOLVER_OPTS)
        exact_x, exact_u = problem.exact(2.0)
        ref = testing.get_reference_state(state)
        errors.append(sum(abs(a - b) for a, b in zip(ref.x.tolist(), exact_x)) +
                      sum(abs(a - b) for a, b in zip(ref.u.tolist(), exact_u)))
    assert testing.measured_order(errors, dts) == pytest.approx(expected_order, abs=0.15)


# --------------------------------------------------------------------------- #
# Pure limits                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(('name', 'backbone'), [
    ('SBDF2', 'BDF2'), ('SBDF3', 'BDF3'),
])
def test_sbdf_pure_implicit_limits_are_bit_exact_against_the_backbone(name, backbone):
    """No split: the driver must reduce to the BDF family bit-for-bit (same
    formula, same solve path, same initial guess)."""
    problem = testing.PROBLEMS['oscillator']()
    a = testing.get_reference_state(
        _run(getIntegrator(name), problem, 0.05, T=0.5, rhs=problem.rhs)).x
    b = testing.get_reference_state(
        _run(getIntegrator(backbone), problem, 0.05, T=0.5, rhs=problem.rhs)).x
    assert torch.equal(a, b), f'{name} pure-implicit limit differs from {backbone}'


def test_cnab2_pure_implicit_limit_is_the_trapezoidal_rule():
    """Same fixed-point equation, different solve starts: the two converge to
    the tolerance, not bit-for-bit (measured rel diff ~1e-5 at dt=0.02)."""
    problem = testing.PROBLEMS['oscillator']()
    a = testing.get_reference_state(
        _run(getIntegrator('CNAB2'), problem, 0.02, T=0.4, rhs=problem.rhs)).x
    b = testing.get_reference_state(
        _run(getIntegrator('Trapezoidal (Crank-Nicolson)'), problem, 0.02, T=0.4,
             rhs=problem.rhs)).x
    rel = float((a - b).norm() / b.norm())
    assert rel < 1e-3, f'CNAB2 vs trapezoidal: rel diff {rel:.3e}'


def test_cnab2_pure_explicit_limit_is_bit_exact_against_ab2():
    """Zero implicit part: the CNAB2 step is exactly the AB2 increment (no
    solve on the explicit side), so the trajectories are bit-identical."""
    problem, imex_rhs = _burgers_split()
    parts = resolve(problem.rhs, scheme_name='zero')
    zero_state = problem.initial()
    zero_state_ref = testing.get_reference_state(zero_state)
    zero_state_ref.x = torch.zeros_like(zero_state_ref.x)
    zero_update = parts.linear(zero_state, 0.01)
    if isinstance(zero_update, tuple):
        zero_update = zero_update[0]
    explicit_only = IMEXRHS(explicit=problem.rhs,
                            implicit=lambda state, dt, *a, **k: (zero_update, None))
    a = testing.get_reference_state(
        _run(getIntegrator('CNAB2'), problem, 0.004, T=0.2, rhs=explicit_only)).x
    b = testing.get_reference_state(
        _run(getIntegrator('Adams-Bashforth 2'), problem, 0.004, T=0.2,
             rhs=problem.rhs)).x
    assert torch.equal(a, b), 'CNAB2 pure-explicit limit differs from AB2'


# --------------------------------------------------------------------------- #
# Pure-explicit-limit stability                                               #
# --------------------------------------------------------------------------- #

def _characteristic(name, z):
    """Characteristic polynomial of the pure-explicit limit (the implicit part
    removed): the amplification polynomials the BDF state combination +
    extrapolation weights define, xi^p - A xi^{p-1} - ... = 0."""
    if name == 'SBDF2':
        return [1.0, -(4 / 3) * (1 + z), (1 / 3) * (1 + 2 * z)]
    if name == 'SBDF3':
        return [1.0, -(18 / 11) * (1 + z), (9 / 11) * (1 + 2 * z),
                -(2 / 11) * (1 + 3 * z)]
    if name in ('CNAB2', 'Adams-Bashforth 2'):
        return [1.0, -(1 + 1.5 * z), 0.5 * z]
    raise KeyError(name)


def _max_root(name, z):
    roots = np.roots(_characteristic(name, z))
    return max(abs(root) for root in roots)


@pytest.mark.parametrize('name', ['SBDF2', 'SBDF3', 'CNAB2'])
def test_pure_explicit_limits_are_zero_stable(name):
    """Dahlquist root condition at z = 0: all roots in the closed unit disk,
    the root at 1 simple (exactly one on the unit circle)."""
    roots = np.roots(_characteristic(name, 0.0))
    assert max(abs(root) for root in roots) <= 1.0 + 1e-9
    on_circle = sum(1 for root in roots if abs(abs(root) - 1.0) < 1e-9)
    assert on_circle == 1, f'{name}: {on_circle} roots on the unit circle'


@pytest.mark.parametrize('name, boundary', [
    ('SBDF2', 4 / 3),      # exact: a root lands on -1 at z = -4/3
    ('SBDF3', 20 / 21),    # exact: measured 0.952381, the rational 20/21
    ('CNAB2', 1.0),        # exact: a root lands on -1 at z = -1 (AB2's)
])
def test_pure_explicit_limits_have_the_pinned_negative_real_axis_boundary(name, boundary):
    assert _max_root(name, -0.9 * boundary) <= 1.0 + 1e-9, \
        f'{name}: should be stable inside its {boundary} interval'
    assert _max_root(name, -1.1 * boundary) > 1.0, \
        f'{name}: should be unstable just outside its {boundary} interval'


@pytest.mark.parametrize('name', ['SBDF2', 'SBDF3', 'CNAB2'])
def test_pure_explicit_limits_do_not_cover_the_imaginary_axis(name):
    """Unit oscillatory steps are unstable for every one of them: the
    imaginary axis is not a stability direction at O(1) size. (The region
    crosses the axis only in a whisker with O(y^4) per-step growth, pinned
    below -- the practical restriction is dt*max|lambda_E| <~ O(0.5).)"""
    assert _max_root(name, 1.0j) > 1.0 + 0.05
    # And the growth is weak, not explosive: at z = i*0.01 the amplification
    # exceeds 1 by far less than 1e-3.
    assert _max_root(name, 0.01j) < 1.0 + 1e-3


def test_pure_explicit_limit_boundaries_are_measured_on_the_burgers_convective_spectrum():
    """End-to-end confirmation on the actual problem: with the implicit part
    zeroed, the run at dt = 0.008 (convective |z| <= 0.512) stays bounded for
    400 steps (peak |u| ~ 0.98) while dt = 0.01 (|z| <= 0.64) runs to NaN.
    Pins the practical restriction the stability lobes imply on this grid."""
    problem = _burgers_split()[0]
    parts = resolve(problem.rhs, scheme_name='zero2')
    zero_state = problem.initial()
    zero_state_ref = testing.get_reference_state(zero_state)
    zero_state_ref.x = torch.zeros_like(zero_state_ref.x)
    zero_update = parts.linear(zero_state, 0.008)
    if isinstance(zero_update, tuple):
        zero_update = zero_update[0]
    explicit_only = IMEXRHS(explicit=problem.rhs,
                            implicit=lambda state, dt, *a, **k: (zero_update, None))
    scheme = getIntegrator('SBDF2')

    def peak_u(dt, steps):
        """Running max of |u|; returns the first non-finite value if one appears."""
        system = problem.initial()
        history = StepHistory(maxlen=2)
        peak = 0.0
        for _ in range(steps):
            result = scheme(system, dt=dt, f=explicit_only, history=history)
            system, history = result.state, result.history
            value = float(testing.get_reference_state(system).x.abs().max())
            if not math.isfinite(value):
                return value
            peak = max(peak, value)
        return peak

    assert peak_u(0.008, 400) < 1.5
    assert not math.isfinite(peak_u(0.01, 400))


# --------------------------------------------------------------------------- #
# Driver contract                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ['SBDF2', 'SBDF3', 'CNAB2'])
def test_cold_start_bootstraps_from_dormand_prince_and_records_snapshots(name):
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS['oscillator']()
    result = scheme(problem.initial(), dt=0.1, f=problem.rhs)
    assert result.history is not None
    assert result.history.latest.state is not None
    assert result.state.t == pytest.approx(0.1)


@pytest.mark.parametrize('name', ['SBDF2', 'SBDF3', 'CNAB2'])
def test_history_entries_carry_state_snapshots(name):
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS['oscillator']()
    system = problem.initial()
    history = StepHistory(maxlen=max(1, scheme.steps))
    for _ in range(4):
        result = scheme(system, dt=0.05, f=problem.rhs, history=history)
        system, history = result.state, result.history
    assert len(history) == max(1, scheme.steps)
    for entry in history.entries:
        assert entry.state is not None


def test_dt_change_resets_history_and_stays_correct():
    """A dt change mid-run invalidates the state-snapshot history and the
    scheme re-bootstraps from Dormand-Prince: no stale-snapshot mixing, still
    near the analytic solution (the BDF family's contract, tests/test_bdf.py)."""
    scheme = getIntegrator('SBDF3')
    problem = testing.PROBLEMS['oscillator']()
    system = problem.initial()
    history = StepHistory(maxlen=2)
    for i in range(8):
        dt = 0.1 if i < 4 else 0.05
        result = scheme(system, dt=dt, f=problem.rhs, history=history,
                        solver_opts=_ORDER_SOLVER_OPTS)
        system, history = result.state, result.history
    final_x = float(testing.get_reference_state(system).x[0])
    exact_x, _ = problem.exact(4 * 0.1 + 4 * 0.05)
    assert abs(final_x - exact_x[0]) < 0.1


@pytest.mark.parametrize('name', ['SBDF2', 'SBDF3', 'CNAB2'])
def test_prior_step_is_refused(name):
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS['oscillator']()
    first = scheme(problem.initial(), dt=0.1, f=problem.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        scheme(problem.initial(), dt=0.1, f=problem.rhs, priorStep=first.stages[-1])


def test_step_diagnostics_count_the_split_evaluations():
    """The diagnostics rhs_evaluations include the non-solve evaluations of
    the step (SBDF2: two explicit-part evaluations + the endpoint
    evaluation), on top of the solve's map evaluations."""
    scheme = getIntegrator('SBDF2')
    problem, imex_rhs = _burgers_split()
    system = problem.initial()
    history = StepHistory(maxlen=1)
    result = None
    for _ in range(3):
        result = scheme(system, dt=0.004, f=imex_rhs, history=history,
                        solver_opts=_ORDER_SOLVER_OPTS)
        system, history = result.state, result.history
    diag = result.solver_diagnostics
    assert diag is not None
    assert diag.termination in ('tolerance', 'stagnation')
    assert diag.rhs_evaluations >= 4, \
        f'expected >= 4 rhs evaluations (2 explicit + 1 endpoint + >= 1 solve), got {diag.rhs_evaluations}'
