import math

import pytest

from warpSPHIntegrators import getIntegrator, testing
from warpSPHIntegrators.history import StepHistory
from warpSPHIntegrators.bdf import getBDFCoefficients
from warpSPHIntegrators.stability import bdf_characteristic_roots, bdf_is_stable


# The order runs solve each BDF step with the exact (autograd) Jacobian and a tight
# tolerance: with the default finite-difference matvec the JFNK residual bottoms out
# at ~1e-7, which is the same size as BDF4/5's truncation error at the finest step
# sizes below and would corrupt the measured order. The order test measures the
# method, so the linear solver must not be the limiting factor.
_ORDER_SOLVER_OPTS = {'matvec': 'jvp', 'tol': 1e-12, 'newton_tol': 1e-12}


def _run(scheme, problem, dt, T, solver_opts=None):
    system = problem.initial()
    history = StepHistory(maxlen=max(1, scheme.steps))
    kwargs = {'solver_opts': solver_opts} if solver_opts is not None else {}
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history, **kwargs)
        system, history = result.state, result.history
    return system


@pytest.mark.parametrize(('name', 'expected_order'), [
    ('BDF1', 1), ('BDF2', 2), ('BDF3', 3), ('BDF4', 4), ('BDF5', 5),
])
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'damped', 'kepler'])
def test_bdf_reaches_its_claimed_order_with_threaded_history(name, expected_order, problem_name):
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS[problem_name]()
    dts = testing.default_step_sizes(0.1, 5)
    errors = []
    for dt in dts:
        state = _run(scheme, problem, dt, T=2.0, solver_opts=_ORDER_SOLVER_OPTS)
        exact_x, exact_u = problem.exact(2.0)
        ref = testing.get_reference_state(state)
        errors.append(sum(abs(a - b) for a, b in zip(ref.x.tolist(), exact_x)) +
                      sum(abs(a - b) for a, b in zip(ref.u.tolist(), exact_u)))
    assert testing.measured_order(errors, dts) == pytest.approx(expected_order, abs=0.15)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_bdf_coefficients_are_consistent(order):
    state_weights, derivative_weight = getBDFCoefficients(order)
    assert len(state_weights) == order
    assert sum(state_weights) == pytest.approx(1.0)
    assert sum((index + 1) * weight for index, weight in enumerate(state_weights)) == pytest.approx(derivative_weight)


@pytest.mark.parametrize('order', [4, 5])
def test_bdf_is_zero_stable(order):
    """Dahlquist's root condition at z = 0: all characteristic roots inside the closed
    unit disk, and the root at 1 simple (exactly one root on the unit circle)."""
    roots = bdf_characteristic_roots(order, 0.0)
    assert max(abs(root) for root in roots) <= 1.0 + 1e-9
    on_circle = sum(1 for root in roots if abs(abs(root) - 1.0) < 1e-9)
    assert on_circle == 1, f'BDF{order}: {on_circle} roots on the unit circle, expected exactly one'


@pytest.mark.parametrize('order', [4, 5])
@pytest.mark.parametrize('z', [-10.0, -100.0, -1000.0])
def test_bdf4_5_retain_the_negative_real_axis(order, z):
    """The A(alpha) cones of BDF4/5 both contain the entire negative real axis, so
    purely real stiff modes stay damped for any step size."""
    assert bdf_is_stable(order, z), f'BDF{order} rejected z={z}'


@pytest.mark.parametrize('order', [3, 4, 5])
def test_bdf3_5_are_not_a_stable(order):
    """All three lose a neighbourhood of the imaginary axis: the point one unit above
    the origin, barely off the real axis, is outside every one of their regions."""
    assert not bdf_is_stable(order, -0.01 + 1.0j), f'BDF{order} should reject -0.01+1j'


def test_bdf4_stable_but_bdf5_unstable_inside_bdfs_cone():
    """A point 55 deg from the negative real axis at |z| = 2.5: inside BDF4's 73.35
    deg cone (and its stability region there), outside BDF5's 51.84 deg cone. This is
    the measured discriminator between the two registered A(alpha) claims."""
    angle = math.radians(55.0)
    z = 2.5 * complex(-math.cos(angle), math.sin(angle))
    assert bdf_is_stable(4, z), f'BDF4 should contain the 55-deg cone point, z={z}'
    assert not bdf_is_stable(5, z), f'BDF5 should reject the 55-deg cone point, z={z}'


def test_bdf4_history_reset_on_dt_change_stays_correct():
    """A dt change mid-run invalidates the whole state-snapshot history (StepHistory's
    own guard) and the scheme falls back to the Dormand-Prince starter until the
    snapshots rebuild: no crash, no stale-snapshot mixing, still near the analytic
    solution."""
    scheme = getIntegrator('BDF4')
    problem = testing.PROBLEMS['oscillator']()
    system = problem.initial()
    history = StepHistory(maxlen=max(1, scheme.steps))
    for i in range(8):
        dt = 0.1 if i < 4 else 0.05  # change dt partway through
        result = scheme(system, dt=dt, f=problem.rhs, history=history,
                        solver_opts=_ORDER_SOLVER_OPTS)
        system, history = result.state, result.history
    final_x = float(testing.get_reference_state(system).x[0])
    exact_x, _ = problem.exact(4 * 0.1 + 4 * 0.05)
    assert abs(final_x - exact_x[0]) < 0.1


def test_bdf2_without_history_uses_a_safe_high_order_startup():
    scheme = getIntegrator('BDF2')
    problem = testing.PROBLEMS['oscillator']()
    result = scheme(problem.initial(), dt=0.1, f=problem.rhs)
    assert result.history is not None
    assert result.state.t == pytest.approx(0.1)


def test_bdf2_history_contains_the_previous_state_snapshot():
    scheme = getIntegrator('BDF2')
    problem = testing.PROBLEMS['oscillator']()
    first = scheme(problem.initial(), dt=0.1, f=problem.rhs, history=StepHistory(maxlen=1))
    assert first.history.latest.state is not None
    second = scheme(first.state, dt=0.1, f=problem.rhs, history=first.history)
    assert second.state.t == pytest.approx(0.2)