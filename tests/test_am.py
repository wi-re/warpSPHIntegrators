"""Fully implicit (iterated) Adams-Moulton corrector (NOTES.md S3.8 Phase 4).

The AM corrector formula solved to convergence with JFNK, as distinct from the
uniterated Adams-Bashforth-Moulton (PECE) corrector in `test_multistep.py`. Covers
what's specific to the implicit family: convergence order, the corrector actually
converging (solver diagnostics), the stiff-nonlinear case where the iterated
corrector beats the uniterated PECE one, the no-history DP5 fallback, the
predictor on/off agreement, and the same priorStep/mutation/restart guards the
explicit family has.
"""

import math

import pytest
import torch

from warpSPHIntegrators import get_reference_state, getIntegrator, testing
from warpSPHIntegrators.butcher import DormandPrince
from warpSPHIntegrators.history import StepHistory
from warpSPHIntegrators.testing import ParticleUpdate

AM_SCHEMES = [('Adams-Moulton 2 (implicit)', 2), ('Adams-Moulton 3 (implicit)', 3),
             ('Adams-Moulton 4 (implicit)', 4)]
# Matching PECE predictor-corrector, for the stiff comparison.
PECE_SCHEMES = [('Adams-Bashforth-Moulton 2 (PECE)', 2),
               ('Adams-Bashforth-Moulton 3 (PECE)', 3),
               ('Adams-Bashforth-Moulton 4 (PECE)', 4)]


def _run(scheme, order, problem, dt, T, **kwargs):
    system = problem.initial()
    history = StepHistory(maxlen=order - 1)
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history, **kwargs)
        system, history = result.state, result.history
    return system


def _final_error(scheme, order, problem, dt, T, **kwargs):
    system = _run(scheme, order, problem, dt, T, **kwargs)
    s = get_reference_state(system)
    ex, eu = problem.exact(T)
    return (sum(abs(a - b) for a, b in zip(s.x.tolist(), ex))
            + sum(abs(a - b) for a, b in zip(s.u.tolist(), eu)))


# The order runs solve each AM step with the exact (autograd) Jacobian and a tight
# tolerance: with the default finite-difference matvec the JFNK residual bottoms
# out at ~1e-7, the same size as AM4's truncation error at the finest step sizes
# below, which would corrupt the measured order. The order test measures the
# method, so the linear solver must not be the limiting factor.
_ORDER_SOLVER_OPTS = {'matvec': 'jvp', 'tol': 1e-12, 'newton_tol': 1e-12}


# --------------------------------------------------------------------------- #
# Convergence order (incl. the non-autonomous `forced` problem)               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', AM_SCHEMES)
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'damped', 'kepler'])
def test_am_scheme_reaches_its_claimed_order(name, order, problem_name):
    s = getIntegrator(name)
    assert s.order == order
    dts = testing.default_step_sizes(0.1, 5)
    problem = testing.PROBLEMS[problem_name]()
    errors = [_final_error(s, order, problem, dt, T=2.0,
                           solver_opts=_ORDER_SOLVER_OPTS) for dt in dts]
    measured = testing.measured_order(errors, dts)
    assert measured == pytest.approx(order, abs=0.15), (
        f'{name} on {problem_name}: measured order {measured:.3f}, claimed {order}. errors={errors}'
    )


# --------------------------------------------------------------------------- #
# The no-history fallback: safe but expensive, never silently wrong           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_without_history_threading_falls_back_to_the_starter_exactly(name, order):
    """Never passing history= must reproduce running Dormand-Prince alone, bit for
    bit -- the same safety argument as the explicit multistep family (test_multistep).
    """
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()

    system_am = prob.initial()
    system_dp5 = prob.initial()
    for _ in range(10):
        system_am = s(system_am, dt=0.1, f=prob.rhs).state
        system_dp5 = DormandPrince(system_dp5, dt=0.1, f=prob.rhs).state

    a, b = get_reference_state(system_am), get_reference_state(system_dp5)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.u, b.u)


@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_with_history_threading_diverges_from_the_starter(name, order):
    """The inverse check: once there is enough history, it must actually use it."""
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()

    system_am = _run(s, order, prob, dt=0.1, T=2.0)
    system_dp5 = prob.initial()
    for _ in range(20):
        system_dp5 = DormandPrince(system_dp5, dt=0.1, f=prob.rhs).state

    a, b = get_reference_state(system_am), get_reference_state(system_dp5)
    assert not torch.equal(a.x, b.x), f'{name}: result identical to pure DP5 even with history threaded'


# --------------------------------------------------------------------------- #
# Stiff nonlinear relaxation: the corrector converges, and beats PECE         #
# --------------------------------------------------------------------------- #

def _stiff_relaxation_rhs(rate):
    """Nonlinear Prothero-Robinson form with exact solution x(t)=tanh(t)."""
    def rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        time = float(system.t)
        target = math.tanh(time)
        target_derivative = 1.0 / math.cosh(time) ** 2
        return ParticleUpdate(
            dxdt=-rate * (state.x - target) + target_derivative,
            dudt=torch.zeros_like(state.u),
            dedt=torch.zeros_like(state.e),
        ), None
    return rhs


@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_am_corrector_converges_on_stiff_nonlinear_relaxation(name, order):
    """rate=10, dt=0.1 (stiff scale z = 1): every JFNK solve must reach the
    corrector's fixed point. 'tolerance' is the normal termination; 'stagnation'
    is acceptable too, because with the default finite-difference matvec the
    residual bottoms out at the noise floor (~1e-7) before it can halve again --
    the assertion that pins the solve is the residual itself, not the label.
    (rate=100 is deliberately not used: the Dormand-Prince startup is explicit,
    so it is unstable there, and the AM stability region does not contain
    z = -10 for order >= 3 either -- the startup, not the corrector, limits it.)
    """
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    history = StepHistory(maxlen=order - 1)
    for _ in range(10):
        result = s(system, dt=0.1, f=_stiff_relaxation_rhs(10.0), history=history)
        system, history = result.state, result.history
        if result.solver_diagnostics is not None:
            d = result.solver_diagnostics
            assert d.termination in ('tolerance', 'stagnation'), f'{name}: {d.termination}'
            assert d.residual < 1e-5, f'{name}: corrector residual {d.residual} did not converge'
    x = float(get_reference_state(system).x[0])
    assert math.isfinite(x)
    assert abs(x - math.tanh(1.0)) < 0.2


def _stiff_final_error(scheme, order, rate, dt, T):
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    history = StepHistory(maxlen=order - 1)
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=_stiff_relaxation_rhs(rate), history=history)
        system, history = result.state, result.history
    return abs(float(get_reference_state(system).x[0]) - math.tanh(T))


@pytest.mark.parametrize(('am_name', 'pece_name', 'order'), [
    ('Adams-Moulton 2 (implicit)', 'Adams-Bashforth-Moulton 2 (PECE)', 2),
    ('Adams-Moulton 3 (implicit)', 'Adams-Bashforth-Moulton 3 (PECE)', 3),
    ('Adams-Moulton 4 (implicit)', 'Adams-Bashforth-Moulton 4 (PECE)', 4),
])
def test_am_corrector_beats_pece_on_stiff_nonlinear_case(am_name, pece_name, order):
    """The Phase 4 gate: on a stiff nonlinear case the *iterated* AM corrector is at
    least as accurate as the *uniterated* PECE one of the same order. Measured at
    rate=10, dt=0.1, 10 steps: AM errors ~6e-5 / 8e-5 / 2e-5 vs PECE
    ~8e-4 / 4e-4 / 3e-3 -- the corrector's extra iteration is worth an order of
    magnitude once the problem is stiff enough for the predictor to be wrong.
    """
    am_err = _stiff_final_error(getIntegrator(am_name), order, rate=10.0, dt=0.1, T=1.0)
    pece_err = _stiff_final_error(getIntegrator(pece_name), order, rate=10.0, dt=0.1, T=1.0)
    assert am_err < pece_err, \
        f'{am_name} err={am_err:.3e} should beat {pece_name} err={pece_err:.3e} on the stiff case'


# --------------------------------------------------------------------------- #
# Predictor on/off: same fixed point, different starting guess                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_predictor_only_changes_the_initial_guess(name, order):
    """predictor=True starts the JFNK solve from the matching Adams-Bashforth
    prediction, predictor=False from the known part. Both must land on the same
    corrector fixed point (to solver tolerance), i.e. the predictor is an
    efficiency choice, not a numerical one, on a non-stiff problem."""
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()

    with_pred = _run(s, order, prob, dt=0.1, T=2.0)
    without_pred = _run(s, order, prob, dt=0.1, T=2.0, predictor=False)

    a = get_reference_state(with_pred)
    b = get_reference_state(without_pred)
    for field in ('x', 'u'):
        assert (getattr(a, field) - getattr(b, field)).abs().max().item() < 1e-6, \
            f'{name}: predictor on/off disagree in {field}'


# --------------------------------------------------------------------------- #
# priorStep / mutation / restart / testing.run integration                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_am_rejects_prior_step_with_a_warning(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name,order', AM_SCHEMES)
def test_am_scheme_does_not_mutate_the_caller_state(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    before_x = system.state.x.clone()
    s(system, dt=0.1, f=prob.rhs, history=StepHistory(maxlen=order - 1))
    assert torch.equal(system.state.x, before_x), f'{name} mutated the caller position'


def test_dt_change_restarts_history_and_stays_correct():
    """A dt change mid-run drops the stale history entries (StepHistory's own guard)
    and the scheme falls back to bootstrapping again: no crash, still near the
    analytic solution."""
    s = getIntegrator('Adams-Moulton 3 (implicit)')
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    history = StepHistory(maxlen=2)
    for i in range(8):
        dt = 0.1 if i < 4 else 0.05  # change dt partway through
        result = s(system, dt=dt, f=prob.rhs, history=history)
        system, history = result.state, result.history
    final_x = float(get_reference_state(system).x[0])
    exact_x, _ = prob.exact(4 * 0.1 + 4 * 0.05)
    assert abs(final_x - exact_x[0]) < 0.1


def test_testing_run_history_true_is_the_blessed_calling_convention():
    """testing.run(..., history=True) (NOTES.md S5) must reach the claimed order too."""
    s = getIntegrator('Adams-Moulton 4 (implicit)')
    prob = testing.PROBLEMS['oscillator']()
    dts = testing.default_step_sizes(0.1, 5)
    errors = []
    for dt in dts:
        system = testing.run(s, prob, dt, T=2.0, history=True)
        se = get_reference_state(system)
        ex, eu = prob.exact(2.0)
        errors.append(sum(abs(a - b) for a, b in zip(se.x.tolist(), ex))
                      + sum(abs(a - b) for a, b in zip(se.u.tolist(), eu)))
    order = testing.measured_order(errors, dts)
    assert order == pytest.approx(4, abs=0.15), f'measured order {order}, errors={errors}'
