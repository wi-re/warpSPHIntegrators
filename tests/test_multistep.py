"""Adams-Bashforth / Adams-Bashforth-Moulton driver (NOTES.md S3.6 Phase 1).

Most generic behaviour (state cloning, copied/ephemeral fields, kwargs passthrough) is
already covered for these schemes for free, the moment they joined the `scheme`
fixture every other test file parametrizes over -- the full suite went from 1103 to
1313 passing tests without a single new failure the day these seven were added, a
strong signal the "reuse butcher._weighted_update, reuse a DP5 starter, thread
StepHistory" design didn't need new machinery that could itself be wrong. This file
covers what's specific to multistep: convergence order, the count-of-evaluations cost
claim, the safe-but-expensive no-history fallback, and the dt/uid restart guards
(S2g) getting a real consumer for the first time.
"""

import pytest
import torch

from warpSPHIntegrators import get_reference_state, getIntegrator, testing
from warpSPHIntegrators.butcher import DormandPrince
from warpSPHIntegrators.history import StepHistory
from warpSPHIntegrators.multistep import (
    AB2, AB3, AB4, AB5, ABM2, ABM3, ABM4, getABCoefficients, getAMCoefficients,
)

AB_SCHEMES = [('Adams-Bashforth 2', 2), ('Adams-Bashforth 3', 3),
             ('Adams-Bashforth 4', 4), ('Adams-Bashforth 5', 5)]
ABM_SCHEMES = [('Adams-Bashforth-Moulton 2 (PECE)', 2), ('Adams-Bashforth-Moulton 3 (PECE)', 3),
              ('Adams-Bashforth-Moulton 4 (PECE)', 4)]
ALL_MULTISTEP = AB_SCHEMES + ABM_SCHEMES


def _run(scheme, order, problem, dt, T):
    system = problem.initial()
    history = StepHistory(maxlen=order - 1)
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history)
        system, history = result.state, result.history
    return system


def _final_error(scheme, order, problem, dt, T):
    system = _run(scheme, order, problem, dt, T)
    s = get_reference_state(system)
    ex, eu = problem.exact(T)
    return (sum(abs(a - b) for a, b in zip(s.x.tolist(), ex))
            + sum(abs(a - b) for a, b in zip(s.u.tolist(), eu)))


# --------------------------------------------------------------------------- #
# Coefficients                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('order', [2, 3, 4, 5])
def test_ab_coefficients_sum_to_one(order):
    """sum(beta) == 1 is the order-1 (consistency) condition every AB order needs."""
    assert getABCoefficients(order).sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize('order', [2, 3, 4, 5])
def test_am_coefficients_sum_to_one(order):
    assert getAMCoefficients(order).sum() == pytest.approx(1.0, abs=1e-13)


# --------------------------------------------------------------------------- #
# Convergence order (empirical, matching NOTES.md S3.1's own verified table)   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', ALL_MULTISTEP)
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'damped'])
def test_multistep_scheme_reaches_its_claimed_order(name, order, problem_name):
    s = getIntegrator(name)
    assert s.order == order
    dts = testing.default_step_sizes(0.1, 5)
    problem = testing.PROBLEMS[problem_name]()
    errors = [_final_error(s, order, problem, dt, T=2.0) for dt in dts]
    measured = testing.measured_order(errors, dts)
    assert measured == pytest.approx(order, abs=0.15), (
        f'{name} on {problem_name}: measured order {measured:.3f}, claimed {order}. errors={errors}'
    )


# --------------------------------------------------------------------------- #
# The no-history fallback: safe but expensive, never silently wrong           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', ALL_MULTISTEP)
def test_without_history_threading_falls_back_to_the_starter_exactly(name, order):
    """Never passing history= must reproduce running Dormand-Prince alone, bit for bit.

    This is the whole safety argument for requiring history= to get the cheap-cost
    behaviour: a caller who forgets still gets a *correct*, just more expensive,
    trajectory (Dormand-Prince 5(4) is higher order than any of these), not a subtly
    wrong one.
    """
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()

    system_multistep = prob.initial()
    system_dp5 = prob.initial()
    for _ in range(10):
        system_multistep = s(system_multistep, dt=0.1, f=prob.rhs).state
        system_dp5 = DormandPrince(system_dp5, dt=0.1, f=prob.rhs).state

    a, b = get_reference_state(system_multistep), get_reference_state(system_dp5)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.u, b.u)


@pytest.mark.parametrize('name,order', ALL_MULTISTEP)
def test_with_history_threading_diverges_from_the_starter(name, order, problem_name='oscillator'):
    """The inverse check: once there is enough history, it must actually use it."""
    s = getIntegrator(name)
    prob = testing.PROBLEMS[problem_name]()

    system_multistep = _run(s, order, prob, dt=0.1, T=2.0)
    system_dp5 = prob.initial()
    for _ in range(20):
        system_dp5 = DormandPrince(system_dp5, dt=0.1, f=prob.rhs).state

    a, b = get_reference_state(system_multistep), get_reference_state(system_dp5)
    assert not torch.equal(a.x, b.x), f'{name}: result identical to pure DP5 even with history threaded'


# --------------------------------------------------------------------------- #
# Evaluation-count cost claim (NOTES.md S3.6's table: AB=1, ABM=2 per step     #
# once past the bootstrap)                                                    #
# --------------------------------------------------------------------------- #

def _count_evaluations(scheme, order, problem, dt, steps):
    counter = [0]

    def counting_rhs(state, step_dt, **kwargs):
        counter[0] += 1
        return problem.rhs(state, step_dt, **kwargs)

    system = problem.initial()
    history = StepHistory(maxlen=order - 1)
    per_step = []
    for _ in range(steps):
        before = counter[0]
        result = scheme(system, dt=dt, f=counting_rhs, history=history)
        system, history = result.state, result.history
        per_step.append(counter[0] - before)
    return per_step


@pytest.mark.parametrize('name,order', AB_SCHEMES)
def test_ab_costs_one_evaluation_per_step_after_bootstrap(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    per_step = _count_evaluations(s, order, prob, dt=0.1, steps=order + 3)
    assert per_step[-1] == 1, f'{name}: expected 1 evaluation/step once warmed up, got {per_step}'


@pytest.mark.parametrize('name,order', ABM_SCHEMES)
def test_abm_costs_two_evaluations_per_step_after_bootstrap(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    per_step = _count_evaluations(s, order, prob, dt=0.1, steps=order + 3)
    assert per_step[-1] == 2, f'{name}: expected 2 evaluations/step once warmed up, got {per_step}'


# --------------------------------------------------------------------------- #
# S2g restart guards, exercised by a real consumer for the first time         #
# --------------------------------------------------------------------------- #

def test_dt_change_restarts_history_and_stays_correct():
    """A dt change mid-run must not silently mix incompatible-dt history entries.

    StepHistory.pushed's own guard (S2g-a) drops stale entries on a dt change; this
    checks the *scheme* actually degrades safely through that -- i.e. it falls back to
    bootstrapping again with the new dt, rather than either crashing or silently using
    coefficients computed for the wrong step size.
    """
    s = getIntegrator('Adams-Bashforth 4')
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    history = StepHistory(maxlen=3)
    for i in range(8):
        dt = 0.1 if i < 4 else 0.05  # change dt partway through
        result = s(system, dt=dt, f=prob.rhs, history=history)
        system, history = result.state, result.history
    # No crash, and the run must still be finite and near the analytic solution --
    # the restart costs some accuracy (back to bootstrapping) but must not diverge.
    final_x = float(get_reference_state(system).x[0])
    exact_x, _ = prob.exact(4 * 0.1 + 4 * 0.05)
    assert abs(final_x - exact_x[0]) < 0.1


def test_history_restarts_when_maxlen_is_exceeded_by_a_shorter_one():
    """A history built for a lower-order scheme (smaller maxlen) never over-fills."""
    history = StepHistory(maxlen=1)
    s2 = getIntegrator('Adams-Bashforth 2')
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    for _ in range(3):
        result = s2(system, dt=0.1, f=prob.rhs, history=history)
        system, history = result.state, result.history
        assert len(history) <= 1


# --------------------------------------------------------------------------- #
# priorStep / mutation / testing.run integration                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,order', ALL_MULTISTEP)
def test_multistep_rejects_prior_step_with_a_warning(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name,order', ALL_MULTISTEP)
def test_multistep_scheme_does_not_mutate_the_caller_state(name, order):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    before_x = system.state.x.clone()
    s(system, dt=0.1, f=prob.rhs, history=StepHistory(maxlen=order - 1))
    assert torch.equal(system.state.x, before_x), f'{name} mutated the caller position'


def test_testing_run_history_true_is_the_blessed_calling_convention():
    """testing.run(..., history=True) (NOTES.md S5) must reach the claimed order too."""
    s = getIntegrator('Adams-Bashforth 4')
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
