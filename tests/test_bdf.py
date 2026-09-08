import pytest

from warpSPHIntegrators import getIntegrator, testing
from warpSPHIntegrators.history import StepHistory


def _run(scheme, problem, dt, T):
    system = problem.initial()
    history = StepHistory(maxlen=1)
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history)
        system, history = result.state, result.history
    return system


@pytest.mark.parametrize(('name', 'expected_order'), [('BDF1', 1), ('BDF2', 2)])
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_bdf_reaches_its_claimed_order_with_threaded_history(name, expected_order, problem_name):
    scheme = getIntegrator(name)
    problem = testing.PROBLEMS[problem_name]()
    dts = testing.default_step_sizes(0.1, 5)
    errors = []
    for dt in dts:
        state = _run(scheme, problem, dt, T=2.0)
        exact_x, exact_u = problem.exact(2.0)
        ref = testing.get_reference_state(state)
        errors.append(sum(abs(a - b) for a, b in zip(ref.x.tolist(), exact_x)) +
                      sum(abs(a - b) for a, b in zip(ref.u.tolist(), exact_u)))
    assert testing.measured_order(errors, dts) == pytest.approx(expected_order, abs=0.15)


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