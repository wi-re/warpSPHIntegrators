"""Caller kwargs must survive the trip through every scheme.

The schemes forward the caller's `**kwargs` wholesale into `initializeNewState`, the
right-hand side, the lifecycle hooks and the internal helpers. That makes any *named*
parameter on an internal helper a collision waiting to happen: if the caller passes a
kwarg of the same name, it arrives both positionally and in `**kwargs`, and Python
raises `TypeError: got multiple values for argument`.

`_weighted_update` had exactly that bug with `verbose` -- which `RungeKuttaB` reads but
never pops -- so every Butcher scheme raised as soon as a caller passed `verbose` at
all, in either state. Nothing in the suite passed `verbose`, so nothing caught it.
"""

import warnings

import pytest

from warpSPHIntegrators import get_reference_state, testing

from conftest import problem


@pytest.mark.parametrize('verbose', [False, True])
def test_scheme_accepts_verbose(scheme, verbose, capsys):
    """`verbose` is read by the schemes but stays in kwargs. Both states must work."""
    prob = problem('oscillator')
    result = scheme(prob.initial(), dt=0.05, f=prob.rhs, verbose=verbose)
    assert result.state is not None
    assert get_reference_state(result.state).x is not None

    if verbose:
        assert capsys.readouterr().out, f'{scheme.name} printed nothing with verbose=True'


@pytest.mark.parametrize('verbose', [False, True])
def test_verbose_does_not_change_the_answer(scheme, verbose):
    """Logging must be observation only."""
    prob = problem('oscillator')
    quiet = scheme(prob.initial(), dt=0.05, f=prob.rhs)
    loud = scheme(prob.initial(), dt=0.05, f=prob.rhs, verbose=verbose)
    assert get_reference_state(loud.state).x.tolist() == \
        pytest.approx(get_reference_state(quiet.state).x.tolist())


def test_scheme_accepts_arbitrary_caller_kwargs(scheme):
    """A kwarg the library knows nothing about must reach the RHS untouched."""
    prob = problem('oscillator')
    seen = []

    def rhs(system, dt, **kwargs):
        seen.append(kwargs.get('my_solver_config'))
        return prob.rhs(system, dt)

    scheme(prob.initial(), dt=0.05, f=rhs, my_solver_config={'alpha': 1})
    assert seen and all(c == {'alpha': 1} for c in seen), (
        f'{scheme.name} dropped or mangled a caller kwarg: {seen}'
    )


def test_verbose_works_alongside_step_reuse(scheme):
    """The reuse path prints too, and takes `verbose` as a positional of its own."""
    if scheme.reuse_order is None:
        pytest.skip(f'{scheme.name} does not implement reuse')
    prob = problem('oscillator')
    system = prob.initial()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        prior = None
        for _ in range(3):
            result = scheme(system, dt=0.05, f=prob.rhs, priorStep=prior, verbose=True)
            system, prior = result.state, result.stages[-1]
    assert get_reference_state(system).x is not None


@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_convergence_is_unaffected_by_verbose(problem_name, step_sizes):
    """One end-to-end check that the logging path is not a different code path."""
    from warpSPHIntegrators import getIntegrator

    s = getIntegrator('RK4')
    prob = testing.PROBLEMS[problem_name]()
    quiet = [testing.final_error(s, prob, dt, 1.0) for dt in step_sizes]

    def run_loud(dt):
        system = prob.initial()
        for _ in range(int(round(1.0 / dt))):
            system = s(system, dt=dt, f=prob.rhs, verbose=False).state
        st = get_reference_state(system)
        ex, eu = prob.exact(1.0)
        return (sum(abs(a - b) for a, b in zip(st.x.tolist(), ex))
                + sum(abs(a - b) for a, b in zip(st.u.tolist(), eu)))

    assert [run_loud(dt) for dt in step_sizes] == pytest.approx(quiet)
