"""`priorStep` reuse must deliver exactly the order the library predicts.

The prediction lives in `integrators.reuse` and is derived from the Butcher tableau
(or recorded by hand for the schemes that have none). These tests are what makes that
prediction trustworthy: for every scheme, the order measured *with* reuse has to match
`scheme.reuse_order`, on both an autonomous and a non-autonomous problem.

They also pin the contract around the feature: schemes that cannot reuse must warn and
ignore rather than silently produce garbage, and the warning must fire when -- and only
when -- reuse actually costs order.
"""

import warnings

import pytest

from integrators import getIntegrator, is_fsal, step_reuse_order, supports_step_reuse, testing
from integrators.reuse import step_reuse_analysis

from conftest import ORDER_TOLERANCE, order_of, problem


#: The FSAL pairs. Reuse on these is exact, not merely harmless: the reused stage is
#: literally f(t^{n+1}, y^{n+1}).
FSAL_SCHEMES = ['Bogacki-Shampine 3(2)', 'Dormand-Prince 5(4)']


@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_measured_reuse_order_matches_prediction(scheme, problem_name, step_sizes):
    if scheme.reuse_order is None:
        pytest.skip(f'{scheme.name} does not implement reuse')
    order, errors = order_of(scheme, problem_name, step_sizes, reuse=True)
    assert order is not None, f'{scheme.name} produced no usable errors: {errors}'
    assert order >= scheme.reuse_order - ORDER_TOLERANCE, (
        f'{scheme.name} on {problem_name} with reuse: predicted order {scheme.reuse_order}, '
        f'measured {order:.2f}. The prediction in integrators.reuse is optimistic.'
    )


@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_reuse_prediction_is_not_pessimistic(scheme, problem_name, step_sizes):
    """A prediction below the truth would warn users off a reuse that is actually free."""
    if scheme.reuse_order is None or scheme.reuse_order >= scheme.order:
        pytest.skip(f'{scheme.name} loses no order under reuse')
    order, _ = order_of(scheme, problem_name, step_sizes, reuse=True)
    assert order < scheme.reuse_order + 1 - ORDER_TOLERANCE, (
        f'{scheme.name} on {problem_name}: predicted {scheme.reuse_order} under reuse but '
        f'measured {order:.2f}. The prediction is too pessimistic; it will warn needlessly.'
    )


@pytest.mark.parametrize('name', FSAL_SCHEMES)
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_fsal_reuse_is_free(name, problem_name, step_sizes):
    """An FSAL pair must lose neither order nor accuracy when reuse is switched on."""
    s = getIntegrator(name)
    assert is_fsal(s), f'{name} should be detected as FSAL from its tableau'
    assert supports_step_reuse(s)

    order_off, errors_off = order_of(s, problem_name, step_sizes)
    order_on, errors_on = order_of(s, problem_name, step_sizes, reuse=True)
    assert order_on >= order_off - ORDER_TOLERANCE

    # Not just the same order: with c_s == 1 and a[-1] == b the reused stage is the
    # same derivative that would have been recomputed, so the errors agree to
    # roundoff rather than merely scaling alike.
    for e_off, e_on in zip(errors_off, errors_on):
        assert e_on == pytest.approx(e_off, rel=1e-9, abs=1e-14), (
            f'{name}: reuse changed the answer ({e_off:.6e} -> {e_on:.6e}); it should be exact'
        )


def test_non_fsal_schemes_are_not_claimed_as_fsal(scheme):
    """FSAL is `c_s == 1 and a[-1] == b`; nothing else may claim it."""
    if scheme.name in FSAL_SCHEMES:
        assert scheme.fsal
    else:
        assert not scheme.fsal, f'{scheme.name} is marked FSAL but is not one of {FSAL_SCHEMES}'


def test_supports_step_reuse_agrees_with_registry(scheme):
    assert supports_step_reuse(scheme) == scheme.supports_reuse
    assert step_reuse_order(scheme) == scheme.reuse_order


def test_single_stage_schemes_refuse_reuse():
    """Forward Euler under reuse becomes y^{n+1} = y^n + dt f(y^{n-1}) -- unstable.

    Consistency analysis alone would call it first order; it is not, because the
    two-step method it degenerates into has no stability region on the imaginary axis.
    The library must refuse rather than report an order.
    """
    s = getIntegrator('Forward Euler')
    assert s.reuse_order is None
    assert not supports_step_reuse(s)
    assert 'stability' in step_reuse_analysis(s).reason

    # And the instability is real: reuse does not converge at all.
    order, _ = order_of(s, 'oscillator', testing.default_step_sizes(), reuse=True)
    assert order < 0.5, f'expected no convergence under reuse, measured order {order:.2f}'


@pytest.mark.parametrize('name', ['TVD RK3', 'TVD RK2', 'PEFRL', 'VEFRL',
                                  'Semi-Implicit Euler', 'Explicit Euler'])
def test_schemes_without_reuse_warn_and_ignore(name, step_sizes):
    """These four used to leak `priorStep` into the user's right-hand side (NOTES 2.9)."""
    s = getIntegrator(name)
    assert s.reuse_order is None

    prob = problem('oscillator')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = s(prob.initial(), dt=0.05, f=prob.rhs, priorStep=('bogus', 'prior'))
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        f'{name} accepted a priorStep without warning'
    assert result.state is not None

    # Ignoring means ignoring: the answer must match the no-reuse run exactly.
    order_on, errors_on = order_of(s, 'oscillator', step_sizes, reuse=True)
    order_off, errors_off = order_of(s, 'oscillator', step_sizes)
    assert errors_on == pytest.approx(errors_off, rel=1e-12)


@pytest.mark.parametrize('name', ['TVD RK3', 'TVD RK2', 'PEFRL', 'VEFRL'])
def test_prior_step_does_not_leak_into_the_rhs(name):
    """`priorStep` must be popped, not forwarded into f / preprocess / postprocess."""
    seen = []
    prob = problem('oscillator')

    def strict_rhs(system, dt, **kwargs):
        seen.append(sorted(kwargs))
        assert 'priorStep' not in kwargs, f'{name} leaked priorStep into the RHS'
        return prob.rhs(system, dt)

    s = getIntegrator(name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        s(prob.initial(), dt=0.05, f=strict_rhs, priorStep=('bogus', 'prior'))
    assert seen, 'the right-hand side was never called'


def test_degrading_reuse_warns_once():
    """The warning names the order actually achieved, and does not repeat per step."""
    from integrators import integration

    s = getIntegrator('SSP RK3')
    assert s.reuse_order == 1 and s.order == 3
    integration._warned_reuse.discard(s.name)

    prob = problem('oscillator')
    system = prob.initial()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prior = None
        for _ in range(5):
            result = s(system, dt=0.05, f=prob.rhs, priorStep=prior)
            system, prior = result.state, result.stages[-1]

    reuse_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(reuse_warnings) == 1, f'expected exactly one warning, got {len(reuse_warnings)}'
    message = str(reuse_warnings[0].message)
    assert 'from 3 to 1' in message, message
    assert 'Dormand-Prince' in message, 'the warning should point at the lossless alternative'


def test_lossless_reuse_does_not_warn():
    """Midpoint keeps its order under reuse, so opting in must be silent."""
    from integrators import integration

    s = getIntegrator('Midpoint')
    assert s.supports_reuse
    integration._warned_reuse.discard(s.name)

    prob = problem('oscillator')
    system = prob.initial()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prior = None
        for _ in range(3):
            result = s(system, dt=0.05, f=prob.rhs, priorStep=prior)
            system, prior = result.state, result.stages[-1]
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
