import warnings

import pytest

from warpSPHIntegrators import testing
from warpSPHIntegrators.integration import IntegrationSchemes


#: Schemes that take a per-step stage count (`s=`, or `lambda_max=`) the generic
#: one-step-callable tests do not provide (NOTES.md S3.13, Phase 12). They are
#: stabilised explicit methods for parabolic (real negative-eigenvalue) right-hand
#: sides, so the generic oscillator / forced / kepler convergence problems do not
#: exercise them correctly either. They are covered in tests/test_rkc.py instead.
NEEDS_STAGE_COUNT = {'RKC1', 'RKC2', 'RKL2'}

#: Schemes that integrate the linear part exactly and therefore require the
#: `linear` accessor of a SemilinearRHS (NOTES.md S3.15, Phase 7). The generic
#: oscillator / forced / kepler convergence problems are registered as plain
#: callables (no `linear` part), so they cannot be run on those as registered.
#: Their order and driver contract are covered in tests/test_exponential.py on
#: the semilinear viscous-Burgers problem instead. (Rosenbrock-W is *not* in this
#: set: it falls back to the combined `f`'s Jacobian, so it runs on a plain
#: callable.)
NEEDS_SEMILINEAR_RHS = {'ETD2RK'}


def _scheme_params():
    params = []
    for s in IntegrationSchemes:
        if s.name in NEEDS_STAGE_COUNT:
            params.append(pytest.param(
                s, id=s.name,
                marks=pytest.mark.skip(
                    reason=(f'{s.name} needs a per-step stage count (s= or lambda_max=); '
                            f'see tests/test_rkc.py'))))
        elif s.name in NEEDS_SEMILINEAR_RHS:
            params.append(pytest.param(
                s, id=s.name,
                marks=pytest.mark.skip(
                    reason=(f'{s.name} needs a SemilinearRHS (the `linear` part); '
                            f'the generic problems are plain callables; '
                            f'see tests/test_exponential.py'))))
        else:
            params.append(pytest.param(s, id=s.name))
    return params


#: Every registered scheme, as pytest params keyed by display name.
ALL_SCHEMES = _scheme_params()

#: How far below the claimed order a measurement is allowed to sit. Convergence
#: orders measured over four halvings land within a few hundredths of the true value;
#: 0.15 leaves room for that without admitting a lost order.
ORDER_TOLERANCE = 0.15


@pytest.fixture(params=ALL_SCHEMES)
def scheme(request):
    return request.param


@pytest.fixture
def step_sizes():
    return testing.default_step_sizes()


def problem(name):
    return testing.PROBLEMS[name]()


def order_of(scheme, problem_name, dts, T=2.0, reuse=False):
    """Measured convergence order, with reuse warnings silenced.

    The warnings are the feature under test in test_step_reuse.py; everywhere else
    they are noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        order, errors = testing.convergence(scheme, problem(problem_name), dts, T, reuse=reuse)
    return order, errors
