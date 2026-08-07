import warnings

import pytest

from warpSPHIntegrators import testing
from warpSPHIntegrators.integration import IntegrationSchemes


#: Every registered scheme, as pytest params keyed by display name.
ALL_SCHEMES = [pytest.param(s, id=s.name) for s in IntegrationSchemes]

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
