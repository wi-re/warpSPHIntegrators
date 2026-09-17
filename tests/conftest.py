import warnings

import pytest

from warpSPHIntegrators import testing
from warpSPHIntegrators.integration import IntegrationSchemes

#: Test categories, as marker name -> test files. Every test file belongs to
#: exactly one category; the marker is assigned at collection time by
#: `pytest_collection_modifyitems` below (no per-file annotation). Categories
#: are the unit of parallel execution -- locally via pytest-xdist
#: (`pytest -m <category> -n auto`) and in CI as the shard matrix of
#: `.github/workflows/tests.yml`. The cross-category net (convergence,
#: stability, gradients, rhs) runs every registered scheme and is the safety
#: net for changes that touch more than one family.
CATEGORIES = {
    # State machinery, field semantics, registry metadata, backend dispatch.
    'core': [
        'test_backend_dispatch.py',
        'test_copied_fields.py',
        'test_family_enums.py',
        'test_groundwork.py',
        'test_kwargs_passthrough.py',
        'test_state.py',
    ],
    # The JFNK / GMRES solver core and its preconditioning hooks.
    'solvers': [
        'test_jfnk.py',
        'test_preconditioner.py',
    ],
    # Family-specific drivers and tableaus.
    'dirk': ['test_dirk.py'],
    'bdf': ['test_bdf.py'],
    'am': ['test_am.py'],
    'multistep': ['test_multistep.py'],
    'ark': ['test_ark.py'],
    'imex': ['test_imex.py', 'test_imexmultistep.py'],
    'fullyimplicit': ['test_fullyimplicit.py'],
    'exponential': ['test_exponential.py'],
    'rosenbrock': ['test_rosenbrock.py'],
    'rkc': ['test_rkc.py'],
    'newmark': ['test_newmark.py'],
    # Cross-family properties over the whole registry.
    'convergence': ['test_convergence.py', 'test_embedded.py', 'test_step_reuse.py'],
    'stability': ['test_benchmarks.py', 'test_hamiltonian.py', 'test_stiff.py', 'test_tvd.py'],
    'gradients': ['test_adaptive.py', 'test_gradients.py'],
    'rhs': ['test_rhs.py'],
}

#: Reverse of CATEGORIES: test file -> category marker.
_FILE_MARKER = {f: marker for marker, files in CATEGORIES.items() for f in files}


def pytest_collection_modifyitems(config, items):
    for item in items:
        marker = _FILE_MARKER.get(item.fspath.basename)
        if marker:
            item.add_marker(getattr(pytest.mark, marker))


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

#: Schemes whose step is the *exact* exponential flow on a linear autonomous
#: right-hand side (NOTES.md S3.16, Phase 7). EXPRB32 freezes the full Jacobian
#: `Jn`; on a linear autonomous problem `Jn` is constant and the nonlinear
#: remainder `D2 = f - f_n - Jn (u - un)` vanishes, so the step reduces to
#: `exp(h Jn)` applied exactly (the two `phi`-Krylov builds exhaust the small
#: Krylov space and are exact to rounding). The measured error is at the
#: roundoff floor and no convergence order is measurable there
#: (`measured_order` returns None) -- the order is pinned on the non-autonomous
#: (`forced`), the nonlinear (`kepler`), and the semilinear canonical
#: (viscous Burgers, in tests/test_exponential.py) problems instead.
EXACT_ON_LINEAR_PROBLEMS = {'EXPRB32'}

#: The generic problems that are linear and autonomous (so
#: `EXACT_ON_LINEAR_PROBLEMS` shows sub-noise error on them).
LINEAR_AUTONOMOUS_PROBLEMS = {'oscillator', 'damped'}


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
