import warnings
import json
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression

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
        'test_catalog.py',
        'test_copied_fields.py',
        'test_drift_fields.py',
        'test_doc_examples.py',
        'test_family_enums.py',
        'test_groundwork.py',
        'test_kwargs_passthrough.py',
        'test_state.py',
        'test_test_gates.py',
        'test_tensor.py',
        'test_run.py', 'test_run_outputs.py', 'test_operator_splitting.py',
    ],
    # The JFNK / GMRES solver core and its preconditioning hooks.
    'solvers': [
        'test_jfnk.py',
        'test_preconditioner.py', 'test_spatial_preconditioner.py',
    ],
    # Family-specific drivers and tableaus.
    'dirk': ['test_dirk.py'],
    'prrk': ['test_prrk.py'],
    'bdf': ['test_bdf.py','test_stiff_startup.py'],
    'am': ['test_am.py'],
    'multistep': ['test_multistep.py'],
    'ark': ['test_ark.py'],
    'imex': ['test_imex.py', 'test_imexmultistep.py'],
    'fullyimplicit': ['test_fullyimplicit.py','test_implicit_prototypes.py','test_firk_wave9.py'],
    'exponential': ['test_exponential.py'],
    'rosenbrock': ['test_rosenbrock.py'],
    'rkc': ['test_rkc.py'],
    'newmark': ['test_newmark.py'],
    # Cross-family properties over the whole registry.
    'convergence': ['test_convergence.py', 'test_embedded.py'],
    'reuse': ['test_step_reuse.py'],
    'geometry': ['test_hamiltonian.py'],
    'stability': ['test_benchmarks.py', 'test_stiff.py', 'test_tvd.py'],
    'gradients': ['test_adaptive.py', 'test_gradients.py'],
    'rhs': ['test_rhs.py'],
}

#: Reverse of CATEGORIES: test file -> category marker.
def category_map(categories):
    result = {}
    for marker, files in categories.items():
        for filename in files:
            if filename in result:
                raise pytest.UsageError(f'{filename} belongs to multiple test categories')
            result[filename] = marker
    return result


_FILE_MARKER = category_map(CATEGORIES)
CI_SHARDS = json.loads((Path(__file__).parents[1] / '.github/test-shards.json').read_text())


def validate_shards(categories, shards):
    expressions = [Expression.compile(shard) for shard in shards]
    for category in categories:
        matches = sum(expr.evaluate(lambda name: name == category) for expr in expressions)
        if matches != 1:
            raise pytest.UsageError(
                f'test category {category!r} matches {matches} CI shards; expected exactly one')
    return expressions


def pytest_collection_modifyitems(config, items):
    expressions = validate_shards(CATEGORIES, CI_SHARDS)
    for item in items:
        marker = _FILE_MARKER.get(item.fspath.basename)
        if marker is None:
            raise pytest.UsageError(
                f'{item.fspath.basename} lacks a test category; add it to CATEGORIES')
        item.add_marker(getattr(pytest.mark, marker))
        markers = {mark.name for mark in item.iter_markers()}
        matches = sum(expr.evaluate(lambda name: name in markers) for expr in expressions)
        if matches != 1:
            raise pytest.UsageError(
                f'{item.nodeid} matches {matches} CI shards; expected exactly one')


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


def emits_estimate(scheme):
    """Read driver/tableau metadata rather than duplicating the growing catalog.

    ROS3P and EXPRB32 have built-in estimates instead of a Butcher companion.
    ETD2RK and the stage-count methods do not emit estimates; their dedicated
    suites cover their special RHS and per-step parameter requirements.
    """
    return scheme.has_error_estimate


ESTIMATE_SCHEMES = [s.name for s in IntegrationSchemes if emits_estimate(s)]


def estimate_order(scheme):
    """Use the same public estimate exponent a caller uses with propose_dt."""
    return scheme.estimate_order

#: How far below the claimed order a measurement is allowed to sit. Convergence
#: orders measured over four halvings land within a few hundredths of the true value;
#: 0.15 leaves room for that without admitting a lost order.
ORDER_TOLERANCE = 0.15

#: Convergence grids for (scheme, problem) pairs whose standard grid (dt
#: 0.1 .. 0.0125) sits at or below the 1e-12 double-precision roundoff floor --
#: so accurate there that measured_order has no usable pair (None), not a low
#: number. The same phenomenon as the EXACT_ON_LINEAR_PROBLEMS skip in
#: test_convergence.py, on problems where the scheme's error is simply too
#: small. Each grid starts coarser (every entry divides T = 2.0) so the errors
#: sit above the floor; the measured order is then taken from the two finest
#: points above the floor, which are asymptotic at those dt.
#:
#: BS5 needs the coarsest grid for an extra reason: its x-component error on
#: forced is superconvergent (the u error is oscillatory and its time integral
#: over [0, 2] nearly cancels), so a higher-order term dominates at moderate dt
#: (ratio 1235 ~ 2^10.3 between dt = 0.5 and 0.25) and crosses the true dt^5
#: term near dt ~ 0.17 -- the L1 apparent order drifts 6.95 (0.5..0.25) to 4.03
#: (0.125..0.0625) across grids, while the u component alone decays at 5.16-5.43.
#: On the [2.0, 1.0, 0.5, 0.25] grid its finest usable pair measures 6.95, above
#: the registered 5, which is pinned directly on oscillator / kepler.
#:
#: The wave-7 high-order tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13) are the
#: first schemes whose errors floor out on more than one problem at once. The
#: 3-point grids (RKV76IIa, TanYam7 on forced; Feagin10 on kepler) are the
#: coarsest that keeps the two finest points above the floor: with a 4th point
#: at dt = 0.25 that point is floor-contaminated and the measured pair drags
#: the order down 0.5-1.5 below the true one.
PROBLEM_COARSE_GRIDS = {
    'forced': {
        # Wave 9: GL3's order-6 collocation error is already ~4e-12 at h=.1;
        # refine from a coarser grid to retain two points above roundoff.
        'Gauss-Legendre 3': [.4,.2,.1,.05],
        'BS5 5(4)': [2.0, 1.0, 0.5, 0.25],
        'Alshina6': [1.0, 0.5, 0.25, 0.125],
        'MSRK6': [1.0, 0.5, 0.25, 0.125],
        'SIR54': [1.0, 0.5, 0.25, 0.125],
        'Vern6 6(5)': [2.0, 1.0, 0.5, 0.25],
        'Vern7 7(6)': [1.0, 0.5, 0.25, 0.125],
        'Vern8 8(7)': [1.0, 0.5, 0.25, 0.125],
        'Vern9 9(8)': [2.0, 1.0, 0.5, 0.25],
        'RKV76IIa 7(6)': [2.0, 1.0, 0.5],
        'TanYam7 7(6)': [2.0, 1.0, 0.5],
        'TsitPap8 8(7)': [2.0, 1.0, 0.5, 0.25],
        'DP8': [1.0, 0.5, 0.25, 0.125],
        'Feagin10': [2.0, 1.0, 0.5, 0.25],
        # Feagin12's MAIN order is unmeasurable on forced (see FORCED_FLOOR);
        # this grid pins its lower-order REUSE order in test_step_reuse.py
        # (measured 7.04 here vs the predicted 6).
        'Feagin12': [1.0, 0.5, 0.25, 0.125],
        # Wave 8 (NOTES.md 3.28): the order-6 ESDIRK's standard-grid errors sit
        # at the floor from dt = 0.025 down (2.8e-13, 9.0e-15, 3.9e-15), so no
        # order is measurable there; on this grid all four points sit above the
        # floor and measure 6.00.
        'ESDIRK6(5)9L[2]SA': [0.5, 0.25, 0.125, 0.0625],
        # Wave 8 (NOTES.md 3.28): the 12-digit ODE DIRK families sit on a
        # higher effective roundoff floor than the explicit schemes (their JFNK
        # solves leave a small residual per stage), so their standard-grid
        # finest points are floor-contaminated and the measured order drifts
        # out of band. These are the coarsest grids whose measured order stays
        # within 0.15 of the claimed one (Cash4: 3.998, SFSDIRK5: 3.914,
        # SFSDIRK7: 4.013, SFSDIRK8: 3.901).
        'Cash4': [0.1, 0.05, 0.025, 0.0125],
        'SFSDIRK5': [0.2, 0.1, 0.05, 0.025],
        'SFSDIRK7': [0.4, 0.2, 0.1, 0.05],
        'SFSDIRK8': [0.2, 0.1, 0.05, 0.025],
    },
    'oscillator': {
        'Vern6 6(5)': [0.4, 0.2, 0.1, 0.05],
        'Vern7 7(6)': [0.4, 0.2, 0.1, 0.05],
        'Vern8 8(7)': [0.4, 0.2, 0.1, 0.05],
        'Vern9 9(8)': [0.4, 0.2, 0.1, 0.05],
        'RKV76IIa 7(6)': [1.0, 0.5, 0.25, 0.125],
        'TanYam7 7(6)': [1.0, 0.5, 0.25, 0.125],
        'TsitPap8 8(7)': [0.4, 0.2, 0.1, 0.05],
        'DP8': [0.4, 0.2, 0.1, 0.05],
        'Feagin10': [0.4, 0.2, 0.1, 0.05],
        'Feagin12': [0.4, 0.2, 0.1, 0.05],
        'Feagin14': [0.4, 0.2, 0.1, 0.05],
        # Wave 8 (NOTES.md 3.28): the order-6 ESDIRK's standard-grid errors are
        # floor-contaminated at the two finest points (8.8e-14, 5.7e-15), which
        # drags the measured order to 5.84 (below the 6 - 0.15 bound); on this
        # grid all four points sit above the floor and measure 5.89.
        'ESDIRK6(5)9L[2]SA': [0.5, 0.25, 0.125, 0.0625],
        # Wave 8 (NOTES.md 3.28): roundoff-floor grids, as under 'forced'
        # (measured 4.050 and 3.990 respectively).
        'SFSDIRK5': [0.1, 0.05, 0.025, 0.0125],
        'SFSDIRK8': [0.2, 0.1, 0.05, 0.025],
    },
    'kepler': {
        # Phase 1: default JFNK noise contaminates the standard-grid finest
        # pair. Strict solves restore orders 3.998 / 5.998 / 3.999 there;
        # these default-solve grids measure 4.009 / 5.966 / 3.990 instead.
        # The strict-solve standard-grid regression lives in test_dirk.py.
        'ESDIRK4(3)7L[2]SA': [0.2, 0.1, 0.05, 0.025],
        'ESDIRK6(5)9L[2]SA': [0.4, 0.2, 0.1, 0.05],
        'SFSDIRK6': [0.4, 0.2, 0.1, 0.05],
        'Vern6 6(5)': [1.0, 0.5, 0.25, 0.125],
        'Vern7 7(6)': [0.4, 0.2, 0.1, 0.05],
        'Vern8 8(7)': [0.4, 0.2, 0.1, 0.05],
        'Vern9 9(8)': [1.0, 0.5, 0.25, 0.125],
        'RKV76IIa 7(6)': [0.4, 0.2, 0.1, 0.05],
        # g4's dt = 0.1 point (1.2e-13) is floor-dropped, leaving only the
        # pre-asymptotic coarse pair (7.69 < 7.85); the (0.5, 0.25) pair on
        # the g2 grid is the finest asymptotic one (7.855).
        'TsitPap8 8(7)': [1.0, 0.5, 0.25, 0.125],
        'DP8': [0.4, 0.2, 0.1, 0.05],
        'Feagin10': [2.0, 1.0, 0.5],
        'Feagin12': [1.0, 0.5, 0.25, 0.125],
        'Feagin14': [1.0, 0.5, 0.25, 0.125],
    },
    'damped': {
        'Vern6 6(5)': [0.4, 0.2, 0.1, 0.05],
        'Vern7 7(6)': [0.4, 0.2, 0.1, 0.05],
        'Vern8 8(7)': [0.4, 0.2, 0.1, 0.05],
        'Vern9 9(8)': [0.4, 0.2, 0.1, 0.05],
        'RKV76IIa 7(6)': [1.0, 0.5, 0.25, 0.125],
        'TanYam7 7(6)': [0.4, 0.2, 0.1, 0.05],
        'TsitPap8 8(7)': [0.4, 0.2, 0.1, 0.05],
        # g4's dt = 0.1 point (6.5e-13) is floor-dropped, leaving the
        # pre-asymptotic (0.2, 0.1) pair (7.77 < 7.85); the g2 grid's
        # (0.25, 0.125) pair measures 7.878.
        'DP8': [1.0, 0.5, 0.25, 0.125],
        'Feagin10': [0.4, 0.2, 0.1, 0.05],
        'Feagin12': [0.4, 0.2, 0.1, 0.05],
        'Feagin14': [0.4, 0.2, 0.1, 0.05],
        # Wave 8 (NOTES.md 3.28): roundoff-floor grids, as under 'forced'
        # (measured 4.087, 4.033 and 3.993 respectively).
        'SFSDIRK5': [0.1, 0.05, 0.025, 0.0125],
        'SFSDIRK7': [0.2, 0.1, 0.05, 0.025],
        'SFSDIRK8': [0.2, 0.1, 0.05, 0.025],
    },
}

#: Schemes whose main-order error on the non-autonomous `forced` problem sits
#: at the roundoff floor at EVERY dt that divides T = 2.0 -- even the single
#: dt = 2.0 step (1.8e-11 for Feagin12, 5.8e-14 for Feagin14) leaves fewer
#: than two points above the floor, so no grid can measure a main order there
#: and test_convergence skips the forced problem for them (the order is pinned
#: on the other three problems instead). Their lower-order REUSE errors are a
#: different quantity, sit comfortably above the floor, and are measured fine
#: (test_step_reuse.py).
FORCED_FLOOR = {'Feagin12', 'Feagin14'}


def dts_for(scheme_name, problem_name, default):
    """The convergence grid for (scheme, problem): the coarser per-problem grid
    when the scheme's standard-grid errors would sit at the roundoff floor; the
    caller's default otherwise."""
    return PROBLEM_COARSE_GRIDS.get(problem_name, {}).get(scheme_name, default)


@pytest.fixture(params=ALL_SCHEMES)
def scheme(request):
    return request.param


@pytest.fixture
def step_sizes():
    return testing.default_step_sizes()


def problem(name):
    return testing.PROBLEMS[name]()


def order_of(scheme, problem_name, dts, T=2.0, reuse=False, **step_options):
    """Measured convergence order, with reuse warnings silenced.

    The warnings are the feature under test in test_step_reuse.py; everywhere else
    they are noise.
    """
    if scheme.name in {'Gauss-Legendre 3','Radau IIA s=3'}:
        # Default FD/Newton floors (~1e-10 globally) obscure orders 5/6.
        # Measure discretization with JVP and a stricter linear solve; leave
        # the production default and every order assertion unchanged.
        step_options.setdefault('solver_opts',dict(matvec='jvp',tol=1e-13,newton_tol=1e-9))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        order, errors = testing.convergence(scheme, problem(problem_name), dts, T,
                                           reuse=reuse, **step_options)
    return order, errors
