"""Gradient-through-step coverage, and the implicit-differentiation path behind it.

Before this file existed, ``docs/research/integrators/notes-index.md``'s Completion Definition claimed the
torch path was "differentiable by construction" and recorded that the claim had no
test. It was in fact false for every implicit scheme: ``gmres`` builds its Hessenberg
factor, Givens rotations and back-substitution vector by in-place element writes, so
the moment a caller asked for a gradient the autograd tape rejected it with
"one of the variables needed for gradient computation has been modified by an inplace
operation". That hit on step 1 for DIRK/Newmark/IMEX/ARK/BDF1, and as soon as the
explicit Dormand-Prince cold start handed over for BDF2+/AM.

``JFNKSolver.solve`` now runs the whole Newton/GMRES iteration under ``no_grad`` and
re-attaches gradients by the implicit function theorem. The tests below pin both the
mechanics (gradients exist and flow) and the property that distinguishes implicit
differentiation from unrolling: the answer depends only on *where* the fixed point is,
not on how many iterations reached it.
"""

import math

import pytest
import torch

from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.state import StepHistory
from warpSPHIntegrators.schemes import IntegrationSchemes
from warpSPHIntegrators.schemes import DIRK as DIRKFamily
from warpSPHIntegrators.testing import ParticleState, ParticleSystem, ParticleUpdate


K = 4.0
DT = 0.05


def _oscillator_at(x0_value=1.0, requires_grad=True):
    """A one-particle oscillator state whose position is the differentiation root."""
    x0 = torch.tensor([x0_value], dtype=torch.float64, requires_grad=requires_grad)
    system = ParticleSystem(
        state=ParticleState(
            x=x0,
            u=torch.tensor([0.0], dtype=torch.float64),
            e=torch.tensor([0.0], dtype=torch.float64),
            m=torch.tensor([1.0], dtype=torch.float64),
        ),
        t=0.0,
    )
    return x0, system


def _integrate(scheme_name, steps, *, x0_value=1.0, requires_grad=True, solver=None):
    """Run `steps` steps of `scheme_name` and return (final x, the x0 leaf)."""
    scheme = get_scheme(scheme_name)
    problem = testing.oscillator_problem(k=K)
    x0, system = _oscillator_at(x0_value, requires_grad)
    history = StepHistory(maxlen=8)
    kwargs = {} if solver is None else {'solver': solver}
    for _ in range(steps):
        result = scheme.function(system, DT, problem.rhs, history=history, **kwargs)
        system = result.state
        if result.history is not None:
            history = result.history
    return get_reference_state(system).x.sum(), x0


def _dx_dx0(scheme_name, steps, **kwargs):
    out, x0 = _integrate(scheme_name, steps, **kwargs)
    (grad,) = torch.autograd.grad(out, x0)
    return float(grad)


#: Every implicit scheme that closes its stage equation with JFNK. Enough steps that
#: the multistep schemes are past their explicit cold start and genuinely running the
#: implicit path -- the bug this file guards hid behind exactly that handover.
# ROS3P uses the frozen linear-operator gradient path instead of JFNK's IFT.
# Its linear and nonlinear parity checks live in test_rosenbrock.py.
IMPLICIT_GRADIENT_EXCLUSIONS = {'ROS3P': 'dedicated frozen-operator parity tests'}
IMPLICIT_SCHEMES = [
    (s.name, s.steps + 3 if s.startup_order is not None else 2)
    for s in IntegrationSchemes
    if s.implicit and s.name not in IMPLICIT_GRADIENT_EXCLUSIONS
]


# Per-step stage-count and exponential/Krylov drivers have dedicated vector
# gradient parity tests in test_rkc.py and test_exponential.py. ETD2RK also
# requires a SemilinearRHS, absent from this oscillator test.
EXPLICIT_GRADIENT_EXCLUSIONS = {
    'RKC1': 'dedicated stage-count gradient parity tests',
    'RKC2': 'dedicated stage-count gradient parity tests',
    'RKL2': 'dedicated stage-count gradient parity tests',
    'ETD2RK': 'dedicated semilinear Krylov gradient parity tests',
    'EXPRB32': 'dedicated exponential Rosenbrock gradient parity tests',
}
EXPLICIT_SCHEMES = [s.name for s in IntegrationSchemes
                    if not s.implicit and s.name not in EXPLICIT_GRADIENT_EXCLUSIONS]


@pytest.mark.parametrize('s', [s for s in IntegrationSchemes if s.family is DIRKFamily],
                         ids=lambda s: s.name)
def test_dirk_rhs_parameter_gradient_matches_finite_differences(s):
    """The IFT must retain captured RHS parameters as well as state roots."""
    solver = JFNKSolver(tol=1e-12, gmres_tol=1e-12, newton_tol=1e-12)

    def integrate(k_value, track):
        k = torch.tensor(k_value, dtype=torch.float64, requires_grad=track)
        _, state = _oscillator_at(requires_grad=track)

        def rhs(system, dt, **kwargs):
            ref = get_reference_state(system)
            return ParticleUpdate(dxdt=ref.u, dudt=-k * ref.x,
                                  dedt=torch.zeros_like(ref.e)), None

        for _ in range(2):
            state = s(state, DT, rhs, solver=solver).state
        return get_reference_state(state).x.sum(), k

    out, k = integrate(K, True)
    (gradient,) = torch.autograd.grad(out, k)
    h = 1e-4
    with torch.no_grad():
        plus, _ = integrate(K + h, False)
        minus, _ = integrate(K - h, False)
    expected = float(plus - minus) / (2 * h)
    assert float(gradient) == pytest.approx(expected, rel=1e-5, abs=1e-8)


def _exact_sensitivity(steps):
    """d x_N / d x_0 for the *exact* oscillator flow: x(t) = x_0 cos(omega t)."""
    return math.cos(math.sqrt(K) * steps * DT)


@pytest.mark.parametrize('name,steps', [pytest.param(n, s, id=n) for n, s in IMPLICIT_SCHEMES])
def test_gradient_flows_through_every_implicit_scheme(name, steps):
    """The regression this file exists for: these all raised RuntimeError before.

    The bar is deliberately loose -- each scheme has its own truncation error at this
    step size, and pinning that is the convergence suite's job, not this one's. What
    matters here is that a finite, correctly-signed, roughly-right gradient comes out
    at all.
    """
    grad = _dx_dx0(name, steps)
    expected = _exact_sensitivity(steps)
    assert grad == pytest.approx(expected, abs=0.02), (
        f'{name} gradient {grad} is far from the exact flow sensitivity {expected}')


@pytest.mark.parametrize('name', EXPLICIT_SCHEMES)
def test_gradient_still_flows_through_explicit_schemes(name):
    """Explicit schemes were never broken; nothing in the fix may change that."""
    s = get_scheme(name)
    steps = max(3, s.steps + 3) if s.startup_order is not None else 3
    grad = _dx_dx0(name, steps)
    assert grad == pytest.approx(_exact_sensitivity(steps), abs=0.02)


# --------------------------------------------------------------------------- #
# Correctness against a closed form, not just "a number came out"             #
# --------------------------------------------------------------------------- #

def _linear_oscillator_sensitivity(one_step_map, steps):
    """d x_N / d x_0 for a linear one-step map on x' = u, u' = -k x."""
    A = torch.tensor([[0.0, 1.0], [-K, 0.0]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    M = one_step_map(identity, A, DT)
    return float(torch.linalg.matrix_power(M, steps)[0, 0])


#: Both maps are exact for these schemes on a *linear* problem, so the measured
#: gradient must match to machine precision, not merely to a finite-difference
#: tolerance.
CLOSED_FORM = {
    'Backward Euler (implicit)': lambda I, A, dt: torch.linalg.inv(I - dt * A),
    'BDF1': lambda I, A, dt: torch.linalg.inv(I - dt * A),
    'IMEX Euler': lambda I, A, dt: torch.linalg.inv(I - dt * A),
    'Trapezoidal (Crank-Nicolson)': lambda I, A, dt: torch.linalg.inv(I - dt / 2 * A) @ (I + dt / 2 * A),
    'Implicit Midpoint': lambda I, A, dt: torch.linalg.inv(I - dt / 2 * A) @ (I + dt / 2 * A),
}
# Adams-Moulton 2 is the trapezoidal rule, but it cold-starts from Dormand-Prince, so
# a run long enough to reach its own formula is no longer a pure trapezoidal map and
# has no single closed form to check against. Its gradient is covered by the
# flows-through and tolerance-independence tests instead.


@pytest.mark.parametrize('name', sorted(CLOSED_FORM))
def test_gradient_matches_the_closed_form_amplification_matrix(name):
    """Implicit differentiation is exact at the fixed point, so this is not approximate.

    Each scheme here has a known rational one-step map on the linear oscillator; the
    sensitivity is the (0,0) entry of its N-th power. Unrolling the Newton iteration
    would only approach this as the solve converged -- implicit differentiation hits
    it outright.
    """
    expected = _linear_oscillator_sensitivity(CLOSED_FORM[name], steps=2)
    measured = _dx_dx0(name, 2, solver=JFNKSolver(tol=1e-14, newton_tol=1e-6,
                                                  max_iterations=200))
    assert measured == pytest.approx(expected, abs=1e-12), (
        f'{name}: gradient {measured!r} vs closed form {expected!r}')


@pytest.mark.parametrize('name', ['Backward Euler (implicit)', 'Trapezoidal (Crank-Nicolson)',
                                  'TR-BDF2', 'ESDIRK4(3)6L[2]SA'])
def test_gradient_is_independent_of_the_newton_tolerance(name):
    """The property that separates implicit differentiation from unrolling.

    A gradient taken *through* the solver iterations depends on how many iterations
    ran, so loosening the tolerance would move it. The implicit function theorem uses
    only the converged iterate, so a 6-order-of-magnitude tolerance sweep must leave
    the gradient alone even where it visibly moves the *iterate*.
    """
    grads = [
        _dx_dx0(name, 2, solver=JFNKSolver(tol=1e-14, newton_tol=nt, max_iterations=200))
        for nt in (1.0, 1e-3, 1e-6)
    ]
    assert grads[0] == pytest.approx(grads[1], abs=1e-11)
    assert grads[1] == pytest.approx(grads[2], abs=1e-11)


def test_gradient_is_correct_for_a_nonlinear_problem():
    """Kepler is nonlinear in the state, so J is not constant across the trajectory."""
    problem = testing.kepler_problem()

    def integrate(scale, requires_grad):
        system = problem.initial()
        ref = get_reference_state(system)
        x = (ref.x.detach() * scale).requires_grad_(requires_grad)
        ref.x = x
        scheme = get_scheme('Backward Euler (implicit)')
        for _ in range(3):
            system = scheme.function(system, 0.01, problem.rhs).state
        return get_reference_state(system).x.sum(), x

    out, leaf = integrate(1.0, True)
    (grad,) = torch.autograd.grad(out, leaf)

    h = 1e-6
    with torch.no_grad():
        plus, _ = integrate(1.0 + h, False)
        minus, _ = integrate(1.0 - h, False)
    # The finite-difference reference perturbs a scale factor, so compare against the
    # directional derivative grad . x0 rather than the raw gradient vector.
    directional = float((grad * get_reference_state(problem.initial()).x).sum())
    assert directional == pytest.approx(float(plus - minus) / (2 * h), rel=1e-4)


# --------------------------------------------------------------------------- #
# The non-differentiable path must be untouched                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name,steps', [pytest.param(n, s, id=n) for n, s in IMPLICIT_SCHEMES])
def test_values_are_identical_with_and_without_grad_tracking(name, steps):
    """Turning gradients on must not perturb the trajectory itself.

    The solve runs under ``no_grad`` either way; the differentiable path only adds
    extra ``step`` applications at the converged point. So the forward answer must be
    bit-for-bit the same, not merely close.
    """
    with_grad, _ = _integrate(name, steps, requires_grad=True)
    without_grad, _ = _integrate(name, steps, requires_grad=False)
    assert float(with_grad.detach()) == float(without_grad)


def test_solve_leaves_diagnostics_untouched_on_the_differentiable_path():
    """Re-attaching gradients must not inflate the reported solver cost.

    The extra ``step`` applications belong to the gradient machinery, not to the
    nonlinear solve, so ``rhs_evaluations`` (whose cost model docs/research/integrators/notes-3-4.md documents)
    must report the same number either way.
    """
    problem = testing.oscillator_problem(k=K)
    scheme = get_scheme('TR-BDF2')

    def diagnostics(requires_grad):
        _, system = _oscillator_at(requires_grad=requires_grad)
        result = scheme.function(system, DT, problem.rhs)
        return [(d.rhs_evaluations, d.gmres_iterations, d.termination)
                for d in result.solver_diagnostics if d is not None]

    assert diagnostics(True) == diagnostics(False)


def test_detached_input_returns_the_plain_solver_result():
    """No grad anywhere means the old code path, including the returned state object."""
    problem = testing.oscillator_problem(k=K)
    scheme = get_scheme('Backward Euler (implicit)')
    _, system = _oscillator_at(requires_grad=False)
    with torch.no_grad():
        result = scheme.function(system, DT, problem.rhs)
    assert not get_reference_state(result.state).x.requires_grad
