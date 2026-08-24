"""JFNK core machinery (JFNK_PLAN.md Phase A): flatten/unflatten (A1), the two
matvec builders (A2 finite-difference, A3 exact-JVP), GMRES (A4), and
``JFNKSolver`` (A5) plugged into the DIRK driver as a ``solver=`` override.

None of this touches anything registered this session -- ``JFNKSolver`` is
opt-in only (JFNK_PLAN.md A5) -- so the full suite's pass/skip count is
unaffected by this file existing.
"""

import pytest
import torch

from warpSPHIntegrators import (
    FixedPointSolver,
    JFNKSolver,
    get_reference_state,
    getIntegrator,
    gmres,
    testing,
)
from warpSPHIntegrators.dirk import DIRK, getDIRKTableau
from warpSPHIntegrators.fields import (
    flatten_integrated,
    integrated_field_names,
    unflatten_integrated,
)
from warpSPHIntegrators.jfnk import fd_matvec, jvp_matvec
from warpSPHIntegrators.solvers import SolveResult
from warpSPHIntegrators.testing import ParticleState, ParticleSystem
from warpSPHIntegrators.util import updateStateEuler, updateStep


# --------------------------------------------------------------------------- #
# A1: flatten/unflatten                                                       #
# --------------------------------------------------------------------------- #

def test_integrated_field_names_lists_only_integrated_tensor_fields():
    sys0 = testing.PROBLEMS['oscillator']().initial()
    assert integrated_field_names(sys0) == ['x', 'u', 'e']


def test_flatten_integrated_concatenates_in_that_order():
    sys0 = testing.PROBLEMS['oscillator']().initial()
    s = get_reference_state(sys0)
    flat = flatten_integrated(sys0)
    assert torch.equal(flat, torch.cat([s.x, s.u, s.e]))


def test_unflatten_integrated_is_the_inverse_of_flatten_integrated():
    sys0 = testing.PROBLEMS['oscillator']().initial()
    flat = flatten_integrated(sys0)
    roundtripped = unflatten_integrated(flat, sys0)
    assert torch.equal(flatten_integrated(roundtripped), flat)


def test_unflatten_integrated_writes_the_new_values():
    sys0 = testing.PROBLEMS['oscillator']().initial()
    flat = flatten_integrated(sys0)
    new_flat = flat + 1.0
    out = unflatten_integrated(new_flat, sys0)
    s = get_reference_state(out)
    assert torch.equal(s.x, get_reference_state(sys0).x + 1.0)
    assert torch.equal(s.u, get_reference_state(sys0).u + 1.0)
    assert torch.equal(s.e, get_reference_state(sys0).e + 1.0)


def test_unflatten_integrated_clones_non_integrated_fields_independently():
    """``m`` (a ``constant`` field) must be preserved but not aliased -- mutating
    the unflattened copy must not affect the original, matching the "leave
    everything else alone, but independent" contract ``state_difference`` uses.
    """
    sys0 = testing.PROBLEMS['oscillator']().initial()
    out = unflatten_integrated(flatten_integrated(sys0), sys0)
    s_out, s_in = get_reference_state(out), get_reference_state(sys0)
    assert torch.equal(s_out.m, s_in.m)
    s_out.m += 1.0
    assert not torch.equal(s_out.m, s_in.m)


def test_unflatten_integrated_rejects_a_mismatched_flat_length():
    sys0 = testing.PROBLEMS['oscillator']().initial()
    with pytest.raises(ValueError):
        unflatten_integrated(torch.zeros(2), sys0)


# --------------------------------------------------------------------------- #
# A4: GMRES, on hand-built operators (no state machinery involved)            #
# --------------------------------------------------------------------------- #

def test_gmres_solves_a_symmetric_system():
    torch.manual_seed(0)
    n = 20
    A = torch.randn(n, n, dtype=torch.float64)
    A = A + A.T + n * torch.eye(n, dtype=torch.float64)  # SPD-ish, diagonally dominant
    b = torch.randn(n, dtype=torch.float64)
    x, _iters = gmres(lambda v: A @ v, b, tol=1e-10, maxiter=100)
    assert torch.linalg.norm(A @ x - b) / torch.linalg.norm(b) < 1e-8


def test_gmres_solves_a_nonsymmetric_system():
    torch.manual_seed(1)
    n = 20
    A = torch.randn(n, n, dtype=torch.float64) + n * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    x, _iters = gmres(lambda v: A @ v, b, tol=1e-10, maxiter=100)
    assert torch.linalg.norm(A @ x - b) / torch.linalg.norm(b) < 1e-8
    # Cross-check against a direct solve, since a nonsymmetric operator has no
    # variational characterization to fall back on.
    x_direct = torch.linalg.solve(A, b)
    assert torch.allclose(x, x_direct, rtol=1e-6, atol=1e-8)


def test_gmres_never_calls_matvec_on_the_zero_vector():
    """The default ``x0=0`` residual (``r = b - matvec(0)``) is skipped rather than
    evaluated, since ``matvec`` is always linear here (``matvec(0) == 0`` exactly)
    -- this also sidesteps a real ``torch.autograd.forward_ad`` incompatibility a
    live JVP matvec hits on an all-zero tangent (see ``jvp_matvec``'s docstring).
    """
    calls = []

    def matvec(v):
        calls.append(v.clone())
        return 5.0 * v

    b = torch.tensor([10.0])
    x, _ = gmres(matvec, b, tol=1e-10)
    assert torch.allclose(x, torch.tensor([2.0]), atol=1e-8)
    assert all(float(v.abs().max()) > 0 for v in calls)


# --------------------------------------------------------------------------- #
# A2/A3: FD and exact-JVP matvecs agree on a case with a known Jacobian        #
# --------------------------------------------------------------------------- #

def _backward_euler_step_fn(prob, dt: float):
    """The same stage map DIRK builds internally for backward Euler's one
    implicit stage: ``Y -> y0 + dt * f(Y)``.
    """
    y0 = prob.initial()

    def step_fn(Y):
        k, _r = updateStep(y0, Y, dt, prob.rhs)
        return updateStateEuler(y0, k, dt, copyState=True)

    return y0, step_fn


def test_fd_and_jvp_matvecs_agree_on_a_linear_problem():
    """oscillator's right-hand side is linear, so both matvecs approximate the
    exact same constant Jacobian -- FD to its own truncation tolerance, JVP
    exactly.
    """
    prob = testing.PROBLEMS['oscillator'](k=4.0)
    y0, step_fn = _backward_euler_step_fn(prob, dt=0.1)

    y_flat = flatten_integrated(y0)
    G_y = y_flat - flatten_integrated(step_fn(y0))
    mv_fd = fd_matvec(step_fn, y0, y_flat, G_y)
    mv_jvp = jvp_matvec(step_fn, y0)

    torch.manual_seed(2)
    v = torch.randn_like(y_flat)
    assert torch.allclose(mv_fd(v), mv_jvp(v), rtol=1e-4, atol=1e-8)


def test_jvp_matvec_is_exact_for_a_linear_problem():
    """No FD truncation error at all: the JVP matvec's `v - Jv` must reproduce
    the exact analytic Jacobian action for this linear right-hand side.
    """
    prob = testing.PROBLEMS['oscillator'](k=4.0)
    y0, step_fn = _backward_euler_step_fn(prob, dt=0.1)
    mv_jvp = jvp_matvec(step_fn, y0)

    # G(Y) = Y - (y0 + dt*f(Y)); for this problem f is linear in (x, u, e) so
    # J_G is the constant matrix [[1, -dt, 0], [dt*k, 1, 0], [0, 0, 1]].
    dt, k = 0.1, 4.0
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    expected = torch.tensor([1.0, dt * k, 0.0], dtype=torch.float64)
    assert torch.allclose(mv_jvp(v), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# A5: JFNKSolver, standalone                                                  #
# --------------------------------------------------------------------------- #

def test_jfnk_solver_converges_on_a_trivial_linear_fixed_point():
    """``x = 0.5*x + 1`` has fixed point ``x = 2`` -- the same problem
    ``FixedPointSolver``'s own groundwork test uses, ported onto a state so it
    exercises ``flatten_integrated``/GMRES rather than raw floats.
    """
    sys0 = ParticleSystem(state=ParticleState(
        x=torch.tensor([0.0]), u=torch.tensor([0.0]), e=torch.tensor([0.0]),
        m=torch.tensor([1.0])))

    def step(Y):
        out = Y.initializeNewState()
        s_out, s_in = get_reference_state(out), get_reference_state(Y)
        s_out.x = 0.5 * s_in.x + 1.0
        s_out.u, s_out.e = s_in.u.clone(), s_in.e.clone()
        return out

    result = JFNKSolver(tol=1e-10, max_iterations=10).solve(step, sys0)
    assert isinstance(result, SolveResult)
    assert result.converged is True
    assert float(get_reference_state(result.y).x[0]) == pytest.approx(2.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# A5 through DIRK: JFNK succeeds where Picard measurably fails                #
# --------------------------------------------------------------------------- #

def test_jfnk_reproduces_exact_backward_euler_where_picard_diverges():
    """Mirrors ``test_dirk.py``'s
    ``test_picard_diverges_on_a_stiff_problem_regardless_of_tableau_stability``:
    same stiffness (``dt*omega = 100``), same tableau. Picard(20) is
    astronomically wrong there; JFNK must land on backward Euler's own *exact*
    stage solution instead (Newton on a linear residual is exact, modulo
    GMRES's own tolerance).
    """
    k = 1e6
    dt = 0.1
    prob = testing.PROBLEMS['oscillator'](k=k)
    tab = getDIRKTableau('backwardEuler')

    result_picard = DIRK(prob.initial(), dt=dt, f=prob.rhs, tableau=tab,
                         solver_opts={'iterations': 20})
    x_picard = abs(float(get_reference_state(result_picard.state).x[0]))
    assert x_picard > 1e30, f'expected Picard(20) to still be astronomically wrong here, got {x_picard:.3e}'

    exact = 1.0 / (1.0 + dt ** 2 * k)  # backward Euler's own algebraic stage solution
    for matvec in ('fd', 'jvp'):
        solver = JFNKSolver(matvec=matvec, tol=1e-10, max_iterations=20)
        result = DIRK(prob.initial(), dt=dt, f=prob.rhs, tableau=tab, solver=solver)
        x = float(get_reference_state(result.state).x[0])
        assert x == pytest.approx(exact, rel=1e-4), (
            f'JFNK({matvec}) at dt*omega=100: got {x:.6e}, expected {exact:.6e}'
        )


@pytest.mark.parametrize('name', ['backwardEuler', 'implicitMidpoint', 'SDIRK2'])
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced'])
def test_jfnk_reaches_the_same_order_as_the_shipped_default_on_other_dirk_schemes(name, problem_name):
    """Cheap bonus (JFNK_PLAN.md A6): the driver is generic, so JFNK should reach
    the same convergence order FixedPointSolver does on every tableau, not just
    backward Euler -- this exercises Implicit Midpoint and SDIRK2 too, for free.
    """
    tab = getDIRKTableau(name)
    order_claimed = {'backwardEuler': 1, 'implicitMidpoint': 2, 'SDIRK2': 2}[name]

    def scheme(state, dt, f, *args, **kwargs):
        solver = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=20)
        return DIRK(state, dt, f, tab, *args, solver=solver, **kwargs)

    dts = testing.default_step_sizes(0.1, 5)
    order, errors = testing.convergence(scheme, testing.PROBLEMS[problem_name](), dts, T=2.0)
    assert order == pytest.approx(order_claimed, abs=0.15), (
        f'JFNK+{name} on {problem_name}: measured order {order}, claimed {order_claimed}. errors={errors}'
    )
