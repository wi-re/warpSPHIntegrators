"""Phase 2: preconditioned JFNK (IMPLICIT_ROADMAP.md Phase 2).

Covers the optional ``preconditioner(v, state, context)`` hook added to
``gmres`` (left or right) and ``JFNKSolver``:

  * the *identity* preconditioner is bitwise identical to no preconditioner in
    both modes -- the "preserve current behaviour / identity baseline" gate;
  * left and right preconditioning both solve a stiff system to the same
    accuracy, and agree with each other;
  * a per-DOF *diagonal* preconditioner materially cuts GMRES iterations on a
    variable-coefficient stiff relaxation, while the preconditioned and
    unpreconditioned solutions agree;
  * the ``JFNKSolver``-level plumbing hands the current Newton iterate and the
    ``preconditioner_context`` to the 3-arg callable, and a preconditioned solve
    reports fewer ``gmres_iterations`` than the unpreconditioned one;
  * ``diagonal_preconditioner`` works for both a fixed tensor and a
    state-dependent callable, and ``gmres`` rejects an unknown mode.

None of this changes the default (no-preconditioner) path, so the rest of the
suite is unaffected; ``test_jfnk.py`` remains the regression guard for that path.
"""

import pytest
import torch

from warpSPHIntegrators import (
    JFNKSolver,
    diagonal_preconditioner,
    get_reference_state,
    gmres,
    identity_preconditioner,
)
from warpSPHIntegrators.solvers import SolveResult
from warpSPHIntegrators.testing import ParticleState, ParticleSystem

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# A variable-coefficient stiff relaxation:  A x = b  with                     #
#   A = I + dt * (diag(K) + eps * L),   K in [1, kmax],  L = tridiag(-1,2,-1).#
# Matrix-free so no dense Jacobian is ever formed.                            #
# --------------------------------------------------------------------------- #

def _stiff_var_matvec(n: int, dt: float = 10.0, eps: float = 0.1, kmax: float = 100.0):
    """Return ``(matvec, inv_diag)`` for ``A = I + dt*(diag(K) + eps*L)``.

    ``K`` is a per-DOF relaxation rate spanning ``[1, kmax]`` (the realistic SPH
    case: stiffness varies by particle), so a *per-DOF* diagonal preconditioner
    is a strong approximation to ``A^{-1}`` whereas a scalar one would not be.
    ``inv_diag = 1 / (1 + dt*K)`` ignores only the small ``eps*L`` coupling.
    """
    K = torch.linspace(1.0, kmax, n, dtype=DTYPE)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        Lx = torch.zeros_like(v)
        Lx[0] = 2 * v[0] - v[1]
        Lx[-1] = 2 * v[-1] - v[-2]
        Lx[1:-1] = -v[:-2] + 2 * v[1:-1] - v[2:]
        return v + dt * (K * v + eps * Lx)

    return matvec, 1.0 / (1.0 + dt * K)


# --------------------------------------------------------------------------- #
# gmres-level: identity == no preconditioner (bitwise, both sides)            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('preconditioning', ['right', 'left'])
def test_identity_preconditioner_is_bitwise_identical_to_none(preconditioning):
    """A supplied identity preconditioner must leave the solve *bitwise*
    identical to passing no preconditioner at all -- the gate that the hook
    preserves current behaviour and that the identity is a true baseline."""
    matvec, _ = _stiff_var_matvec(n=40)
    torch.manual_seed(0)
    b = torch.randn(40, dtype=DTYPE)

    x_ref, it_ref = gmres(matvec, b, tol=1e-10, maxiter=400, restart=30)
    # 3-arg identity, adapted to the 1-arg form gmres consumes.
    x_id, it_id = gmres(matvec, b, tol=1e-10, maxiter=400, restart=30,
                        preconditioner=lambda v: v, preconditioning=preconditioning)
    assert torch.equal(x_ref, x_id), f'{preconditioning}: identity must be bitwise identical'
    assert it_id == it_ref, f'{preconditioning}: identity must use the same iteration count'


def test_identity_preconditioner_helper_matches_raw_identity():
    """The exported 3-arg ``identity_preconditioner`` is a plain ``v -> v``."""
    v = torch.randn(7, dtype=DTYPE)
    assert torch.equal(identity_preconditioner(v, state=None, context={}), v)


# --------------------------------------------------------------------------- #
# gmres-level: left and right both solve, and agree                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('preconditioning', ['right', 'left'])
def test_preconditioned_gmres_solves_the_same_system_as_unpreconditioned(preconditioning):
    matvec, inv_diag = _stiff_var_matvec(n=200)
    torch.manual_seed(0)
    b = torch.randn(200, dtype=DTYPE)

    x_ref, _ = gmres(matvec, b, tol=1e-9, maxiter=4000, restart=30)
    x_prec, _ = gmres(matvec, b, tol=1e-9, maxiter=4000, restart=30,
                      preconditioner=lambda v: v * inv_diag,
                      preconditioning=preconditioning)
    # Both drive the true residual down to the requested tolerance ...
    assert (b - matvec(x_ref)).norm() / b.norm() < 1e-7
    assert (b - matvec(x_prec)).norm() / b.norm() < 1e-7
    # ... and so they agree with each other to well inside that tolerance.
    assert (x_prec - x_ref).norm() / x_ref.norm() < 1e-5, preconditioning


# --------------------------------------------------------------------------- #
# gmres-level: the diagonal preconditioner materially cuts iterations         #
# --------------------------------------------------------------------------- #

def test_diagonal_preconditioner_reduces_gmres_iterations():
    """The 'materially fewer Krylov iterations' gate, at the linear-solver level:
    a per-DOF diagonal preconditioner must cut the GMRES iteration count by a
    clear factor on a variable-coefficient stiff relaxation, without changing
    the achieved residual."""
    matvec, inv_diag = _stiff_var_matvec(n=200)
    torch.manual_seed(0)
    b = torch.randn(200, dtype=DTYPE)

    x_ref, it_ref = gmres(matvec, b, tol=1e-9, maxiter=4000, restart=30)
    x_right, it_right = gmres(matvec, b, tol=1e-9, maxiter=4000, restart=30,
                              preconditioner=lambda v: v * inv_diag, preconditioning='right')
    x_left, it_left = gmres(matvec, b, tol=1e-9, maxiter=4000, restart=30,
                            preconditioner=lambda v: v * inv_diag, preconditioning='left')

    # Reference solve is genuinely expensive (this is the stiff regime).
    assert it_ref > 40, f'unpreconditioned solve should be stiff here, got {it_ref} iters'
    # Right preconditioning cuts it by a solid factor; left by more.
    assert it_right < 0.6 * it_ref, f'right: {it_right} iters not < 0.6*{it_ref}'
    assert it_left < 0.6 * it_ref, f'left: {it_left} iters not < 0.6*{it_ref}'
    # And the extra work buys the same accuracy (residual still at tolerance).
    for x in (x_right, x_left):
        assert (b - matvec(x)).norm() / b.norm() < 1e-7


def test_gmres_rejects_an_unknown_preconditioning_mode():
    matvec, _ = _stiff_var_matvec(n=10)
    b = torch.randn(10, dtype=DTYPE)
    with pytest.raises(ValueError, match="preconditioning must be 'left' or 'right'"):
        gmres(matvec, b, tol=1e-9, maxiter=20, preconditioner=lambda v: v,
              preconditioning='two-sided')


# --------------------------------------------------------------------------- #
# diagonal_preconditioner helper: fixed tensor and state-dependent callable   #
# --------------------------------------------------------------------------- #

def test_diagonal_preconditioner_fixed_tensor():
    inv = torch.arange(1.0, 6.0, dtype=DTYPE)
    prec = diagonal_preconditioner(inv)
    v = torch.ones(5, dtype=DTYPE)
    assert torch.equal(prec(v, state=None, context={}), inv)


def test_diagonal_preconditioner_state_dependent_callable():
    """A state-dependent diagonal reads its value from the current iterate --
    the shape of a relaxation-term preconditioner like ``1/(1 + dt*damping)``."""
    def inv_from_state(state, context):
        s = get_reference_state(state)
        return 1.0 / (1.0 + context['dt'] * s.e)  # e plays the role of the rate

    prec = diagonal_preconditioner(inv_from_state)
    state = ParticleSystem(state=ParticleState(
        x=torch.ones(3, dtype=DTYPE), u=torch.zeros(3, dtype=DTYPE),
        e=torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
        m=torch.ones(3, dtype=DTYPE)))
    v = torch.ones(3, dtype=DTYPE)
    out = prec(v, state=state, context={'dt': 1.0})
    assert torch.equal(out, 1.0 / torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE))


# --------------------------------------------------------------------------- #
# JFNKSolver-level: plumbing, agreement, and the gmres_iterations diagnostic  #
# --------------------------------------------------------------------------- #

def _stiff_particle_problem(n: int, dt: float = 10.0, eps: float = 0.1, kmax: float = 100.0):
    """A length-``n`` stiff relaxation on the ``x`` block (``u``/``e`` carried),
    in backward-Euler fixed-point form ``step(x) = x0 + dt*(-K x + eps L x)``.
    Linear, so JFNK's Newton loop is a single correction and the GMRES
    iteration count is exactly the cost of the one stage linear solve.
    Returns ``(system, x0, inv_full)`` where ``inv_full`` is the per-DOF
    diagonal inverse for the *full* flattened ``(x, u, e)`` state.
    """
    K = torch.linspace(1.0, kmax, n, dtype=DTYPE)
    x0 = torch.rand(n, dtype=DTYPE)

    def step(Y):
        s_in = get_reference_state(Y)
        out = Y.initializeNewState()
        s_out = get_reference_state(out)
        x = s_in.x
        Lx = torch.zeros_like(x)
        Lx[0] = 2 * x[0] - x[1]
        Lx[-1] = 2 * x[-1] - x[-2]
        Lx[1:-1] = -x[:-2] + 2 * x[1:-1] - x[2:]
        s_out.x = x0 + dt * (-K * x + eps * Lx)
        s_out.u, s_out.e = s_in.u.clone(), s_in.e.clone()
        return out

    system = ParticleSystem(state=ParticleState(
        x=x0.clone(), u=torch.zeros(n, dtype=DTYPE), e=torch.zeros(n, dtype=DTYPE),
        m=torch.ones(n, dtype=DTYPE)))
    inv_full = torch.cat([1.0 / (1.0 + dt * K),
                          torch.ones(n, dtype=DTYPE),
                          torch.ones(n, dtype=DTYPE)])
    return system, step, inv_full


def test_jfnk_preconditioned_matches_unpreconditioned_solution():
    """Gate: 'preconditioned and unpreconditioned solutions agree on a linear
    reference problem.' Both must land on the same fixed point."""
    system, step, inv_full = _stiff_particle_problem(n=128)

    ref = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10).solve(step, system)
    prec = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10,
                      preconditioner=lambda v, state, context: v * context['inv'],
                      preconditioning='right',
                      preconditioner_context={'inv': inv_full}).solve(step, system)

    assert ref.converged and prec.converged
    assert isinstance(prec, SolveResult)
    x_ref = get_reference_state(ref.y).x
    x_prec = get_reference_state(prec.y).x
    assert (x_prec - x_ref).norm() / x_ref.norm() < 1e-6


def test_jfnk_preconditioner_reduces_reported_gmres_iterations():
    """Gate: 'materially fewer GMRES iterations with a supplied preconditioner',
    read off the ``SolveDiagnostics.gmres_iterations`` a caller actually sees."""
    system, step, inv_full = _stiff_particle_problem(n=200)

    ref = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10).solve(step, system)
    prec = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10,
                      preconditioner=lambda v, state, context: v * context['inv'],
                      preconditioning='right',
                      preconditioner_context={'inv': inv_full}).solve(step, system)

    it_ref = ref.diagnostics.gmres_iterations
    it_prec = prec.diagnostics.gmres_iterations
    assert it_ref > 40, f'unpreconditioned solve should be stiff here, got {it_ref}'
    assert it_prec < 0.6 * it_ref, f'preconditioned {it_prec} iters not < 0.6*{it_ref}'


def test_jfnk_preconditioner_receives_state_and_context():
    """The 3-arg contract: the callable gets the current iterate as ``state`` and
    the merged ``{'state': ..., **preconditioner_context}`` as ``context``."""
    system, step, _ = _stiff_particle_problem(n=16)
    seen = {}

    def prec(v, state, context):
        seen.setdefault('calls', 0)
        seen['calls'] += 1
        seen['state_is_iterate'] = get_reference_state(state).x.shape == get_reference_state(system).x.shape
        seen['context_has_inv'] = 'marker' in context
        seen['context_state_matches'] = context['state'] is state
        return v  # identity: no effect on the solve

    JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10,
               preconditioner=prec, preconditioning='left',
               preconditioner_context={'marker': 1.0}).solve(step, system)

    assert seen['calls'] > 0
    assert seen['state_is_iterate']
    assert seen['context_has_inv']
    assert seen['context_state_matches']
