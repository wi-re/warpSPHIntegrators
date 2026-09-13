"""Exponential integrators: ETD2RK (NOTES.md S3.15) and EXPRB32 (NOTES.md S3.16),
Phase 7.

A **two-stage, second-order exponential time-differencing Runge-Kutta (ETD2RK)**
one-step method for **semilinear** right-hand sides ``f = L·y + N``. Unlike the
DIRK/BDF drivers (outer Newton loop per stage) and the Rosenbrock-W driver (one
frozen-operator linear solve per stage), an exponential integrator integrates the
*linear* part ``L`` **exactly** -- through the matrix exponential and the related
entire ``phi`` functions -- and only quadratures the mild nonlinear remainder
``N``. The stiffness is carried by ``L``, so the method is unconditionally stable
for the linear part (``exp(dt L)`` for the diffusion is a contraction) and can run
comfortably past the explicit wall.

The scheme (the base, unsplit ETD2RK of Sarumi, arXiv:2601.06849, eqs. (7)-(8);
the ``phi``-function definition is Caliari & Ostermann, *Appl. Numer. Math.* 59
(2009) 568-581, eq. (2.4)):

    N_n  = N(tn, yn)
    w    = exp(dt L) yn + dt phi_1(dt L) N_n              (stage 1: exp-Euler predictor)
    N_w  = N(tn+dt, w)
    yn+1 = exp(dt L) yn + dt phi_1(dt L) N_n
         + dt phi_2(dt L) (N_w - N_n)                     (stage 2)

with ``phi_1(z) = (e^z - 1)/z`` and ``phi_2(z) = (e^z - 1 - z)/z^2``. It is
explicit (``w`` is built from ``yn`` alone, then ``N(w)`` is evaluated), second
order, and L-stable for the linear part (``N = 0`` gives ``yn+1 = exp(dt L) yn``
exactly, so stiff linear modes are damped to ``|exp(dt lambda)|`` per step, killed
as ``dt |lambda| -> inf``). A step costs two ``N`` evaluations plus three
matrix-function-vector products (``exp(dt L)``, ``phi_1(dt L)``, ``phi_2(dt L)``).

EXPRB32
-------
A **two-stage, third-order exponential Rosenbrock method** (``exprb32``) with an
embedded second-order estimator (``exprb22``), from Hochbrueck, Ostermann and
Schweitzer, *SIAM J. Numer. Anal.* 47(1) (2009) 786-803 (the coefficients of
their section 5; the reformulation of their sections 2.3 / 6.5). Where ETD2RK
needs a semilinear split ``f = L·y + N``, an exponential *Rosenbrock* method
needs no split at all: it freezes the **full** Jacobian ``Jn = D_u f(tn, yn)``
of the right-hand side and carries *all* of the stiffness (the linear part
**and** the linearization of the nonlinear part) through the ``phi`` functions
of ``hJn``. The tableau (``phi_k(hJn)`` abbreviates the entire ``phi`` functions
evaluated at the frozen operator) is

    c1 = 0,  c2 = 1,  a21 = phi_1
    b1  = phi_1 - 2 phi_3,  b2 = 2 phi_3,
    bhat1 = phi_1            (embedded order-2 estimator: the exp-Rosenbrock-Euler stage)

which, in the HOS section-2.3 reformulation, costs **two** ``phi``-Krylov builds
per step:

    U2      = yn + h phi_1(hJn) Fn
    D2      = f(tn+h, U2) - Fn - Jn (U2 - yn)
    yn+1    = U2 + 2 h phi_3(hJn) D2
    uhat    = U2                              (embedded 2nd order: err = yn+1 - uhat)

so the embedded error estimate ``yn+1 - uhat`` is the ``2 h phi_3(hJn) D2``
correction itself -- free, no extra stage.

For a **non-autonomous** right-hand side the HOS section-6 treatment applies:
with ``vn = f_t(tn, yn)`` the stage gains the term ``h^2 phi_2(hJn) vn`` (a
third build; zero for autonomous problems -- skipped when ``f_t='none'`` or when
the ``f_t='fd'`` probe returns a zero ``vn``), and ``D2`` gains ``-h vn``.
``vn`` is probed by a one-sided finite difference at the same
``eps_machine^(1/3)`` scale the Rosenbrock driver uses for its
stage-time-shift-invariant ``f_t`` probe.

Properties: exact on linear problems (``f(t, u)`` linear in ``u`` gives
``yn+1 = exp(hJn) yn`` exactly and ``D2 = 0``), order 3, L-stable for the linear
part (the scalar stability function is ``exp(h lambda)``), **not** stiffly
accurate (``U2 != yn+1`` -- no first-stage reuse), and fully explicit (``Jn`` is
applied only through the ``phi`` builds, never inverted). A step costs two full
``f`` evaluations, one ``Jn`` application (to ``U2 - yn``), and two ``phi``-Krylov
builds (three with the non-autonomous ``vn`` term). Because the Jacobian is the
**full** one (not a ``linear`` accessor), EXPRB32 runs on plain callables with
no ``linear`` part -- the inverse of ETD2RK's requirement. There is deliberately
**no** ``w='linear'`` option: a ``J = L``-only frozen operator is only first
order on a genuinely nonlinear problem, the same order drop the Rosenbrock
driver documents for its ``w='linear'``.

The gradient story differs from ETD2RK's: ``Jn`` depends on ``yn`` (it is the
Jacobian of the *full* right-hand side), so the ``phi(hJn)`` actions are **not**
constant maps. The driver re-attaches the *frozen* ``Jn`` linearization instead
(each ``phi_k(hJn) v`` action with the transposed frozen operator
``phi_k(hJn)^T`` as its adjoint, by the same backward-hook mechanism), which
drops the ``dJn/du`` (Hessian) terms: the gradient is exact up to the Krylov
tolerance on **linear** problems (where ``Jn`` is constant, so nothing is
dropped). On semilinear ones the dropped terms are ``O(dt^2) * ||dJn/du||``
(one power better than the Rosenbrock frozen-``W`` ``O(dt)``): every occurrence
of the frozen operator in the step map carries a factor of ``dt`` (``h phi_1``,
``h phi_3``, and ``Jn (U2 - yn) = Jn (h phi_1(hJn) Fn)``), so differentiating
the map in ``Jn`` contributes only at ``O(dt^2)``.

Matrix-free ``phi_k(dt L) v``
-----------------------------
The linear operator ``L`` is applied **matrix-free** through the Phase 14
``linear`` accessor (a cheap ``v -> L v``, e.g. the diffusion ``nu u_xx``); no
dense matrix is ever formed. Each ``phi_k(dt L) v`` (and ``exp(dt L) v``) is a
Krylov approximation (Hochbrueck's method for ``phi_k(zA)v``): build the Arnoldi
basis from ``v`` (the same Givens-free Hessenberg process as ``jfnk.gmres``),
project ``dt L`` to the small dense Hessenberg ``H``, and apply ``phi_k`` to that
block -- ``phi_k(dt L) v ~= ||v|| V phi_k(H) e1``. ``phi_k(H) e1`` is a short
Taylor-vector series (``phi_k`` is entire), so no matrix exponential or
eigendecomposition of the large operator is needed. One Arnoldi build is used per
distinct right-hand-side vector (three per ETD2RK step), and the number of
``L``-applications is reported as the step's Krylov-iteration count.

Gradient
--------
When the input state requires a gradient, the forward runs under ``no_grad`` and
the result is re-attached to the autograd tape (the Phase 9
``_implicit_diff_reattach`` pattern). Because ``L`` is **constant** (the linear
part does not depend on ``y``), the three matrix-function actions are *constant
linear maps*: each is re-attached with its **transposed** operator as the adjoint
(``exp(dt L)^T``, ``phi_k(dt L)^T``), supplied by backward hooks -- there is no
unrolled Krylov and, unlike the Rosenbrock frozen-``W`` (which depends on ``y``
and drops a ``dW/dy`` Hessian term), **no structural approximation**. The only
adjoint error is the Krylov tolerance of the transpose ``phi``-actions, so the
gradient is exact up to that tolerance. The nonlinearity enters only through the
two ``N`` evaluations, whose VJPs are the ordinary autograd VJPs.

The transpose ``phi``-actions are computed by the same Krylov machinery on the
transposed operator ``L^T``. For the Phase 7 benchmark ``L = nu u_xx`` is
**symmetric** (the periodic central-difference Laplacian), so ``L^T = L`` and the
adjoint reuses the forward ``linear`` accessor. For a non-self-adjoint ``L`` the
adjoint would need ``L^T`` (a future extension); the self-adjoint case is what
this benchmark exercises.
"""

import math

import torch
from torch.profiler import record_function

from .fields import (
    flatten_integrated,
    get_reference_state,
    integrated_field_names,
    unflatten_integrated,
)
from .jfnk import _state_requires_grad, fd_matvec, jvp_matvec
from .rhs import resolve
from .rosenbrock import _default_ft_eps
from .specs import IntegrationResult, StageResult
from .solvers import SolveDiagnostics
from .util import (
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    split_return,
    updateStateEuler,
    updateStep,
)


# --------------------------------------------------------------------------- #
# Matrix-free phi_k(A) v via an Arnoldi (Krylov) approximation                #
# --------------------------------------------------------------------------- #

def _phi_series(H, e1, k, tol=1e-15, max_terms=500):
    """``phi_k(H) e1`` for a small dense matrix ``H`` by the Taylor series
    ``phi_k(H) = sum_{j>=0} H^j / (j+k)!``, applied to ``e1`` by a vector
    recurrence so no matrix power or matrix function is ever formed:

        term_0     = e1 / k!
        term_{j+1} = H term_j / (j + k + 1)
        phi_k(H) e1 = sum_j term_j

    ``phi_k`` is entire, so this converges for any ``H``; for the small Hessenberg
    blocks met here (``||H|| ~ ||dt L||``) it converges in a couple of dozen
    terms. ``k = 0`` recovers ``exp(H) e1``.

    The series is accurate while ``||H||`` is moderate: the terms peak near
    ``exp(||H||) / sqrt(2 pi ||H||)`` before summing back down, so for a strongly
    negative ``H`` the cancellation costs digits once ``||H||`` grows. The Phase 7
    benchmark has ``||dt L|| ~= 3`` (well inside the safe range); a genuinely
    over-stiff step (``||dt L||`` well past ~15) would need a scaling-and-squaring
    or Pade evaluation of ``phi_k(H)`` instead (a future extension, out of scope
    here).
    """
    term = e1 / math.factorial(k)
    s = term.clone()
    for j in range(max_terms):
        term = (H @ term) / (j + k + 1)
        s = s + term
        if float(torch.linalg.norm(term)) < tol * max(1.0, float(torch.linalg.norm(s))):
            break
    return s


def _arnoldi(matvec, v, m, tol):
    """Classic Arnoldi process for the matrix-free operator ``matvec`` built on
    the starting vector ``v`` -- the same Hessenberg construction as
    ``jfnk.gmres`` minus the restart / right-hand-side machinery (we only need the
    Krylov subspace and its Hessenberg projection, not a solve).

    Returns ``(V, H, norm_v, n_iter)``: ``V`` is the orthonormal Krylov basis
    (``len(V) >= n_iter``), ``H`` is the ``(m+1) x m`` Hessenberg projection with
    its first ``n_iter`` columns filled, ``norm_v = ||v||``, and ``n_iter`` the
    number of ``matvec`` applications. The residual ``H[n_iter, n_iter-1]`` is the
    Krylov residual, so the accuracy of the projected ``phi_k`` is readable from
    it.
    """
    norm_v = float(torch.linalg.norm(v))
    if norm_v == 0.0:
        return [], torch.zeros(0, 0, dtype=v.dtype, device=v.device), 0.0, 0
    m_eff = max(1, min(m, v.shape[0]))
    v0 = v / norm_v
    w0 = matvec(v0)
    scale = max(1.0, float(torch.linalg.norm(w0)))
    H = torch.zeros(m_eff + 1, m_eff, dtype=v.dtype, device=v.device)
    V = [v0]
    n_iter = 0
    for k in range(m_eff):
        w = w0 if k == 0 else matvec(V[k])
        n_iter = k + 1
        for i in range(k + 1):
            H[i, k] = torch.dot(w, V[i])
            w = w - H[i, k] * V[i]
        H[k + 1, k] = torch.linalg.norm(w)
        if float(H[k + 1, k]) < tol * scale:
            break
        V.append(w / H[k + 1, k])
    return V, H, norm_v, n_iter


def krylov_phi(matvec, v, k, *, m, tol):
    """Matrix-free ``phi_k(A) v`` where ``A`` is the linear operator ``matvec``
    applies: build the Krylov subspace from ``v`` (``_arnoldi``), project ``A`` to
    the small dense Hessenberg ``H``, and apply ``phi_k`` to that block --
    ``phi_k(A) v ~= ||v|| V phi_k(H) e1`` (Hochbrueck's Krylov approximation of
    ``phi_k``; the same subspace idea as ``gmres`` but a matrix *function* of the
    projection instead of a solve). ``k = 0`` gives ``exp(A) v``.

    Returns ``(result, n_iter)`` where ``n_iter`` is the number of ``matvec``
    applications (the cost of this action).
    """
    V, H, norm_v, n_iter = _arnoldi(matvec, v, m, tol)
    if n_iter == 0:
        return torch.zeros_like(v), 0
    Hm = H[:n_iter, :n_iter]
    e1 = torch.zeros(n_iter, dtype=v.dtype, device=v.device)
    e1[0] = 1.0
    phi_e1 = _phi_series(Hm, e1, k)
    out = torch.zeros_like(v)
    for i in range(n_iter):
        out = out + phi_e1[i] * V[i]
    return norm_v * out, n_iter


# --------------------------------------------------------------------------- #
# Differentiable re-attachment of a constant linear map                       #
# --------------------------------------------------------------------------- #

def _linear_map_reattach(x_g, value, transpose_action):
    """Re-attach the value of a *constant* linear map ``M x`` to the autograd tape.

    ``x_g`` is the (differentiable) input, ``value`` is the detached forward value
    of ``M x_g``, and ``transpose_action`` answers ``v -> M^T v``. The returned
    tensor has the exact value ``value`` and the adjoint gradient ``d/dx_g = M^T``
    (the map is a constant, so its derivative is itself, not a Hessian). This is
    the exponential-integrator analogue of the Rosenbrock frozen-operator
    re-attachment, and of ``jfnk._implicit_diff_reattach``.

    When ``x_g`` carries no gradient (e.g. the nonlinear part ``N`` is identically
    zero on a purely linear problem), the value is independent of the input and
    there is nothing to attach: return the detached ``value`` as is.
    """
    if not x_g.requires_grad:
        return value

    out = value + (x_g - x_g.detach())

    def hook(grad):
        with torch.no_grad():
            return transpose_action(grad)

    out.register_hook(hook)
    return out


# --------------------------------------------------------------------------- #
# The driver                                                                   #
# --------------------------------------------------------------------------- #

def integrateETD2RK(state, dt, f, *args, **kwargs) -> IntegrationResult:
    """One step of ETD2RK (a 2-stage, order-2, L-stable exponential integrator).

    Scheme keywords (consumed here, never forwarded to the right-hand side):

    * ``phi_dim=`` the Krylov dimension (max Arnoldi steps) for each
      ``phi_k(dt L) v`` approximation (default 30); the build stops early once the
      Krylov residual drops below ``phi_tol``.
    * ``phi_tol=`` the Arnoldi residual tolerance for the ``phi_k`` approximations
      (default 1e-10).
    * ``history=`` optional ``StepHistory`` (threaded through like the other
      drivers; ETD2RK is one-step and does not consume it).
    * ``priorStep=`` always rejected (ETD2RK is not stiffly accurate).

    The method needs the Phase 14 ``linear`` part of the semilinear split
    ``f = L·y + N``: it integrates ``L`` exactly through the exponential / ``phi``
    functions (matrix-free, via a Krylov approximation) and quadratures ``N``.
    Pass an ``RHS`` providing ``linear`` (e.g. a ``SemilinearRHS``); a plain
    callable has no ``linear`` part and is rejected *before the solve*.
    """
    name = 'ETD2RK'
    priorStep = kwargs.pop('priorStep', None)
    reject_prior_step(name, priorStep)
    phi_dim = kwargs.pop('phi_dim', 30)
    phi_tol = kwargs.pop('phi_tol', 1e-10)
    history = kwargs.pop('history', None)

    resolved = resolve(f, scheme_name=name, need_linear=True)
    linear_fn = resolved.linear
    nonlinear_fn = resolved.nonlinear

    with record_function(f"[Integration] {name}"):
        initializeSystem(state, dt, *args, **kwargs)
        tn = float(state.t)
        y_flat = flatten_integrated(state)
        zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)

        # Count full nonlinear (N) evaluations; the `linear` accessor is a cheap
        # operator and is not counted as a full right-hand-side evaluation.
        n_count = {'n': 0}

        def nonlinear_counted(system, dtp, *a, **kw):
            n_count['n'] += 1
            return nonlinear_fn(system, dtp, *a, **kw)

        def flat_k(k):
            # Flatten an update to the integrated-field flat space by applying it
            # (with unit dt) to a zero state: zero + 1*k = k.
            return flatten_integrated(updateStateEuler(zero_state, k, 1.0))

        # The linear-operator Krylov matvec: A = dt * L, applied to a flat vector by
        # handing the `linear` accessor a state whose integrated field is that
        # vector (the Phase 14 contract for applying L to an arbitrary vector, e.g.
        # a Krylov iterate).
        def Amatvec(v):
            z = unflatten_integrated(v, state)
            k, _ = split_return(linear_fn(z, dt, *args, **kwargs))
            return dt * flat_k(k)

        def n_flat(z_flat, t_val):
            # Build the stage buffer from `initializeNewState` so the non-integrated
            # fields (a shared stage counter, a neighbour-list object, ...) stay
            # *shared* with the caller's state -- the copied-fields contract the
            # other drivers rely on -- with the integrated fields set in place from
            # `z_flat`, at time `t_val`. Evaluate the nonlinear part N through the
            # full step machinery (preprocess + N + postprocess). Returns the flat
            # N, the update object, and the stage state (carrying the copied fields).
            z = state.initializeNewState(*args, **kwargs)
            z_ref = get_reference_state(z)
            offset = 0
            for nm in integrated_field_names(state):
                value = getattr(z_ref, nm)
                setattr(z_ref, nm, z_flat[offset:offset + value.numel()].reshape(value.shape))
                offset += value.numel()
            z.t = t_val
            k, r = updateStep(state, z, dt, nonlinear_counted, *args, **kwargs)
            return flat_k(k), k, z

        # --- forward pass (always detached; re-attached below if needed) --- #
        with torch.no_grad():
            # Stage 1: N_n at (tn, yn), then the exponential-Euler predictor pieces.
            c1 = n_count['n']
            N_n_flat, N_n_update, z_n = n_flat(y_flat, tn)
            rhs1 = n_count['n'] - c1  # == 1 (the N_n evaluation)
            exp_yn, it0 = krylov_phi(Amatvec, y_flat, 0, m=phi_dim, tol=phi_tol)
            phi1_Nn, it1 = krylov_phi(Amatvec, dt * N_n_flat, 1, m=phi_dim, tol=phi_tol)
            w = exp_yn + phi1_Nn

            # Stage 2: N_w at (tn+dt, w), then the corrector correction.
            c2 = n_count['n']
            N_w_flat, N_w_update, z_w = n_flat(w, tn + dt)
            rhs2 = n_count['n'] - c2  # == 1 (the N_w evaluation)
            phi2_M, it2 = krylov_phi(Amatvec, dt * (N_w_flat - N_n_flat), 2,
                                     m=phi_dim, tol=phi_tol)

            y_next_flat = exp_yn + phi1_Nn + phi2_M

            # Per-stage cost: stage 1 = N_n + (exp + phi1) Krylov; stage 2 = N_w +
            # phi2 Krylov. The `gmres_iterations` field carries the total `L`-matvec
            # count (Krylov steps of the linear operator), the ETD analogue of the
            # JFNK Krylov-iteration count; `rhs_evaluations` counts the N evals.
            iters1 = it0 + it1
            iters2 = it2

        stages = [
            StageResult(aux=None, update=N_n_update),
            StageResult(aux=None, update=N_w_update),
        ]
        solver_diagnostics = [
            SolveDiagnostics(gmres_iterations=int(iters1), rhs_evaluations=int(rhs1),
                             termination='fixed_iterations'),
            SolveDiagnostics(gmres_iterations=int(iters2), rhs_evaluations=int(rhs2),
                             termination='fixed_iterations'),
        ]

        if torch.is_grad_enabled() and _state_requires_grad(state):
            new_state = _attach_grad(
                state, tn, dt, resolved, phi_dim, phi_tol,
                y_flat, y_next_flat, exp_yn, phi1_Nn, phi2_M, *args, **kwargs)
        else:
            new_state = unflatten_integrated(y_next_flat, state)
            new_state.t = tn + dt

        # finalize: carry copied fields from the last stage that was evaluated
        # (the corrector's buffer, on which that stage's preprocess ran).
        finalizeSystem(new_state, state, dt,
                       [s.aux for s in stages], [s.update for s in stages],
                       [1.0, 0.0], *args, lastStageSystem=z_w, **kwargs)

        def _next_history():
            if history is None:
                return None
            from .history import HistoryEntry
            entry = HistoryEntry(t=tn, dt=dt, update=stages[-1].update,
                                 aux=stages[-1].aux)
            return history.pushed(entry)

        return IntegrationResult(
            state=new_state, stages=stages, error=None,
            history=_next_history(), solver_diagnostics=solver_diagnostics)


def _attach_grad(state, tn, dt, resolved, phi_dim, phi_tol,
                 y_flat, y_next_flat, exp_yn, phi1_Nn, phi2_M, *args, **kwargs):
    """Re-attach the ETD2RK step to the autograd tape (see the module docstring).

    Returns the new state with the exact forward value and the (exact up to the
    Krylov tolerance) adjoint gradient with respect to the input state.
    """
    zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)
    linear_fn = resolved.linear
    nonlinear_fn = resolved.nonlinear

    # The constant transposed linear-operator matvec: v -> dt * L^T v. For the
    # self-adjoint benchmark (L = nu u_xx, symmetric) L^T = L, so this reuses the
    # forward `linear` accessor; a non-self-adjoint L would need L^T (see the
    # module docstring).
    def Amatvec_T(v):
        z = unflatten_integrated(v, state)
        k, _ = split_return(linear_fn(z, dt, *args, **kwargs))
        return dt * flatten_integrated(updateStateEuler(zero_state, k, 1.0))

    def transpose_phi(k):
        # v -> phi_k(dt L)^T v = phi_k(dt L^T) v; self-adjoint L: = phi_k(dt L) v.
        def apply(v):
            return krylov_phi(Amatvec_T, v, k, m=phi_dim, tol=phi_tol)[0]
        return apply

    # A differentiable flat-N evaluation at (t, z); ``z_flat_g`` may require grad.
    def n_flat_g(z_flat_g, t_val):
        z = unflatten_integrated(z_flat_g, state)
        z.t = t_val
        k, _ = updateStep(state, z, dt, nonlinear_fn, *args, **kwargs)
        return flatten_integrated(updateStateEuler(zero_state, k, 1.0))

    # The differentiable input is the *actual* input state's flat vector (so the
    # gradient reaches the caller's tensors), not a detached clone.
    yn_g = flatten_integrated(state)

    N_n_g = n_flat_g(yn_g, tn)
    # The phi1 map from N_n is `dt * phi_1(hL)` (the forward folds `dt` into the
    # Krylov argument, phi_1(hL)(dt N_n) = dt phi_1(hL) N_n), so feed the reattach
    # `dt * N_n_g` -- its map is then phi_1(hL) (transpose phi_1(hL)^T), and the
    # `dt` flows back through the scalar multiply. This mirrors the phi2 term,
    # whose input M_g = dt (N_w - N_n) already carries the `dt`.
    exp_yn_g = _linear_map_reattach(yn_g, exp_yn, transpose_phi(0))
    phi1_Nn_g = _linear_map_reattach(dt * N_n_g, phi1_Nn, transpose_phi(1))
    w_g = exp_yn_g + phi1_Nn_g
    N_w_g = n_flat_g(w_g, tn + dt)
    M_g = dt * (N_w_g - N_n_g)
    phi2_M_g = _linear_map_reattach(M_g, phi2_M, transpose_phi(2))
    y_next_g = exp_yn_g + phi1_Nn_g + phi2_M_g

    # Carry y_next_g's gradient but y_next_flat's exact value.
    out_flat = y_next_flat + (y_next_g - y_next_g.detach())
    new_state = unflatten_integrated(out_flat, state)
    new_state.t = tn + dt
    return new_state


# --------------------------------------------------------------------------- #
# EXPRB32 (Hochbrueck, Ostermann & Schweitzer 2009)                           #
# --------------------------------------------------------------------------- #

def integrateEXPRB32(state, dt, f, *args, **kwargs) -> IntegrationResult:
    """One step of EXPRB32 (a 2-stage, order-3, L-stable exponential Rosenbrock
    method with an embedded order-2 estimator; see the module docstring).

    Scheme keywords (consumed here, never forwarded to the right-hand side):

    * ``w=`` the frozen operator ``Jn = D_u f(tn, yn)``: ``'jvp'`` (default, exact
      Jacobian by forward-mode AD) or ``'fd'`` (finite-difference Jacobian).
      There is deliberately no ``'linear'`` option: a ``J = L``-only frozen
      operator is only first order on a genuinely nonlinear problem (the same
      order drop the Rosenbrock driver documents for its ``w='linear'``).
    * ``f_t=`` the time derivative: ``'fd'`` (default, one-sided finite
      difference, one extra ``f`` per step) or ``'none'`` (autonomous -- no probe,
      no ``phi_2`` build).
    * ``ft_eps=`` optional override for the ``f_t`` finite-difference step.
    * ``phi_dim=`` the Krylov dimension (max Arnoldi steps) for each
      ``phi_k(h Jn) v`` approximation (default 30); the build stops early once
      the Krylov residual drops below ``phi_tol``.
    * ``phi_tol=`` the Arnoldi residual tolerance for the ``phi_k`` approximations
      (default 1e-8 -- the library's standard Krylov tolerance; the action error
      is well below the method's O(h^4) local error at the benchmark step sizes).
    * ``history=`` optional ``StepHistory`` (threaded through; one-step method).
    * ``priorStep=`` always rejected (EXPRB32 is not stiffly accurate).

    Unlike ETD2RK, the method needs no semilinear split: it freezes the **full**
    Jacobian of the right-hand side, so it runs on plain callables (a plain
    callable has no ``linear`` part and there is nothing to resolve for one).
    """
    name = 'EXPRB32'
    priorStep = kwargs.pop('priorStep', None)
    reject_prior_step(name, priorStep)
    w = kwargs.pop('w', 'jvp')
    f_t = kwargs.pop('f_t', 'fd')
    ft_eps = kwargs.pop('ft_eps', None)
    phi_dim = kwargs.pop('phi_dim', 30)
    phi_tol = kwargs.pop('phi_tol', 1e-8)
    history = kwargs.pop('history', None)
    if w not in ('jvp', 'fd'):
        raise ValueError(f"{name}: w= must be 'jvp' or 'fd', got {w!r}")
    if f_t not in ('fd', 'none'):
        raise ValueError(f"{name}: f_t= must be 'fd' or 'none', got {f_t!r}")

    resolved = resolve(f, scheme_name=name)
    combined = resolved.combined

    with record_function(f"[Integration] {name}"):
        initializeSystem(state, dt, *args, **kwargs)
        tn = float(state.t)
        y_flat = flatten_integrated(state)
        dtype = y_flat.dtype
        zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)

        f_count = {'n': 0}

        def combined_counted(system, dtp, *a, **kw):
            f_count['n'] += 1
            return combined(system, dtp, *a, **kw)

        def flat_k(k):
            # Flatten an update to the integrated-field flat space by applying it
            # (with unit dt) to a zero state: zero + 1*k = k.
            return flatten_integrated(updateStateEuler(zero_state, k, 1.0))

        def f_flat(z_flat, t_val, ffn):
            # Build the stage buffer from ``initializeNewState`` so the non-integrated
            # fields (a shared stage counter, a neighbour-list object, ...) stay *shared*
            # with the caller's state -- the same contract the DIRK/BDF drivers rely on
            # (see ``bdf._copy_integrated``). The integrated tensor fields are then set
            # in place from ``z_flat``, the flat space the Krylov builds live in.
            z = state.initializeNewState(*args, **kwargs)
            z_ref = get_reference_state(z)
            offset = 0
            for nm in integrated_field_names(state):
                value = getattr(z_ref, nm)
                setattr(z_ref, nm, z_flat[offset:offset + value.numel()].reshape(value.shape))
                offset += value.numel()
            z.t = t_val
            k, r = updateStep(state, z, dt, ffn, *args, **kwargs)
            # Return ``z`` too: ``updateStep``'s ``preprocess`` ran *on it*, in place,
            # so it is the object carrying this stage's ``copied`` fields (density,
            # pressure, ...). The caller keeps the last stage's ``z`` for finalize.
            return flat_k(k), k, r, z

        # --- the frozen operator Jn's (detached) action -------------------- #
        def _step(Y):
            # Run the full step machinery (preprocess + f + postprocess), not just
            # the bare right-hand side: the Jn-action is the Jacobian of that full
            # right-hand side, and the right-hand side needs any ``copied`` fields
            # that ``preprocess`` fills. The frozen point is the step start
            # ``(tn, un)``, so every Krylov probe re-runs ``preprocess`` on the
            # step-start buffer it is perturbing.
            Y.t = tn
            k, _ = split_return(updateStep(state, Y, dt, combined_counted, *args, **kwargs))
            return updateStateEuler(Y, k, 1.0)

        # Jact v = Jn v; jvp_matvec/fd_matvec return v -> -Jn v (their G is
        # flatten(Y) - flatten(Y + f(Y)) = -f(Y)), so negate.
        if w == 'jvp':
            _op = jvp_matvec(_step, state)

            def Jact(v):
                return -_op(v)
        else:  # 'fd'
            # G_y = -f(tn, un) is computed from the stage-1 evaluation below;
            # build the matvec once it is known.
            _fd_op = None

            def Jact(v):
                return -_fd_op(v)

        def Amatvec(v):
            # The Krylov operator A = h Jn, so a build answers phi_k(h Jn) v.
            return dt * Jact(v)

        # --- forward pass (always detached; re-attached below if needed) --- #
        with torch.no_grad():
            F_n_flat, k1, r1, z1 = f_flat(y_flat, tn, combined_counted)
            if w == 'fd':
                # fd_matvec needs G_y = flatten(Y) - flatten(step(Y)) = -F_n_flat.
                _fd_op = fd_matvec(_step, state, y_flat, -F_n_flat)

            # The time derivative vn = f_t(tn, yn) (the HOS section-6 term).
            if f_t == 'fd':
                eps_t = ft_eps if ft_eps is not None else _default_ft_eps(tn, dt, dtype)
                fte_flat, _, _, _ = f_flat(y_flat, tn + eps_t, combined_counted)
                vn_flat = (fte_flat - F_n_flat) / eps_t
            else:
                vn_flat = None

            # Stage 1: U2 = yn + h phi_1(hJn) Fn [+ h^2 phi_2(hJn) vn].
            # (The h^2 term is a third build; on an autonomous problem the 'fd'
            # probe returns vn = 0 bit-exactly and the build is skipped.)
            phi1_term, it1 = krylov_phi(Amatvec, dt * F_n_flat, 1, m=phi_dim, tol=phi_tol)
            if vn_flat is not None and float(torch.linalg.norm(vn_flat)) > 0.0:
                phi2_term, it2 = krylov_phi(Amatvec, dt * dt * vn_flat, 2,
                                            m=phi_dim, tol=phi_tol)
            else:
                phi2_term, it2 = None, 0
            U2 = y_flat + phi1_term + (phi2_term if phi2_term is not None else 0.0)
            rhs1 = 1 + (1 if f_t == 'fd' else 0)  # F_n (+ the f_t probe)

            # Stage 2: D2 = f(tn+h, U2) - Fn - Jn (U2 - yn) [- h vn]; corrector.
            FU2_flat, k2, r2, z2 = f_flat(U2, tn + dt, combined_counted)
            JU2 = Jact(U2 - y_flat)
            D2 = FU2_flat - F_n_flat - JU2
            if vn_flat is not None:
                D2 = D2 - dt * vn_flat
            rhs2 = 2  # f(U2) + the frozen-Jn application Jn (U2 - yn)

            # The corrector correction 2 h phi_3(hJn) D2 -- also the embedded
            # error estimate (u_{n+1} - uhat, with the order-2 estimate uhat = U2).
            # On a linear problem D2 is at rounding, the correction is ~ h*1e-16,
            # and the build is skipped (the gradient path is unaffected: the
            # adjoint chain through D2 vanishes on a linear graph).
            d2_scale = max(1.0, float(torch.linalg.norm(F_n_flat)))
            if float(torch.linalg.norm(D2)) <= 1e-12 * d2_scale:
                correction, it3 = torch.zeros_like(D2), 0
            else:
                correction, it3 = krylov_phi(Amatvec, 2.0 * dt * D2, 3,
                                             m=phi_dim, tol=phi_tol)
            y_next_flat = U2 + correction

        stages = [
            StageResult(aux=r1, update=k1),
            StageResult(aux=r2, update=k2),
        ]
        solver_diagnostics = [
            SolveDiagnostics(gmres_iterations=int(it1 + it2), rhs_evaluations=int(rhs1),
                             termination='fixed_iterations'),
            SolveDiagnostics(gmres_iterations=int(it3), rhs_evaluations=int(rhs2),
                             termination='fixed_iterations'),
        ]

        error_state = unflatten_integrated(correction, state)
        error_state.t = tn + dt

        if torch.is_grad_enabled() and _state_requires_grad(state):
            new_state = _attach_grad_exprb32(
                state, tn, dt, combined, f_t, ft_eps, phi_dim, phi_tol,
                y_flat, y_next_flat, F_n_flat, vn_flat, phi1_term, phi2_term,
                JU2, correction, *args, **kwargs)
        else:
            new_state = unflatten_integrated(y_next_flat, state)
            new_state.t = tn + dt

        # finalize: carry copied fields from the last stage that was evaluated
        # (the corrector's buffer, on which that stage's preprocess ran).
        finalizeSystem(new_state, state, dt,
                       [s.aux for s in stages], [s.update for s in stages],
                       [1.0, 0.0], *args, lastStageSystem=z2, **kwargs)

        def _next_history():
            if history is None:
                return None
            from .history import HistoryEntry
            entry = HistoryEntry(t=tn, dt=dt, update=stages[-1].update,
                                 aux=stages[-1].aux)
            return history.pushed(entry)

        return IntegrationResult(
            state=new_state, stages=stages, error=error_state,
            history=_next_history(), solver_diagnostics=solver_diagnostics)


def _attach_grad_exprb32(state, tn, dt, combined, f_t, ft_eps, phi_dim, phi_tol,
                         y_flat, y_next_flat, F_n_flat, vn_flat, phi1_term,
                         phi2_term, JU2, correction, *args, **kwargs):
    """Re-attach the EXPRB32 step to the autograd tape (see the module docstring).

    The frozen-``Jn`` linearization is re-attached: each ``phi_k(hJn) v`` action
    with the transposed frozen operator ``phi_k(hJn)^T`` as its adjoint (backward
    hooks), and ``Jn (U2 - yn)`` with ``Jn^T``. Exact up to the Krylov tolerance
    on **linear** problems (``Jn`` constant, nothing dropped); on semilinear ones
    the dropped ``dJn/du`` (Hessian) terms are ``O(dt^2) * ||dJn/du||`` -- every
    frozen-operator occurrence in the step map carries a factor of ``dt``, so
    this is one power better than the Rosenbrock frozen-``W`` ``O(dt)``.
    """
    dtype = y_flat.dtype
    zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)

    # A differentiable flat-f evaluation at (t, z); ``z_flat_g`` may require grad.
    def f_flat_g(z_flat_g, t_val):
        z = unflatten_integrated(z_flat_g, state)
        z.t = t_val
        k, _ = updateStep(state, z, dt, combined, *args, **kwargs)
        return flatten_integrated(updateStateEuler(zero_state, k, 1.0))

    # The frozen Jacobian's transpose Jn^T, by one VJP graph at (tn, yn) (the
    # ROS3P wT pattern): a separate leaf suffices -- the VJP only answers Jn^T v
    # at the frozen point, a constant linear map -- it does not need to connect
    # to the main tape.
    def _build_jT():
        with torch.enable_grad():
            y_var = y_flat.clone().requires_grad_(True)
            ys = unflatten_integrated(y_var, state)
            ys.t = tn
            k, _ = updateStep(state, ys, dt, combined, *args, **kwargs)
            f_wt_flat = flatten_integrated(updateStateEuler(zero_state, k, 1.0))

        def jT(v):
            (g,) = torch.autograd.grad(f_wt_flat, y_var, grad_outputs=v,
                                       retain_graph=True)
            return g

        return jT

    jT = _build_jT()

    def transpose_phi(k):
        # v -> phi_k(hJn)^T v = phi_k((hJn)^T) v = phi_k(h Jn^T) v, by the same
        # Krylov machinery on the transposed frozen operator.
        def apply(v):
            return krylov_phi(lambda x: dt * jT(x), v, k, m=phi_dim, tol=phi_tol)[0]
        return apply

    # The differentiable input is the *actual* input state's flat vector (so the
    # gradient reaches the caller's tensors), not a detached clone.
    yn_g = flatten_integrated(state)
    F_n_g = f_flat_g(yn_g, tn)
    if f_t == 'fd':
        eps_t = ft_eps if ft_eps is not None else _default_ft_eps(tn, dt, dtype)
        vn_g = (f_flat_g(yn_g, tn + eps_t) - F_n_g) / eps_t
    else:
        vn_g = None

    # The forward folds each coefficient (h, h^2, 2h) into the Krylov argument,
    # so the reattach input must be exactly what the build consumed: its map is
    # then phi_k(hJn) (transpose phi_k(hJn)^T), and the scalar flows back through
    # the multiply -- the ETD2RK reattach trick.
    phi1_g = _linear_map_reattach(dt * F_n_g, phi1_term, transpose_phi(1))
    if phi2_term is not None:
        phi2_g = _linear_map_reattach(dt * dt * vn_g, phi2_term, transpose_phi(2))
        U2_g = yn_g + phi1_g + phi2_g
    else:
        U2_g = yn_g + phi1_g

    FU2_g = f_flat_g(U2_g, tn + dt)
    JU2_g = _linear_map_reattach(U2_g - yn_g, JU2, jT)
    D2_g = FU2_g - F_n_g - JU2_g
    if vn_g is not None:
        D2_g = D2_g - dt * vn_g
    correction_g = _linear_map_reattach(2.0 * dt * D2_g, correction, transpose_phi(3))
    y_next_g = U2_g + correction_g

    # Carry y_next_g's gradient but y_next_flat's exact value.
    out_flat = y_next_flat + (y_next_g - y_next_g.detach())
    new_state = unflatten_integrated(out_flat, state)
    new_state.t = tn + dt
    return new_state
