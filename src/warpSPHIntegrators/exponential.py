"""Exponential integrators: ETD2RK (NOTES.md S3.15, Phase 7).

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
from .jfnk import _state_requires_grad
from .rhs import resolve
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
