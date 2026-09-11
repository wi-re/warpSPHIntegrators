"""Rosenbrock-W driver: ROS3P (NOTES.md S3.14, Phase 7).

A **Rosenbrock-W** (implicit-explicit, linearly-implicit) one-step method for
**semilinear** right-hand sides ``f = L·y + N``. Unlike the DIRK/BDF drivers
(which close each stage with an outer Newton loop), every Rosenbrock-W stage is a
*single linear solve* against a **frozen** operator ``W`` (the Jacobian of ``f``
at the step start, or the ``linear`` part of the split), so there is no nonlinear
iteration: three ``gmres`` solves per step, no line search.

ROS3P
-----
The specific method is **ROS3P**, a 3-stage, order-3, **A-stable** (not
L-stable) Rosenbrock-W method with a built-in order-2 embedded estimate, from the
3-stage Rosenbrock-W construction in **Lang & Verwer, BIT 41(4) (2001) 731-738**
(the citable coefficient source). The coefficients satisfy the Rosenbrock-W order
conditions exactly (the order-3 condition ``gamma^2 - gamma + 1/6 = 0`` holds in
``Q(sqrt(3))``); the stability function satisfies ``R(inf) = 1 - sqrt(3) ~=
-0.732``, so stiff modes are damped to ``|1 - sqrt(3)|`` per step, not killed --
the method is A-stable but **not** L-stable.

The one-step form (Lang & Verwer's general Rosenbrock-W form (2.2)):

    (I - tau*gamma*W) K_i = F(tn + a_i*tau, un + tau*sum_{j<i} a_ij K_j)
                           + tau*W*sum_{j<i} g_ij K_j
                           + tau*gamma_row_i * dF/dt(tn, un)
    un+1 = un + tau * sum_i b_i K_i

with the frozen operator ``W = F_y(tn, un)`` (or the ``linear`` part). For ROS3P
the three stage *states* collapse to two distinct right-hand-side points --
``(tn, un)`` and ``(tn + tau, un + tau K1)`` (stage 3 reuses stage 2's state and
time) -- so a step costs two ``f`` evaluations (plus one more for the time
derivative when ``f_t='fd'``), not three.

The coefficients (exact in ``sqrt(3)``; ``gamma = 1/2 + sqrt(3)/6``):

    a  = (0, 1, 1)                      (stage nodes)
    a21 = 1, a31 = 1, a32 = 0
    g21 = -1, g31 = -gamma, g32 = 1/2 - 2*gamma
    gamma_row = (gamma, gamma - 1, 1/2 - 2*gamma)
    b  = (2/3, 0, 1/3)
    b_hat = (1/3, 1/3, 1/3)             (embedded order-2 estimate)

The common per-stage operator is ``M = I - tau*gamma*W`` (identical for all three
stages, so the three solves share one Krylov operator). The embedded error
estimate is ``y^{n+1} - y_hat = tau (K1 - K2) / 3``.

W (the frozen operator)
-----------------------
``w=`` selects the frozen operator used in the stage solves:

* ``'jvp'`` (default) -- the **exact** Jacobian action ``W v = J_f(tn, un) v`` by
  forward-mode AD (``jfnk.jvp_matvec``). Exact for any ``f`` built from
  warpSPHCore's wrapped operators; fails loudly otherwise.
* ``'fd'`` -- the Jacobian action by finite differences (``jfnk.fd_matvec``);
  works for any ``f``.
* ``'linear'`` -- the ``linear`` part of a ``SemilinearRHS`` (``L``) only. This
  is the fast, matrix-free option, but it **drops the nonlinear Jacobian**
  ``J_N`` from the frozen operator. It is exact (order 3) only when the right-hand
  side is *linear in the integrated state* (``J_N = 0``); on a genuinely
  semilinear problem it is a **first-order** method (verified on viscous Burgers:
  the asymptotic order drops from 3 to 1). Use it as a cheap low-order option,
  not as a stand-in for the full Jacobian.

``f_t`` (the explicit time derivative)
--------------------------------------
``f_t=`` handles the non-autonomous term ``dF/dt(tn, un)``:

* ``'fd'`` (default) -- one-sided finite difference
  ``(F(tn + eps, un) - F(tn, un)) / eps``, reusing the stage-1 evaluation
  ``F(tn, un)`` so it costs exactly one extra ``f`` per step. Order 3 is
  retained (the FD error is ``O(eps)``, below the method order for ``eps`` near
  ``machine_eps^(1/3)``).
* ``'none'`` -- treat the right-hand side as autonomous (drop the term). For an
  autonomous ``f`` this is exact and saves the extra evaluation.

Gradient
--------
When the input state requires a gradient, the forward is run under ``no_grad`` and
the result re-attached to the autograd tape (the Phase 9
``_implicit_diff_reattach`` pattern, generalized to three linear solves): each
stage solve ``K_i = M^{-1} b_i`` and each frozen-operator application ``W x`` is
re-attached as a constant linear map whose adjoint is the **transposed** operator
(``M^{-T}`` for the solves, ``W^T`` for the operator terms), supplied by backward
hooks rather than by unrolling the Krylov iterations. The nonlinearity enters
through the ``f`` evaluations (``f1`` at ``(tn, un)``, ``f2`` at
``(tn+tau, un+tau K1)``, and the ``f_t`` probe), so the adjoint needs those VJPs
plus the transposed solves. The frozen operator is held **constant** in the
adjoint (its own derivative ``dW/dy`` is a Hessian term that is dropped): this
costs ``O(tau)`` per step in the adjoint, the standard Rosenbrock-adjoint
approximation, and is why the gradient is *exact* on linear problems (``J_N = 0``
so nothing is dropped) and accurate to a few percent on nonlinear ones.
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
from .jfnk import gmres, jvp_matvec, fd_matvec, _state_requires_grad
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
# ROS3P coefficients (exact in sqrt(3); Lang & Verwer, BIT 41(4) 2001)         #
# --------------------------------------------------------------------------- #

_GAMMA = 0.5 + math.sqrt(3.0) / 6.0                 # 0.7886751345948129
_GAMMA_ROW = (_GAMMA, _GAMMA - 1.0, 0.5 - 2.0 * _GAMMA)
_G21 = -1.0
_G31 = -_GAMMA
_G32 = 0.5 - 2.0 * _GAMMA                            # == _GAMMA_ROW[2]
_B = (2.0 / 3.0, 0.0, 1.0 / 3.0)                     # propagated weights
# Embedded error y^{n+1} - y_hat = tau*sum (b_i - b_hat_i) K_i = tau (K1 - K2)/3.


def _default_ft_eps(tn: float, dt: float, dtype) -> float:
    """Forward-difference step for dF/dt: ``machine_eps^(1/3) * max(1, |dt|)``.

    Scaled by the step, not the absolute time (``tn`` is accepted for a stable
    call signature but deliberately unused): a time that is independent of ``tn``
    is what makes the probe time ``tn + eps`` translate by exactly ``dt`` from step
    to step -- the stage-time-shift invariant the generic driver tests check. A
    fixed small step is also the right FD scale for a derivative with respect to
    the time argument.
    """
    eps_machine = torch.finfo(dtype).eps
    return float(eps_machine ** (1.0 / 3.0)) * max(1.0, abs(dt))


# --------------------------------------------------------------------------- #
# Differentiable re-attachment of a frozen linear map                          #
# --------------------------------------------------------------------------- #

def _linear_map_reattach(x_g, value, transpose_action):
    """Re-attach the value of a *frozen* linear map ``L x`` to the autograd tape.

    ``x_g`` is the (differentiable) input, ``value`` is the detached forward value
    of ``L x_g``, and ``transpose_action`` answers ``v -> L^T v``. The returned
    tensor has the exact value ``value`` and the adjoint gradient ``d/dx_g = L^T``
    (the map is a constant, so its derivative is itself, not a Hessian). This is
    the linear-solve / frozen-operator analogue of ``jfnk._implicit_diff_reattach``.
    """
    out = value + (x_g - x_g.detach())

    def hook(grad):
        with torch.no_grad():
            return transpose_action(grad)

    out.register_hook(hook)
    return out


# --------------------------------------------------------------------------- #
# The driver                                                                   #
# --------------------------------------------------------------------------- #

def integrateROS3P(state, dt, f, *args, **kwargs) -> IntegrationResult:
    """One step of ROS3P (a 3-stage, order-3, A-stable Rosenbrock-W method).

    Scheme keywords (consumed here, never forwarded to the right-hand side):

    * ``w=`` the frozen operator: ``'jvp'`` (default, exact Jacobian by
      forward-mode AD), ``'fd'`` (finite-difference Jacobian), or ``'linear'``
      (the ``linear`` part of a ``SemilinearRHS`` -- fast, but only order 3 when
      the right-hand side is linear in the state; first order otherwise).
    * ``f_t=`` the time derivative: ``'fd'`` (default, one-sided finite
      difference, one extra ``f`` per step) or ``'none'`` (autonomous).
    * ``ft_eps=`` optional override for the ``f_t`` finite-difference step.
    * ``gmres_tol=`` / ``gmres_maxiter=`` / ``gmres_restart=`` the per-stage
      ``gmres`` settings.
    * ``history=`` optional ``StepHistory`` (threaded through like the DIRK
      driver).
    * ``priorStep=`` always rejected (no first-stage reuse; the last stage is not
      the step, so the scheme is not stiffly accurate).
    """
    name = 'ROS3P'
    priorStep = kwargs.pop('priorStep', None)
    reject_prior_step(name, priorStep)
    w = kwargs.pop('w', 'jvp')
    f_t = kwargs.pop('f_t', 'fd')
    ft_eps = kwargs.pop('ft_eps', None)
    gmres_tol = kwargs.pop('gmres_tol', 1e-8)
    gmres_maxiter = kwargs.pop('gmres_maxiter', None)
    gmres_restart = kwargs.pop('gmres_restart', 30)
    history = kwargs.pop('history', None)
    if w not in ('jvp', 'fd', 'linear'):
        raise ValueError(f"{name}: w= must be 'jvp', 'fd', or 'linear', got {w!r}")
    if f_t not in ('fd', 'none'):
        raise ValueError(f"{name}: f_t= must be 'fd' or 'none', got {f_t!r}")

    resolved = resolve(f, scheme_name=name, need_linear=(w == 'linear'))
    combined = resolved.combined

    with record_function(f"[Integration] {name}"):
        initializeSystem(state, dt, *args, **kwargs)
        tn = float(state.t)
        y_flat = flatten_integrated(state)
        n = y_flat.numel()
        dtype = y_flat.dtype
        zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)
        maxiter = gmres_maxiter if gmres_maxiter is not None else n

        # Count full right-hand-side evaluations (the ``linear`` accessor, used by
        # the 'linear' W, is a cheap operator and is not counted as a full f).
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
            # (see ``bdf._copy_integrated``). ``unflatten_integrated`` would clone them
            # (via ``replace_integrated_fields``/``_op``), so a ``preprocess`` that
            # mutates one in place (e.g. incrementing a shared list) would miss the
            # caller's copy and ``copied`` fields would carry the wrong stage's value.
            # The integrated tensor fields are then set in place from ``z_flat``, the
            # flat space the GMRES solves live in.
            z = state.initializeNewState(*args, **kwargs)
            z_ref = get_reference_state(z)
            offset = 0
            for name in integrated_field_names(state):
                value = getattr(z_ref, name)
                setattr(z_ref, name, z_flat[offset:offset + value.numel()].reshape(value.shape))
                offset += value.numel()
            z.t = t_val
            k, r = updateStep(state, z, dt, ffn, *args, **kwargs)
            # Return ``z`` too: ``updateStep``'s ``preprocess`` ran *on it*, in place,
            # so it is the object carrying this stage's ``copied`` fields (density,
            # pressure, ...). The caller keeps the last stage's ``z`` for finalize.
            return flat_k(k), k, r, z

        # --- the frozen operator W's (detached) action -------------------- #
        if w == 'linear':
            linear_fn = resolved.linear

            def Wact(v):
                z = unflatten_integrated(v, state)
                k, _ = split_return(linear_fn(z, dt, *args, **kwargs))
                return flat_k(k)
        else:
            def _step(Y):
                # Run the full step machinery (preprocess + f + postprocess), not just
                # the bare right-hand side: the W-action is the Jacobian of that full
                # right-hand side, and the right-hand side needs any ``copied`` fields
                # (density, pressure, ...) that ``preprocess`` fills. The frozen point
                # is the step start ``(tn, un)``, so every Krylov probe re-runs
                # ``preprocess`` on the step-start buffer it is perturbing.
                Y.t = tn
                k, _ = split_return(updateStep(state, Y, dt, combined_counted, *args, **kwargs))
                return updateStateEuler(Y, k, 1.0)

            # W v = J_f v; jvp_matvec/fd_matvec return v -> -J_f v (their G is
            # flatten(Y) - flatten(Y + f(Y)) = -f(Y)), so negate.
            if w == 'jvp':
                _op = jvp_matvec(_step, state)

                def Wact(v):
                    return -_op(v)
            else:  # 'fd'
                # G_y = -f(tn, un) is computed from the stage-1 evaluation below;
                # build the matvec once it is known.
                _fd_op = None

                def Wact(v):
                    return -_fd_op(v)

        # --- forward pass (always detached; re-attached below if needed) --- #
        with torch.no_grad():
            # Stage-1 evaluation at (tn, un).
            f1_flat, k1, r1, _ = f_flat(y_flat, tn, combined_counted)

            if w == 'fd':
                # fd_matvec needs G_y = flatten(Y) - flatten(step(Y)) = -f1_flat.
                _fd_op = fd_matvec(_step, state, y_flat, -f1_flat)

            # The time derivative dF/dt(tn, un).
            if f_t == 'fd':
                eps_t = ft_eps if ft_eps is not None else _default_ft_eps(tn, dt, dtype)
                fte_flat, _, _, _ = f_flat(y_flat, tn + eps_t, combined_counted)
                ft_flat = (fte_flat - f1_flat) / eps_t
            else:
                ft_flat = torch.zeros_like(f1_flat)

            tau = dt
            tau_gamma = dt * _GAMMA

            def matvec(v):
                return v - tau_gamma * Wact(v)

            # Stage 1:  M K1 = f1 + tau*gamma_row_1*ft
            # (the f_t term is tau*gamma_row_1*ft, NOT tau*gamma*gamma_row_1*ft --
            #  tau_gamma below is the operator's dt*gamma, a different quantity)
            c1 = f_count['n']
            b1 = f1_flat + tau * _GAMMA_ROW[0] * ft_flat
            K1, it1 = gmres(matvec, b1, tol=gmres_tol, maxiter=maxiter,
                            restart=gmres_restart)
            w1 = f_count['n'] - c1

            # Stages 2 & 3 share the state z2 = un + tau*K1 at t = tn + tau.
            z2_flat = y_flat + tau * K1
            f2_flat, k2, r2, z2_state = f_flat(z2_flat, tn + tau, combined_counted)
            WK1 = Wact(K1)

            # Stage 2:  M K2 = f2 + tau*W*(g21*K1) + tau*gamma_row_2*ft
            c = f_count['n']
            b2 = f2_flat + tau * _G21 * WK1 + tau * _GAMMA_ROW[1] * ft_flat
            K2, it2 = gmres(matvec, b2, tol=gmres_tol, maxiter=maxiter,
                            restart=gmres_restart)
            w2 = f_count['n'] - c

            # Stage 3:  M K3 = f2 + tau*W*(g31*K1 + g32*K2) + tau*gamma_row_3*ft
            c = f_count['n']
            q = _G31 * K1 + _G32 * K2
            Wq = Wact(q)
            b3 = f2_flat + tau * Wq + tau * _GAMMA_ROW[2] * ft_flat
            K3, it3 = gmres(matvec, b3, tol=gmres_tol, maxiter=maxiter,
                            restart=gmres_restart)
            w3 = f_count['n'] - c

            # Propagated step and embedded estimate.
            y_next_flat = y_flat + tau * (_B[0] * K1 + _B[2] * K3)   # b_2 = 0
            err_flat = tau * (K1 - K2) / 3.0

            # Per-stage cost: full-f evaluations attributable to each stage.
            #   stage 1: f1 (+ the f_t probe, when f_t='fd') + its Krylov operator
            #            applications -- `c1` is the f-count just before this solve,
            #            i.e. exactly f1 (+ the probe).
            #   stage 2: f2 + its Krylov operator applications.
            #   stage 3: (f2 reused, no new full-f) + its Krylov operator applications.
            # The two *RHS-side* operator applications WK1 (= W*K1) and Wq (= W*(g31 K1 +
            # g32 K2)) are full right-hand-side evaluations when the frozen operator is the
            # full Jacobian ('jvp'/'fd') -- they run ``combined_counted`` -- but cheap
            # linear-operator applications when it is the 'linear' part (``linear_fn``,
            # not counted), so count them only in the former case. Omitting them would
            # undercount ROS3P's cost by two per step against the JFNK baselines, which
            # count every full-f evaluation in their own ``rhs_evaluations``.
            w_op = 1 if w in ('jvp', 'fd') else 0
            rhs1 = c1 + w1
            rhs2 = (1 + w_op) + w2
            rhs3 = w_op + w3

        stages = [
            StageResult(aux=r1, update=k1),
            StageResult(aux=r2, update=k2),
            StageResult(aux=r2, update=k2),   # stage 3 reuses stage 2's f
        ]
        solver_diagnostics = [
            SolveDiagnostics(gmres_iterations=int(it1), rhs_evaluations=int(rhs1),
                             termination='fixed_iterations'),
            SolveDiagnostics(gmres_iterations=int(it2), rhs_evaluations=int(rhs2),
                             termination='fixed_iterations'),
            SolveDiagnostics(gmres_iterations=int(it3), rhs_evaluations=int(rhs3),
                             termination='fixed_iterations'),
        ]

        error_state = unflatten_integrated(err_flat, state)
        error_state.t = tn + tau

        if torch.is_grad_enabled() and _state_requires_grad(state):
            new_state = _attach_grad(
                state, tn, dt, combined, resolved, w, f_t, ft_eps,
                y_flat, y_next_flat, K1, K2, K3, WK1, q, Wq, *args, **kwargs)
        else:
            new_state = unflatten_integrated(y_next_flat, state)
            new_state.t = tn + tau

        # finalize: carry copied fields from the last stage that was evaluated
        # (stage 2/3's state -- the buffer ``f_flat`` returned, on which that
        # stage's ``preprocess`` ran), then run the user's hook.
        finalizeSystem(new_state, state, dt,
                       [s.aux for s in stages], [s.update for s in stages],
                       list(_B), *args, lastStageSystem=z2_state, **kwargs)

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


def _attach_grad(state, tn, dt, combined, resolved, w, f_t, ft_eps,
                 y_flat, y_next_flat, K1, K2, K3, WK1, q, Wq, *args, **kwargs):
    """Re-attach the ROS3P step to the autograd tape (see the module docstring).

    Returns the new state with the exact forward value and the (approximate,
    frozen-operator) adjoint gradient with respect to the input state.
    """
    n = y_flat.numel()
    dtype = y_flat.dtype
    tau = dt
    tau_gamma = dt * _GAMMA
    zero_state = unflatten_integrated(torch.zeros_like(y_flat), state)

    # A differentiable flat-f evaluation at (t, z); ``z_flat_g`` may require grad.
    def f_flat_g(z_flat_g, t_val):
        z = unflatten_integrated(z_flat_g, state)
        z.t = t_val
        k, _ = updateStep(state, z, dt, combined, *args, **kwargs)
        return flatten_integrated(updateStateEuler(zero_state, k, 1.0))

    # The frozen operator's transpose W^T, by one VJP graph of the operator's
    # source (full f for 'jvp'/'fd', the linear part for 'linear'). A separate
    # leaf is fine: the VJP only needs to answer J^T v at the frozen point, a
    # constant linear map -- it does not need to connect to the main tape.
    f_for_WT = resolved.linear if w == 'linear' else combined

    def _build_wT():
        with torch.enable_grad():
            y_var = y_flat.clone().requires_grad_(True)
            ys = unflatten_integrated(y_var, state)
            ys.t = tn
            k, _ = updateStep(state, ys, dt, f_for_WT, *args, **kwargs)
            f_wt_flat = flatten_integrated(updateStateEuler(zero_state, k, 1.0))

        def wT(v):
            (g,) = torch.autograd.grad(f_wt_flat, y_var, grad_outputs=v,
                                       retain_graph=True)
            return g

        return wT

    wT = _build_wT()

    def transpose_solve(v):
        # (I - tau*gamma*W^T)^{-1} v
        return gmres(lambda x: x - tau_gamma * wT(x), v, tol=1e-10,
                     maxiter=n, restart=30)[0]

    # The differentiable input is the *actual* input state's flat vector (so the
    # gradient reaches the caller's tensors), not a detached clone.
    yn_g = flatten_integrated(state)
    f1_g = f_flat_g(yn_g, tn)
    if f_t == 'fd':
        eps_t = ft_eps if ft_eps is not None else _default_ft_eps(tn, dt, dtype)
        fte_g = f_flat_g(yn_g, tn + eps_t)
        ft_g = (fte_g - f1_g) / eps_t
    else:
        ft_g = torch.zeros_like(f1_g)

    b1_g = f1_g + tau_gamma * _GAMMA_ROW[0] * ft_g
    K1_g = _linear_map_reattach(b1_g, K1, transpose_solve)

    z2_g = yn_g + tau * K1_g
    f2_g = f_flat_g(z2_g, tn + tau)
    WK1_g = _linear_map_reattach(K1_g, WK1, wT)
    b2_g = f2_g + tau * _G21 * WK1_g + tau * _GAMMA_ROW[1] * ft_g
    K2_g = _linear_map_reattach(b2_g, K2, transpose_solve)

    q_g = _G31 * K1_g + _G32 * K2_g
    Wq_g = _linear_map_reattach(q_g, Wq, wT)
    b3_g = f2_g + tau * Wq_g + tau * _GAMMA_ROW[2] * ft_g
    K3_g = _linear_map_reattach(b3_g, K3, transpose_solve)

    y_next_g = yn_g + tau * (_B[0] * K1_g + _B[2] * K3_g)

    # Carry y_next_g's gradient but y_next_flat's exact value.
    out_flat = y_next_flat + (y_next_g - y_next_g.detach())
    new_state = unflatten_integrated(out_flat, state)
    new_state.t = tn + tau
    return new_state
