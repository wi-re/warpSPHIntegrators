"""Jacobian-free Newton-Krylov solver (JFNK_PLAN.md Phase A).

``FixedPointSolver`` (``solvers.py``) is a fixed-count Picard iteration -- the
right default for the non-stiff regime, but not a stiff solver at any iteration
count (NOTES.md S3.4: it measures -9999 at 2 Picard iterations and 10**39 at 20,
same stiffness, same tableau). ``JFNKSolver`` is the rung above it on NOTES.md
S3.4's ladder: an outer inexact-Newton loop, each correction found by GMRES
(``gmres``, A4) against a Jacobian-vector product that is never formed as a dense
matrix (NOTES.md S3.4's own caution -- an SPH state has 10**6-10**7 unknowns, and
a dense per-column Jacobian would cost that many extra force evaluations *per
Newton iteration*) -- only ever applied to a single vector, either by finite
differences (``fd_matvec``, A2, works for any ``step``) or, when ``step`` is built
entirely from warpSPHCore's six JVP-wrapped operators, by an exact forward-mode
directional derivative (``jvp_matvec``, A3).

``JFNKSolver`` is the default nonlinear solver for registered DIRK and Newmark
schemes. Callers who need a fixed-depth, CUDA-graph-capturable solve can explicitly
select ``FixedPointSolver`` or ``RelaxedFixedPointSolver`` instead.
"""

import math
from typing import Callable, Optional

import torch
import torch.autograd.forward_ad as fwAD

from .fields import (
    BlockState,
    _maybe_reference_state,
    flatten_integrated,
    integrated_field_names,
    replace_integrated_fields,
    unflatten_integrated,
)
from .solvers import SolveDiagnostics, SolveResult

__all__ = ['fd_matvec', 'jvp_matvec', 'gmres', 'identity_preconditioner',
           'diagonal_preconditioner', 'JFNKSolver']


# --------------------------------------------------------------------------- #
# A2: generic finite-difference matvec                                        #
# --------------------------------------------------------------------------- #

def _fd_epsilon(y_norm: float, v: torch.Tensor) -> float:
    """Knoll-Keyes' scaled forward-difference step: ``sqrt(eps_machine)``, scaled
    by the current iterate's own magnitude and normalized by the direction's, so
    the same formula gives a sensible ``eps`` whether ``v`` is a unit Krylov basis
    vector or something larger.

    ``y_norm`` is the caller's already-computed ``float(torch.linalg.norm(y_flat))``
    -- ``y_flat`` (the Newton iterate this whole matvec belongs to) is identical
    across every Krylov iteration of the GMRES loop, so its norm is computed once
    by ``fd_matvec`` below rather than recomputed (with a fresh GPU->CPU sync)
    on every single matvec call, the JFNK-driver-level analogue of
    ``warpSPHCore``'s own hasLiveTangent sync-batching fix
    (``JFNK_DRIVER_OVERHEAD_NOTES.md`` item 2) -- ``v_norm`` below is genuinely
    call-varying (``v`` is a fresh Krylov basis vector each call) and still
    computed fresh every time.
    """
    eps_machine = torch.finfo(v.dtype).eps
    v_norm = float(torch.linalg.norm(v))
    if v_norm == 0.0:
        return math.sqrt(eps_machine)
    return math.sqrt(eps_machine) * (1.0 + y_norm) / v_norm


def fd_matvec(step: Callable, Y, y_flat: torch.Tensor, G_y: torch.Tensor,
              eps: Optional[float] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """``v -> J_G(Y) @ v`` by forward differences, where
    ``G(Y) = flatten(Y) - flatten(step(Y))``.

    ``y_flat``/``G_y`` are the caller's already-computed ``flatten_integrated(Y)``
    and ``G(Y)`` -- one ``step`` evaluation, shared across every Krylov iteration of
    the outer Newton step this matvec belongs to (NOTES.md S3.4: "one extra
    evaluation per Krylov iteration, not one per unknown"), not recomputed here.
    Works for any ``step``, no capability gate; this is the default matvec.
    """
    y_norm = None if eps is not None else float(torch.linalg.norm(y_flat))

    def matvec(v: torch.Tensor) -> torch.Tensor:
        h = eps if eps is not None else _fd_epsilon(y_norm, v)
        Y_pert = unflatten_integrated(y_flat + h * v, Y)
        G_pert = (y_flat + h * v) - flatten_integrated(step(Y_pert))
        return (G_pert - G_y) / h
    return matvec


# --------------------------------------------------------------------------- #
# A3: generic exact-JVP matvec                                                #
# --------------------------------------------------------------------------- #

def jvp_matvec(step: Callable, Y) -> Callable[[torch.Tensor], torch.Tensor]:
    """``v -> J_G(Y) @ v`` by an exact forward-mode directional derivative.

    Opens one ``torch.autograd.forward_ad.dual_level()``, seeds ``v`` as the
    tangent of every ``integrated`` field of ``Y``, runs ``step(Y)`` once inside
    the level, and reads the tangent back off the result's ``integrated`` fields.

    Exact for any ``step`` built entirely from warpSPHCore's wrapped operators
    (Density, Interpolate, Gradient, Divergence, Curl, Laplacian, Covariance) --
    the entire warp-native forward-mode surface that exists (NOTES.md S3.4's
    correction). For anything else this **must fail loudly**:
    ``StateAwareWarpFunction.jvp`` already raises ``NotImplementedError`` the
    moment a live tangent reaches an operator with no registered ``JVPSpec``
    (``warpSPHCore/autograd/operator_spec.py``), and nothing here catches that --
    catching it would turn "some terms have no JVP" into a matvec that silently
    drops them instead of one that refuses to build (JFNK_PLAN.md A3).
    """
    is_block = isinstance(_maybe_reference_state(Y), BlockState)
    if is_block:
        sub_refs = [_maybe_reference_state(sub) for sub in Y.states]
        # Flat index space is substate-major, the 'i:name' order
        # flatten_integrated uses for blocks; one entry per (substate, integrated field).
        block_fields = [(i, name) for i in range(len(sub_refs))
                        for name in integrated_field_names(sub_refs[i])]
    else:
        s = _maybe_reference_state(Y)
        names = integrated_field_names(Y)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        with fwAD.dual_level():
            values, tangents = [], []
            offset = 0
            if is_block:
                for i, name in block_fields:
                    value = getattr(sub_refs[i], name)
                    n = value.numel()
                    tangents.append(v[offset:offset + n].reshape(value.shape))
                    values.append(value)
                    offset += n
            else:
                for name in names:
                    value = getattr(s, name)
                    n = value.numel()
                    tangents.append(v[offset:offset + n].reshape(value.shape))
                    values.append(value)
                    offset += n

            # A tangent slice that is exactly zero contributes nothing to the
            # JVP by linearity, so skip `make_dual` for it and leave the field
            # primal, rather than always wrapping every integrated field.
            # This isn't only an optimization: warpSPHCore's forward-mode
            # bridge trips an internal PyTorch assertion
            # ("expected both tensor and its forward grad to be floating
            # point or complex") when a live dual tensor and an
            # all-zero-tangent dual tensor reach the same operator launch
            # together in one `dual_level()` -- confirmed directly against
            # an isolated `warpOperation` call. This is exactly the shape
            # of the wave-equation validation case (`v(0) = 0`): at the
            # first Newton iterate, `du/dt = v = 0` identically, so
            # `G(y0)`'s `u`-block is exact zero while its `v`-block is not,
            # and the naive "wrap every field" version crashes on that
            # residual's very first Krylov vector.
            #
            # Checked once, batched, for every field here (one GPU->CPU sync
            # for the whole call via one `.tolist()`) instead of one
            # `bool(...)`/sync per field -- this repo's own version of the
            # anti-pattern warpSPHCore's Fix 2 batched away inside the bridge
            # (see JFNK_DRIVER_OVERHEAD_NOTES.md item 3); same "> 0" test,
            # same per-field result, just not one sync per field.
            live = (torch.stack([t.abs().max() for t in tangents]) > 0).tolist() if tangents else []
            if is_block:
                per_sub = {i: {} for i in range(len(sub_refs))}
                for (i, name), value, tangent, isLive in zip(block_fields, values, tangents, live):
                    if isLive:
                        per_sub[i][name] = fwAD.make_dual(value, tangent)
                Y_dual = BlockState(tuple(
                    replace_integrated_fields(Y.states[i], per_sub[i]) if per_sub[i] else Y.states[i]
                    for i in range(len(sub_refs))
                ))
            else:
                replacements = {
                    name: fwAD.make_dual(value, tangent)
                    for name, value, tangent, isLive in zip(names, values, tangents, live)
                    if isLive
                }
                Y_dual = replace_integrated_fields(Y, replacements)

            result = step(Y_dual)

            result_state = _maybe_reference_state(result)
            tangent_parts = []
            if is_block:
                for i, sub in enumerate(result_state.states):
                    sub_s = _maybe_reference_state(sub)
                    for name in integrated_field_names(sub_s):
                        primal, tangent = fwAD.unpack_dual(getattr(sub_s, name))
                        tangent_parts.append((tangent if tangent is not None else torch.zeros_like(primal)).reshape(-1))
            else:
                for name in names:
                    primal, tangent = fwAD.unpack_dual(getattr(result_state, name))
                    tangent_parts.append((tangent if tangent is not None else torch.zeros_like(primal)).reshape(-1))
            Jstep_v = torch.cat(tangent_parts)
        return v - Jstep_v
    return matvec


# --------------------------------------------------------------------------- #
# A4: GMRES                                                                   #
# --------------------------------------------------------------------------- #

def gmres(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor,
          x0: Optional[torch.Tensor] = None, tol: float = 1e-8,
          maxiter: Optional[int] = None, restart: int = 30,
          preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
          preconditioning: str = 'right'):
    """Restarted GMRES, no symmetry assumption (JFNK_PLAN.md A4), with an optional
    left or right preconditioner (Phase 2).

    Matrix-free: ``matvec(v)`` is the only way this ever touches the operator, so
    it composes directly with ``fd_matvec``/``jvp_matvec`` above -- neither ever
    forms a dense Jacobian. Classic Arnoldi process with Givens-rotation QR
    updates of the growing Hessenberg matrix (Saad's GMRES(m)).
    ``warpSPH/modules/incompressible/krylov.py`` has a GMRES too, at the
    SPH-pressure-field abstraction level rather than this module's arbitrary
    flat-vector one -- a correctness cross-check while developing this, not
    something to import (JFNK_PLAN.md A4).

    ``preconditioner`` is a 1-arg *linear* operator ``v -> M(v)`` approximating
    the inverse of the operator ``matvec`` applies (``M ~= A^{-1}``). The 3-arg
    ``preconditioner(v, state, context)`` form is what ``JFNKSolver`` and callers
    supply -- ``JFNKSolver`` adapts it to this 1-arg form once per Newton iterate.
    ``preconditioning`` selects the side:
      * ``'right'`` (default): solve ``A M z = b`` for ``z``, return ``x = M z``.
        The Arnoldi operator is ``v -> A(M v)`` and the residual GMRES drives down
        is the *true* residual ``b - A x``. The standard JFNK choice: it keeps the
        residual and the returned correction in the physical space.
      * ``'left'``: solve ``M A x = M b`` for ``x`` and return it directly. The
        Arnoldi operator is ``v -> M(A v)`` and the right-hand side is ``M b``;
        the residual is the preconditioned one ``M (b - A x)``.
    With ``preconditioner=None`` (the default) neither wrap is applied and the
    result is bitwise identical to the unpreconditioned GMRES, so existing callers
    are unaffected. A supplied *identity* preconditioner is likewise bitwise
    identical in both modes -- that equivalence is a regression check in
    ``tests/test_preconditioner.py``.

    Returns ``(x, iterations)``; ``iterations`` counts total Arnoldi steps (one
    matvec each) across every restart cycle.
    """
    if preconditioner is None:
        op = matvec
        rhs = b

        def _finalize(z: torch.Tensor) -> torch.Tensor:
            return z
    elif preconditioning == 'right':
        # x = M z, so A M z = b; the Arnoldi operator is A o M, RHS is b, and the
        # answer is x = M z (one extra preconditioner apply at the end).
        def op(v: torch.Tensor) -> torch.Tensor:
            return matvec(preconditioner(v))
        rhs = b
        _finalize = preconditioner
    elif preconditioning == 'left':
        # M A x = M b; the Arnoldi operator is M o A, RHS is M b, x returned as-is.
        def op(v: torch.Tensor) -> torch.Tensor:
            return preconditioner(matvec(v))
        rhs = preconditioner(b)

        def _finalize(z: torch.Tensor) -> torch.Tensor:
            return z
    else:
        raise ValueError(f"gmres preconditioning must be 'left' or 'right', got {preconditioning!r}")

    n = rhs.shape[0]
    x = x0.clone() if x0 is not None else torch.zeros_like(rhs)
    rhs_norm = float(torch.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return torch.zeros_like(rhs), 0
    atol = tol * max(rhs_norm, 1.0)
    if maxiter is None:
        maxiter = n
    m = max(1, min(restart, n))

    total_iters = 0
    while total_iters < maxiter:
        # `op` is a Jacobian-vector product (fd_matvec/jvp_matvec) composed with a
        # linear preconditioner, so `op(0) == 0` holds exactly -- skip the call
        # rather than evaluate it. This is the standard "x0 is zero" GMRES
        # shortcut, and it also sidesteps a real `torch.autograd.forward_ad`
        # incompatibility: an all-zero tangent reaching
        # `StateAwareWarpFunction.apply` (warpSPHCore) trips an internal PyTorch
        # assertion ("expected both tensor and its forward grad to be floating
        # point or complex") rather than returning a zero tangent, confirmed by
        # isolating a single dual `warpOperation(Laplacian, ...)` call with an
        # all-zero seeded tangent -- not specific to this module's matvec
        # composition. `x` is exactly zero on the first iteration of every restart
        # cycle whose initial guess was the zero vector (the default `x0`), so this
        # triggers on the very first GMRES call in the common case, not just as a
        # rare edge case.
        r = rhs.clone() if float(torch.linalg.norm(x)) == 0.0 else rhs - op(x)
        r_norm = torch.linalg.norm(r)
        if float(r_norm) < atol:
            return _finalize(x), total_iters

        cycle = min(m, maxiter - total_iters)
        V = [r / r_norm]
        H = torch.zeros(cycle + 1, cycle, dtype=rhs.dtype, device=rhs.device)
        g = torch.zeros(cycle + 1, dtype=rhs.dtype, device=rhs.device)
        g[0] = r_norm
        cs = torch.zeros(cycle, dtype=rhs.dtype, device=rhs.device)
        sn = torch.zeros(cycle, dtype=rhs.dtype, device=rhs.device)

        k_used = 0
        for k in range(cycle):
            w = op(V[k])
            for i in range(k + 1):
                H[i, k] = torch.dot(w, V[i])
                w = w - H[i, k] * V[i]
            H[k + 1, k] = torch.linalg.norm(w)
            total_iters += 1
            k_used = k + 1
            V.append(w / H[k + 1, k] if float(H[k + 1, k]) > 1e-14 else torch.zeros_like(w))

            for i in range(k):
                temp = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = temp
            denom = torch.sqrt(H[k, k] ** 2 + H[k + 1, k] ** 2)
            if float(denom) == 0.0:
                cs[k], sn[k] = 1.0, 0.0
            else:
                cs[k] = H[k, k] / denom
                sn[k] = H[k + 1, k] / denom
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]

            if abs(float(g[k + 1])) < atol or total_iters >= maxiter:
                break

        y = torch.zeros(k_used, dtype=rhs.dtype, device=rhs.device)
        for i in reversed(range(k_used)):
            denom_ii = H[i, i]
            if abs(float(denom_ii)) > 1e-300:
                y[i] = (g[i] - torch.dot(H[i, i + 1:k_used], y[i + 1:k_used])) / denom_ii
        for i in range(k_used):
            x = x + y[i] * V[i]

        if abs(float(g[k_used])) < atol:
            return _finalize(x), total_iters

    return _finalize(x), total_iters


# --------------------------------------------------------------------------- #
# A4b: preconditioner helpers                                                 #
# --------------------------------------------------------------------------- #

def identity_preconditioner(v: torch.Tensor, state, context) -> torch.Tensor:
    """The identity preconditioner ``v -> v`` (the 3-arg ``JFNKSolver`` contract).

    The baseline that must reproduce the unpreconditioned solve exactly: passing
    this to ``JFNKSolver``/``gmres`` (either side) is bitwise identical to passing
    no preconditioner at all. Useful to exercise the preconditioning plumbing
    without changing the operator, and as the reference in
    ``tests/test_preconditioner.py``.
    """
    return v


def diagonal_preconditioner(inv_diag):
    """A diagonal preconditioner factory (the cheapest useful stiff-term hook).

    ``inv_diag`` is either a per-DOF ``torch.Tensor`` (a *fixed* diagonal inverse)
    or a callable ``inv_diag(state, context) -> torch.Tensor`` (a state-dependent
    diagonal, e.g. ``1 / (1 + dt * damping)`` for a relaxation term). The returned
    preconditioner applies it elementwise in the flattened integrated-field space:
    ``preconditioner(v, state, context) = v * inv_diag``.

    ``inv_diag`` must have the same shape as the flattened integrated state the
    GMRES loop operates on; build it by concatenating a per-field diagonal in the
    same field order ``integrated_field_names`` uses. For diffusion/relaxation
    terms the operator's own diagonal already captures most of the stiffness, so
    this is a strong preconditioner at negligible cost (one elementwise multiply).
    """
    if torch.is_tensor(inv_diag):
        def preconditioner(v: torch.Tensor, state, context) -> torch.Tensor:
            return v * inv_diag
    else:
        def preconditioner(v: torch.Tensor, state, context) -> torch.Tensor:
            return v * inv_diag(state, context)
    return preconditioner


# --------------------------------------------------------------------------- #
# A5: JFNKSolver                                                              #
# --------------------------------------------------------------------------- #

def _default_flat_norm() -> Callable:
    """``(a, b) -> ||flatten(a) - flatten(b)|| / max(||flatten(a)||, 1)``.

    Used only when the caller passes no ``norm`` of its own. ``state_norm``'s
    Hairer-Wanner weighted RMS (what ``dirk.py``'s own default norm builds) is a
    different, per-element-weighted convention -- this one instead matches
    ``test_implicitWaveEquation.py``'s hand-rolled CG reference's own ``tol``
    convention (``atol = tol * max(||b||, 1)``), so ``JFNKSolver``'s default
    ``tol`` means the same thing whether or not a caller supplies ``norm``.
    """
    def norm(a, b) -> float:
        diff = flatten_integrated(a) - flatten_integrated(b)
        scale = max(float(torch.linalg.norm(flatten_integrated(a))), 1.0)
        return float(torch.linalg.norm(diff)) / scale
    return norm


class JFNKSolver:
    """Jacobian-free Newton-Krylov: an outer inexact-Newton loop over GMRES (A4)
    against a matrix-free Jacobian-vector product (A2 finite-difference by
    default, A3 exact-JVP opt-in). Implements the ``NonlinearSolver`` protocol
    (NOTES.md S4) so a DIRK driver can swap it in for ``FixedPointSolver`` with no
    other change (JFNK_PLAN.md A5).

    The default solver for registered DIRK and Newmark schemes. ``FixedPointSolver``
    remains available for a fixed-depth, CUDA-graph-capturable non-stiff solve, but
    this Newton-based path is required once the Picard map is no longer contractive.

    Each outer iteration evaluates ``step(Y)`` once, checks convergence, and (if
    not yet converged) computes one Newton correction via GMRES against a
    matvec of ``G(Y) = flatten(Y) - flatten(step(Y))``'s Jacobian. For an exactly
    linear ``step`` (e.g. the wave equation's stage operator) this converges in a
    single correction -- Newton on a linear residual is exact, modulo GMRES's own
    tolerance.

    ``solve``'s ``**opts`` (also settable at construction) select:
      ``matvec``: ``'fd'`` (default, works for any ``step``) or ``'jvp'`` (exact,
        only for a ``step`` built entirely from warpSPHCore's wrapped operators --
        see ``jvp_matvec``'s docstring for what happens otherwise).
      ``tol``: GMRES's own inner linear-solve tolerance (``gmres_tol`` below
        defaults to it) -- unrelated to whatever convention the caller's own
        ``norm`` uses, since GMRES always operates on a raw, unnormalized
        residual vector regardless of what ``norm`` measures.
      ``newton_tol``/``norm``: Newton's own outer convergence check,
        ``norm(Y, step(Y)) < newton_tol``. Defaults to ``tol`` when not given
        (preserving, unchanged, the ``_default_flat_norm()`` case below --
        that norm's convention is a raw relative residual, the same
        convention ``tol`` was already tuned for). A caller supplying its own
        ``norm`` with a *different* convention (e.g. ``dirk.py``'s Hairer-
        Wanner weighted-RMS, whose own convention is "< 1.0 means converged"
        -- see ``fields.state_norm``) must supply a matching ``newton_tol``
        explicitly; ``dirk.py`` does (``1e-3``, found by bisecting against
        this repo's own test suite). Found the hard way: before this
        parameter existed, ``dirk.py`` always passed its own ``norm`` but
        never a compatible ``tol``, so ``JFNKSolver``'s ``tol`` (tuned for
        the *other* convention) silently leaked through -- Newton would reach
        genuine convergence (``norm_fn`` well under 1.0) within 1-2
        corrections but never trip the check, and burn its entire
        ``max_iterations`` budget on every single stage solve chasing a
        threshold the weighted-RMS norm can't satisfy past float32 noise
        (see ``warpSPHIntegrators``'s ``JFNK_DRIVER_OVERHEAD_NOTES.md``).
      ``newton_stagnation_ratio``/``newton_stagnation_patience``: a *second*,
        resolution-independent early exit -- ``newton_tol`` alone is not
        enough, because the float32 noise floor ``norm_fn`` actually reaches
        is itself resolution-dependent (more particles -> more summed
        round-off in each RHS evaluation's reductions -> a *higher* floor),
        so no single fixed ``newton_tol`` sits below every problem size's
        floor (found directly: ~1.5e-4 at 16K particles, comfortably under
        the ``1e-3`` default; ~2.3e-3-2.8e-3 at 1M particles, comfortably
        *over* it -- both are genuine convergence, just to different floors).
        Tracked regardless of ``newton_tol``'s convention, so it needs no
        caller-side tuning: if ``norm_fn`` fails to improve by at least a
        factor of ``newton_stagnation_ratio`` (default ``0.9``, i.e. less
        than 10% better) against *both* its best value so far and the
        immediately preceding iterate's value, for
        ``newton_stagnation_patience`` (default ``2``) consecutive
        iterations, that plateau *is* this solve's floor -- reported as
        converged (this is the best available answer, not a failure), same
        as tripping ``newton_tol`` would be. Only evaluated once at least
        one non-``newton_tol`` iteration has a prior norm to compare
        against, so it can never trigger on the very first correction,
        before any progress has been measured at all. The preceding-iterate
        half of the check is what keeps a transient Newton overshoot -- the
        residual rising for a step or two as the iterates wander outside the
        quadratic-convergence basin, then falling again -- from reading as a
        floor: the overshoot's successor improves on it well past the ratio
        and resets the count, while a true floor wobbles at most round-off
        scale around the same value.
      ``line_search`` (default ``False``): backtracking on the trial residual.
        When the full GMRES correction would *increase* the nonlinear residual
        -- a badly scaled or noise-dominated matvec does this -- the correction
        is damped (alpha halved, one extra ``step`` evaluation per trial) until
        the residual drops. When no damping down to ``line_search_min_step``
        produces a decrease, the best iterate measured so far is reported with
        ``termination='stagnation'`` (converged) -- the line-search-side twin
        of the stagnation-floor policy the no-line-search path applies to a
        non-improving plateau -- instead of taking a known-worse step.
        Rejections are counted in ``SolveDiagnostics.line_search_backtracks``;
        trial evaluations count toward ``rhs_evaluations``, so the cost model
        ``total step evaluations = rhs_evaluations + gmres_iterations`` holds
        with or without it.
      ``line_search_min_step`` (default ``1e-2``): smallest damping alpha tried.
      ``max_iterations``: outer Newton *correction* budget (default 20) -- the
        number of ``step`` calls is this plus one (one final evaluation to verify
        the last correction, or to report on running out of budget).
      ``gmres_tol``/``gmres_maxiter``/``gmres_restart``: forwarded to ``gmres``
        (A4); ``gmres_tol`` defaults to ``tol``.
      ``preconditioner`` (default ``None``): an optional
        ``preconditioner(v, state, context) -> vector`` callable applied inside
        the GMRES loop to cluster the stage-Jacobian spectrum and cut Krylov
        iterations (Phase 2). ``state`` is the current Newton iterate and
        ``context`` is ``{'state': state, **preconditioner_context}``. ``None``
        (the default) leaves the solve unpreconditioned -- bitwise identical to
        the pre-hook behaviour. See ``gmres`` for the left/right formulation and
        ``identity_preconditioner``/``diagonal_preconditioner`` for ready-made
        examples.
      ``preconditioning`` (default ``'right'``): ``'right'`` solves
        ``A M z = b`` and returns ``x = M z``; ``'left'`` solves
        ``M A x = M b`` and returns ``x``.
      ``preconditioner_context`` (default ``None``): extra keys merged into the
        ``context`` dict the preconditioner receives -- e.g. ``dt``, the system,
        or an operator it needs.
    """

    def __init__(self, matvec: str = 'fd', tol: float = 1e-8, max_iterations: int = 20,
                 gmres_tol: Optional[float] = None, gmres_maxiter: Optional[int] = None,
                 gmres_restart: int = 30, fd_eps: Optional[float] = None,
                 newton_tol: Optional[float] = None,
                 newton_stagnation_ratio: float = 0.9,
                 newton_stagnation_patience: int = 2,
                 line_search: bool = False,
                 line_search_min_step: float = 1e-2,
                 preconditioner: Optional[Callable] = None,
                 preconditioning: str = 'right',
                 preconditioner_context: Optional[dict] = None):
        if matvec not in ('fd', 'jvp'):
            raise ValueError(f"JFNKSolver needs matvec in ('fd', 'jvp'), got {matvec!r}")
        if tol <= 0.0:
            raise ValueError(f'JFNKSolver needs tol > 0, got {tol}')
        if max_iterations < 1:
            raise ValueError(f'JFNKSolver needs max_iterations >= 1, got {max_iterations}')
        if not 0.0 < newton_stagnation_ratio <= 1.0:
            raise ValueError(
                'JFNKSolver needs newton_stagnation_ratio in (0, 1], '
                f'got {newton_stagnation_ratio}'
            )
        if newton_stagnation_patience < 1:
            raise ValueError(
                'JFNKSolver needs newton_stagnation_patience >= 1, '
                f'got {newton_stagnation_patience}'
            )
        if not 0.0 < line_search_min_step <= 1.0:
            raise ValueError(
                'JFNKSolver needs line_search_min_step in (0, 1], '
                f'got {line_search_min_step}'
            )
        if preconditioning not in ('left', 'right'):
            raise ValueError(
                f"JFNKSolver needs preconditioning in ('left', 'right'), "
                f'got {preconditioning!r}'
            )
        self.matvec = matvec
        self.tol = tol
        self.max_iterations = max_iterations
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.fd_eps = fd_eps
        self.newton_tol = newton_tol
        self.newton_stagnation_ratio = newton_stagnation_ratio
        self.newton_stagnation_patience = newton_stagnation_patience
        self.line_search = line_search
        self.line_search_min_step = line_search_min_step
        self.preconditioner = preconditioner
        self.preconditioning = preconditioning
        self.preconditioner_context = preconditioner_context

    def _solve_core(self, step: Callable, y0, norm: Optional[Callable] = None, **opts) -> SolveResult:
        matvec_kind = opts.get('matvec', self.matvec)
        tol = opts.get('tol', self.tol)
        max_iterations = opts.get('max_iterations', self.max_iterations)
        gmres_tol = opts.get('gmres_tol', self.gmres_tol) or tol
        gmres_maxiter = opts.get('gmres_maxiter', self.gmres_maxiter)
        gmres_restart = opts.get('gmres_restart', self.gmres_restart)
        fd_eps = opts.get('fd_eps', self.fd_eps)
        newton_tol = opts.get('newton_tol', self.newton_tol)
        if newton_tol is None:
            newton_tol = tol
        stagnation_ratio = opts.get('newton_stagnation_ratio', self.newton_stagnation_ratio)
        stagnation_patience = opts.get('newton_stagnation_patience', self.newton_stagnation_patience)
        line_search = bool(opts.get('line_search', self.line_search))
        line_search_min_step = opts.get('line_search_min_step', self.line_search_min_step)
        preconditioner = opts.get('preconditioner', self.preconditioner)
        preconditioning = opts.get('preconditioning', self.preconditioning)
        preconditioner_context = dict(
            opts.get('preconditioner_context', self.preconditioner_context) or {}
        )
        norm_fn = norm if norm is not None else _default_flat_norm()

        Y = y0
        n = 0
        total_gmres_iterations = 0
        best_norm = None
        last_norm = None
        stagnant_count = 0
        backtracks = 0
        for _ in range(max_iterations):
            Y_step = step(Y)
            n += 1
            nv = norm_fn(Y, Y_step)
            if not math.isfinite(nv):
                return SolveResult(Y, False, n, SolveDiagnostics(
                    residual=nv, gmres_iterations=total_gmres_iterations,
                    rhs_evaluations=n, line_search_backtracks=backtracks,
                    termination='invalid_residual'))
            if nv < newton_tol:
                return SolveResult(Y_step, True, n, SolveDiagnostics(
                    residual=nv, gmres_iterations=total_gmres_iterations,
                    rhs_evaluations=n, line_search_backtracks=backtracks,
                    termination='tolerance'))

            if best_norm is not None and nv >= max(best_norm, last_norm) * stagnation_ratio:
                stagnant_count += 1
                if stagnant_count >= stagnation_patience:
                    # Not "gave up" -- this plateau is this problem's
                    # (resolution- and precision-dependent) floor, and
                    # nv is already the best correction reached; further
                    # iterations only re-measure the same noise. The check is
                    # against the *worse* of the best-so-far and the previous
                    # iterate's residual: a transient Newton overshoot (the
                    # residual rising, then falling again as the iterates
                    # return to the convergence basin) improves on its
                    # predecessor well past the ratio and resets the count,
                    # while a true floor wobbles at most round-off scale
                    # around the same value.
                    return SolveResult(Y_step, True, n, SolveDiagnostics(
                        residual=nv, gmres_iterations=total_gmres_iterations,
                        rhs_evaluations=n, line_search_backtracks=backtracks,
                        termination='stagnation'))
            else:
                stagnant_count = 0
            best_norm = min(best_norm, nv) if best_norm is not None else nv
            last_norm = nv

            y_flat = flatten_integrated(Y)
            G_y = y_flat - flatten_integrated(Y_step)
            matvec_fn = (jvp_matvec(step, Y) if matvec_kind == 'jvp'
                         else fd_matvec(step, Y, y_flat, G_y, eps=fd_eps))
            if preconditioner is not None:
                # Adapt the caller's 3-arg preconditioner(v, state, context) to
                # the 1-arg operator gmres consumes, bound to this iterate: the
                # preconditioner approximates J_G(Y)^{-1} at the current Newton
                # iterate Y. gmres is called synchronously in this iteration, so
                # the closure over Y / context is stable for the whole call.
                context = {'state': Y, **preconditioner_context}
                _gmres_prec = lambda v: preconditioner(v, Y, context)
            else:
                _gmres_prec = None
            delta, _gmres_iters = gmres(
                matvec_fn, -G_y, tol=gmres_tol,
                maxiter=gmres_maxiter if gmres_maxiter is not None else y_flat.numel(),
                restart=gmres_restart,
                preconditioner=_gmres_prec,
                preconditioning=preconditioning,
            )
            total_gmres_iterations += _gmres_iters
            if not line_search:
                Y = unflatten_integrated(y_flat + delta, Y)
                continue
            # Backtracking on the trial residual: shrink the Newton correction
            # until the residual actually drops, instead of accepting a full
            # correction that makes the problem worse (the regime a badly
            # scaled or noise-dominated matvec produces). Each trial is one
            # extra ``step`` evaluation, so it counts toward ``n`` exactly like
            # the base residual does -- the documented cost model keeps
            # holding: total step evaluations = rhs_evaluations +
            # gmres_iterations (NOTES.md S3.4).
            alpha = 1.0
            while True:
                Y_trial = unflatten_integrated(y_flat + alpha * delta, Y)
                Y_trial_step = step(Y_trial)
                n += 1
                nv_trial = norm_fn(Y_trial, Y_trial_step)
                if math.isfinite(nv_trial) and nv_trial < nv:
                    Y = Y_trial
                    break
                alpha *= 0.5
                backtracks += 1
                if alpha < line_search_min_step:
                    # No damping produced a decrease. The current iterate is
                    # the best one measured so far (acceptance is strict
                    # decrease), so that plateau *is* this solve's floor --
                    # the same reading the no-line-search path gives a
                    # non-improving plateau (the 'stagnation' exit above):
                    # the best available answer, not a failure. That covers
                    # both a genuinely converged answer at the precision
                    # floor (float32 Newton: the correction falls below
                    # ulp/2 and no damping can decrease the residual) and a
                    # matvec bad enough that no damped step helps.
                    return SolveResult(Y_step, True, n, SolveDiagnostics(
                        residual=nv, gmres_iterations=total_gmres_iterations,
                        rhs_evaluations=n,
                        termination='stagnation',
                        line_search_backtracks=backtracks))

        Y_step = step(Y)
        n += 1
        residual = norm_fn(Y, Y_step)
        return SolveResult(Y_step, residual < newton_tol, n, SolveDiagnostics(
            residual=residual, gmres_iterations=total_gmres_iterations,
            rhs_evaluations=n, line_search_backtracks=backtracks,
            termination='tolerance' if residual < newton_tol else 'max_iterations'))

    def solve(self, step: Callable, y0, norm: Optional[Callable] = None, **opts) -> SolveResult:
        """Solve `y == step(y)`, differentiably when the caller's state carries grad.

        The Newton/GMRES iteration itself always runs under `torch.no_grad()`. That
        is not only an optimisation: `gmres` builds its Hessenberg factor `H`, its
        Givens rotations and its back-substitution vector by *in-place* element
        writes, so recording it on the autograd tape used to make any implicit
        scheme raise "one of the variables needed for gradient computation has been
        modified by an inplace operation" the moment a caller asked for a gradient
        (every DIRK/Newmark/IMEX/ARK scheme on its first step; BDF/AM as soon as the
        explicit cold start handed over). Unrolling the solver would also be the
        *wrong* derivative to take: it differentiates the path to the fixed point
        rather than the fixed point, and with `matvec='fd'` it would differentiate a
        divided difference.

        Gradients are re-attached instead by the implicit function theorem, which is
        exact at a converged fixed point and independent of how many iterations
        reaching it took. With `G(y, theta) = y - step(y, theta)` and `G(y*, theta) = 0`,

            dy*/dtheta = (I - J)^-1 dstep/dtheta,   J = dstep/dy at y*

        so a cotangent `g` arriving at `y*` must be mapped to
        `lambda = (I - J^T)^-1 g` before it is propagated into `step`'s own inputs.
        That transpose solve is one more matrix-free GMRES, using vector-Jacobian
        products (ordinary reverse-mode `torch.autograd.grad` through a single
        `step` application) in place of the forward matvec -- no dense Jacobian,
        matching the forward solver's own cost model.

        Cost when a gradient is actually requested: two extra `step` evaluations in
        the forward pass (see `_implicit_diff_reattach` for why the adjoint needs a
        graph of its own), and one VJP per Krylov iteration of the adjoint solve in
        the backward pass. When no input requires grad -- every non-differentiable
        use, including all of the stiff benchmarks -- this is exactly the old code
        path plus one `requires_grad` scan, and `_solve_core`'s diagnostics are
        returned unchanged.
        """
        differentiable = torch.is_grad_enabled() and _state_requires_grad(y0)
        with torch.no_grad():
            result = self._solve_core(step, y0, norm, **opts)
        if not differentiable:
            return result

        tol = opts.get('tol', self.tol)
        y_attached = _implicit_diff_reattach(
            step, result.y,
            gmres_tol=opts.get('gmres_tol', self.gmres_tol) or tol,
            gmres_maxiter=opts.get('gmres_maxiter', self.gmres_maxiter),
            gmres_restart=opts.get('gmres_restart', self.gmres_restart),
        )
        return SolveResult(y_attached, result.converged, result.iterations, result.diagnostics)


def _state_requires_grad(state) -> bool:
    """True if any `integrated` field of `state` is on the autograd tape."""
    if isinstance(_maybe_reference_state(state), BlockState):
        return any(_state_requires_grad(sub) for sub in state.states)
    s = _maybe_reference_state(state)
    for name in integrated_field_names(state):
        value = getattr(s, name, None)
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return True
    return False


def _implicit_diff_reattach(step: Callable, Y_star, *, gmres_tol: float,
                            gmres_maxiter: Optional[int], gmres_restart: int):
    """Put `Y_star` back on the autograd tape as the fixed point it is.

    See `JFNKSolver.solve` for the derivation. `Y_star` arrives detached (the solve
    ran under `no_grad`); the returned state is a differentiable function of
    whatever `step` closes over, with the `(I - J^T)^-1` factor supplied by a
    backward hook rather than by unrolling.
    """
    y_star = flatten_integrated(Y_star).detach()

    with torch.enable_grad():
        # Two *separate* applications of `step`, deliberately not one:
        #
        #   `out`     holds the theta-path alone (`y_star` enters detached), and is
        #             the tensor returned and hooked.
        #   `out_vjp` holds the y-path, and exists only to answer `J^T v`.
        #
        # They cannot be the same graph. A backward hook on `out` runs while the
        # autograd engine is inside `out`'s own node, so calling
        # `torch.autograd.grad(out, ...)` from that hook re-enters the node the
        # engine is already holding and deadlocks -- silently, with no error, just
        # a hang. Evaluating `step` a second time gives the adjoint solve a graph
        # of its own to traverse, which is ordinary supported reentrant autograd.
        out = flatten_integrated(step(unflatten_integrated(y_star, Y_star)))
        y_var = y_star.clone().requires_grad_(True)
        out_vjp = flatten_integrated(step(unflatten_integrated(y_var, Y_star)))

    if not out.requires_grad:
        # `step` turned out not to depend differentiably on anything after all
        # (e.g. every upstream tensor was detached inside the rhs). Nothing to
        # re-attach, and no adjoint solve worth running.
        return Y_star

    def jacobian_transpose(v: torch.Tensor) -> torch.Tensor:
        (g,) = torch.autograd.grad(out_vjp, y_var, grad_outputs=v, retain_graph=True)
        return g

    def adjoint_hook(grad: torch.Tensor) -> torch.Tensor:
        # Solve (I - J^T) lambda = grad. `gmres` mutates its own workspace in
        # place, so it must not be recorded; `torch.autograd.grad` inside
        # `jacobian_transpose` still works under `no_grad` because it replays an
        # already-built graph rather than extending one.
        with torch.no_grad():
            lam, _ = gmres(
                lambda v: v - jacobian_transpose(v), grad,
                tol=gmres_tol,
                maxiter=gmres_maxiter if gmres_maxiter is not None else grad.numel(),
                restart=gmres_restart,
            )
        return lam

    out.register_hook(adjoint_hook)
    # Carry `out`'s *gradient* but `y_star`'s *value*. `out` is one further
    # application of `step` than the solver actually returned, so returning it
    # directly would move the answer by the nonlinear residual (~1e-11 on a
    # converged solve) purely as a side effect of asking for a gradient. Adding a
    # numerically-zero `out - out.detach()` keeps the trajectory bit-for-bit
    # identical to the no-grad path while leaving the autograd graph intact --
    # `tests/test_gradients.py` pins both halves of that.
    value_preserving = y_star + (out - out.detach())
    return unflatten_integrated(value_preserving, Y_star)
