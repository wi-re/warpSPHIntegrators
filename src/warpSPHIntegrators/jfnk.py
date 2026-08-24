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

Opt-in only (JFNK_PLAN.md A5, resolved 2026-08-24): ``FixedPointSolver`` stays the
registry default for every DIRK scheme; pass ``solver=JFNKSolver()`` to a driver
call explicitly to use this instead.
"""

import math
from typing import Callable, Optional

import torch
import torch.autograd.forward_ad as fwAD

from .fields import (
    _maybe_reference_state,
    flatten_integrated,
    integrated_field_names,
    replace_integrated_fields,
    unflatten_integrated,
)
from .solvers import SolveResult

__all__ = ['fd_matvec', 'jvp_matvec', 'gmres', 'JFNKSolver']


# --------------------------------------------------------------------------- #
# A2: generic finite-difference matvec                                        #
# --------------------------------------------------------------------------- #

def _fd_epsilon(y: torch.Tensor, v: torch.Tensor) -> float:
    """Knoll-Keyes' scaled forward-difference step: ``sqrt(eps_machine)``, scaled
    by the current iterate's own magnitude and normalized by the direction's, so
    the same formula gives a sensible ``eps`` whether ``v`` is a unit Krylov basis
    vector or something larger.
    """
    eps_machine = torch.finfo(y.dtype).eps
    v_norm = float(torch.linalg.norm(v))
    if v_norm == 0.0:
        return math.sqrt(eps_machine)
    y_norm = float(torch.linalg.norm(y))
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
    def matvec(v: torch.Tensor) -> torch.Tensor:
        h = eps if eps is not None else _fd_epsilon(y_flat, v)
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
    names = integrated_field_names(Y)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        with fwAD.dual_level():
            s = _maybe_reference_state(Y)
            replacements = {}
            offset = 0
            for name in names:
                value = getattr(s, name)
                n = value.numel()
                tangent = v[offset:offset + n].reshape(value.shape)
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
                if bool(tangent.abs().max() > 0):
                    replacements[name] = fwAD.make_dual(value, tangent)
            Y_dual = replace_integrated_fields(Y, replacements)

            result = step(Y_dual)

            result_state = _maybe_reference_state(result)
            tangent_parts = []
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
          maxiter: Optional[int] = None, restart: int = 30):
    """Restarted GMRES, no symmetry assumption (JFNK_PLAN.md A4).

    Matrix-free: ``matvec(v)`` is the only way this ever touches the operator, so
    it composes directly with ``fd_matvec``/``jvp_matvec`` above -- neither ever
    forms a dense Jacobian. Classic Arnoldi process with Givens-rotation QR
    updates of the growing Hessenberg matrix (Saad's GMRES(m)).
    ``warpSPH/modules/incompressible/krylov.py`` has a GMRES too, at the
    SPH-pressure-field abstraction level rather than this module's arbitrary
    flat-vector one -- a correctness cross-check while developing this, not
    something to import (JFNK_PLAN.md A4).

    Returns ``(x, iterations)``; ``iterations`` counts total Arnoldi steps (one
    matvec each) across every restart cycle.
    """
    n = b.shape[0]
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    b_norm = float(torch.linalg.norm(b))
    if b_norm == 0.0:
        return torch.zeros_like(b), 0
    atol = tol * max(b_norm, 1.0)
    if maxiter is None:
        maxiter = n
    m = max(1, min(restart, n))

    total_iters = 0
    while total_iters < maxiter:
        # `matvec` is always a Jacobian-vector product here (fd_matvec/jvp_matvec),
        # linear in its argument by construction, so `matvec(0) == 0` holds exactly
        # -- skip the call rather than evaluate it. This is the standard "x0 is
        # zero" GMRES shortcut, and it also sidesteps a real
        # `torch.autograd.forward_ad` incompatibility: an all-zero tangent reaching
        # `StateAwareWarpFunction.apply` (warpSPHCore) trips an internal PyTorch
        # assertion ("expected both tensor and its forward grad to be floating
        # point or complex") rather than returning a zero tangent, confirmed by
        # isolating a single dual `warpOperation(Laplacian, ...)` call with an
        # all-zero seeded tangent -- not specific to this module's matvec
        # composition. `x` is exactly zero on the first iteration of every restart
        # cycle whose initial guess was the zero vector (the default `x0`), so this
        # triggers on the very first GMRES call in the common case, not just as a
        # rare edge case.
        r = b.clone() if float(torch.linalg.norm(x)) == 0.0 else b - matvec(x)
        r_norm = torch.linalg.norm(r)
        if float(r_norm) < atol:
            return x, total_iters

        cycle = min(m, maxiter - total_iters)
        V = [r / r_norm]
        H = torch.zeros(cycle + 1, cycle, dtype=b.dtype, device=b.device)
        g = torch.zeros(cycle + 1, dtype=b.dtype, device=b.device)
        g[0] = r_norm
        cs = torch.zeros(cycle, dtype=b.dtype, device=b.device)
        sn = torch.zeros(cycle, dtype=b.dtype, device=b.device)

        k_used = 0
        for k in range(cycle):
            w = matvec(V[k])
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

        y = torch.zeros(k_used, dtype=b.dtype, device=b.device)
        for i in reversed(range(k_used)):
            denom_ii = H[i, i]
            if abs(float(denom_ii)) > 1e-300:
                y[i] = (g[i] - torch.dot(H[i, i + 1:k_used], y[i + 1:k_used])) / denom_ii
        for i in range(k_used):
            x = x + y[i] * V[i]

        if abs(float(g[k_used])) < atol:
            return x, total_iters

    return x, total_iters


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

    Opt-in only, never a registry default (JFNK_PLAN.md A5's own resolution,
    2026-08-24): ``FixedPointSolver`` already handles the non-stiff regime well
    for free (no norm, no Jacobian, no branching, CUDA-graph-capturable); this is
    for the stiff regime ``FixedPointSolver``'s own docstring says needs a
    Newton-based solver instead of more Picard iterations.

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
      ``tol``/``norm``: Newton convergence check, ``norm(Y, step(Y)) < tol`` --
        the same convention ``FixedPointSolver``'s own ``tol`` path uses. Defaults
        to ``_default_flat_norm()`` when no ``norm`` is supplied.
      ``max_iterations``: outer Newton *correction* budget (default 20) -- the
        number of ``step`` calls is this plus one (one final evaluation to verify
        the last correction, or to report on running out of budget).
      ``gmres_tol``/``gmres_maxiter``/``gmres_restart``: forwarded to ``gmres``
        (A4); ``gmres_tol`` defaults to ``tol``.
    """

    def __init__(self, matvec: str = 'fd', tol: float = 1e-8, max_iterations: int = 20,
                 gmres_tol: Optional[float] = None, gmres_maxiter: Optional[int] = None,
                 gmres_restart: int = 30, fd_eps: Optional[float] = None):
        if matvec not in ('fd', 'jvp'):
            raise ValueError(f"JFNKSolver needs matvec in ('fd', 'jvp'), got {matvec!r}")
        self.matvec = matvec
        self.tol = tol
        self.max_iterations = max_iterations
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.fd_eps = fd_eps

    def solve(self, step: Callable, y0, norm: Optional[Callable] = None, **opts) -> SolveResult:
        matvec_kind = opts.get('matvec', self.matvec)
        tol = opts.get('tol', self.tol)
        max_iterations = opts.get('max_iterations', self.max_iterations)
        gmres_tol = opts.get('gmres_tol', self.gmres_tol) or tol
        gmres_maxiter = opts.get('gmres_maxiter', self.gmres_maxiter)
        gmres_restart = opts.get('gmres_restart', self.gmres_restart)
        fd_eps = opts.get('fd_eps', self.fd_eps)
        norm_fn = norm if norm is not None else _default_flat_norm()

        Y = y0
        n = 0
        for _ in range(max_iterations):
            Y_step = step(Y)
            n += 1
            if norm_fn(Y, Y_step) < tol:
                return SolveResult(Y_step, True, n)

            y_flat = flatten_integrated(Y)
            G_y = y_flat - flatten_integrated(Y_step)
            matvec_fn = (jvp_matvec(step, Y) if matvec_kind == 'jvp'
                         else fd_matvec(step, Y, y_flat, G_y, eps=fd_eps))
            delta, _gmres_iters = gmres(
                matvec_fn, -G_y, tol=gmres_tol,
                maxiter=gmres_maxiter if gmres_maxiter is not None else y_flat.numel(),
                restart=gmres_restart,
            )
            Y = unflatten_integrated(y_flat + delta, Y)

        Y_step = step(Y)
        n += 1
        return SolveResult(Y_step, norm_fn(Y, Y_step) < tol, n)
