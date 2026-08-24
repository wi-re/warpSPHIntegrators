"""Pluggable stage-equation solvers (NOTES.md S4).

An implicit scheme's stage equation -- `Y = g(Y)` for a DIRK stage, more generally
`F(Y) = 0` for a residual -- needs something to drive `Y` towards a solution.
`NonlinearSolver` is the one interface every rung of the ladder in NOTES.md S3.4
implements: fixed-count Picard today (`FixedPointSolver`, all Phase 2 ships), JFNK
with finite-difference matvecs and a user-supplied `solve_linear` hook later. A driver
is written once against the protocol and gets a solver swapped under it without a
rewrite -- that pluggability, not any one solver, is the point of landing this now.
"""

from typing import Any, Callable, NamedTuple, Optional, Protocol, runtime_checkable


class SolveResult(NamedTuple):
    """What a `NonlinearSolver.solve` call produced."""

    y: Any
    #: See each solver's own docstring for what this claims -- a fixed-count solver
    #: with no `tol` reports "completed its schedule", not "met a tolerance".
    converged: bool
    iterations: int


@runtime_checkable
class NonlinearSolver(Protocol):
    def solve(
        self,
        step: Callable[[Any], Any],
        y0: Any,
        norm: Optional[Callable[[Any, Any], float]] = None,
        **opts: Any,
    ) -> SolveResult:
        """Drive `y0` towards a fixed point of `step`, i.e. `y == step(y)`.

        `step(y)` returns the next iterate. For a DIRK stage equation
        `Y = y^n + dt*a_ii*f(Y) + rest`, `step` is
        `lambda Y: apply(y_n, dt*a_ii, f(Y), rest)` -- everything the caller needs to
        turn "the right-hand side at Y" into "the next iterate" is closed over
        `step`, so every solver in the ladder shares the same call shape regardless of
        how it gets from one iterate to the next internally (plain reassignment for
        Picard, a Newton correction for JFNK).

        `norm(y_new, y_old)` is an optional scalar convergence measure -- typically
        `fields.state_norm(fields.state_difference(y_new, y_old), rtol, atol)` --
        used only by solvers that iterate to a tolerance rather than a fixed count.
        """
        ...


class FixedPointSolver:
    """Picard iteration with a fixed, data-independent iteration count.

    The default, and the only solver Phase 2 ships (NOTES.md S3.1, S3.4 rung 1). Two
    iterations reach full order for every second-order implicit tableau measured
    there, with no norm, no Jacobian, and no branching -- which also means the
    unrolled graph has a fixed depth, so it is differentiable in both backends by
    construction and CUDA-graph-capturable (NOTES.md S3.7 pain point 12).

    Diverges outside the Picard stability limit (`|dt*a_ii*L| < 1`, tighter in
    practice -- NOTES.md S3.2); that regime needs a Newton-based solver instead of
    more Picard iterations, since more iterations of a divergent map only diverge
    faster.
    """

    def __init__(self, iterations: int = 2):
        if iterations < 1:
            raise ValueError(f'FixedPointSolver needs iterations >= 1, got {iterations}')
        self.iterations = iterations

    def solve(self, step, y0, norm=None, **opts) -> SolveResult:
        iterations = opts.get('iterations', self.iterations)
        tol = opts.get('tol')
        y = y0
        n = 0
        for _ in range(iterations):
            y_new = step(y)
            n += 1
            if tol is not None and norm is not None and norm(y_new, y) < tol:
                return SolveResult(y_new, True, n)
            y = y_new
        # No `tol`: running to a fixed count *is* the contract, not a convergence
        # failure, so `converged=True` here means "completed its fixed schedule" --
        # the only claim this mode makes (NOTES.md S3.4 rung 1). With `tol` set but
        # never reached within `iterations`, it genuinely did not converge.
        return SolveResult(y, tol is None, n)
