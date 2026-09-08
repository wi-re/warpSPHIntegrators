"""Pluggable stage-equation solvers (NOTES.md S4).

An implicit scheme's stage equation -- `Y = g(Y)` for a DIRK stage, more generally
`F(Y) = 0` for a residual -- needs something to drive `Y` towards a solution.
`NonlinearSolver` is the one interface every rung of the ladder in NOTES.md S3.4
implements: fixed-count Picard today (`FixedPointSolver`, all Phase 2 ships), JFNK
with finite-difference matvecs and a user-supplied `solve_linear` hook later. A driver
is written once against the protocol and gets a solver swapped under it without a
rewrite -- that pluggability, not any one solver, is the point of landing this now.
"""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Protocol, runtime_checkable

from .fields import flatten_integrated, unflatten_integrated


@dataclass(frozen=True)
class SolveDiagnostics:
    """Observable outcome of one nonlinear solve."""

    residual: Optional[float] = None
    gmres_iterations: int = 0
    rhs_evaluations: int = 0
    termination: str = 'fixed_iterations'


class SolveResult(NamedTuple):
    """What a `NonlinearSolver.solve` call produced."""

    y: Any
    #: See each solver's own docstring for what this claims -- a fixed-count solver
    #: with no `tol` reports "completed its schedule", not "met a tolerance".
    converged: bool
    iterations: int
    diagnostics: Optional[SolveDiagnostics] = None


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
            residual = norm(y_new, y) if norm is not None else None
            if tol is not None and residual is not None and residual < tol:
                return SolveResult(y_new, True, n, SolveDiagnostics(
                    residual=residual, rhs_evaluations=n, termination='tolerance'))
            y = y_new
        # No `tol`: running to a fixed count *is* the contract, not a convergence
        # failure, so `converged=True` here means "completed its fixed schedule" --
        # the only claim this mode makes (NOTES.md S3.4 rung 1). With `tol` set but
        # never reached within `iterations`, it genuinely did not converge.
        return SolveResult(y, tol is None, n, SolveDiagnostics(
            residual=residual if norm is not None else None,
            rhs_evaluations=n,
            termination='fixed_iterations' if tol is None else 'max_iterations',
        ))


class RelaxedFixedPointSolver(FixedPointSolver):
    """Damped Picard iteration: ``y <- y + relaxation * (step(y) - y)``.

    Relaxation can improve a non-stiff Picard solve whose fixed-point map is too
    aggressive, but it does not make Picard a stiff solver. Use ``JFNKSolver`` for
    a genuinely stiff system.
    """

    def __init__(self, relaxation: float = 0.5, iterations: int = 2):
        super().__init__(iterations=iterations)
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(
                f'RelaxedFixedPointSolver needs relaxation in (0, 1], got {relaxation}'
            )
        self.relaxation = relaxation

    def solve(self, step, y0, norm=None, **opts) -> SolveResult:
        iterations = opts.get('iterations', self.iterations)
        relaxation = opts.get('relaxation', self.relaxation)
        tol = opts.get('tol')
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f'relaxation must lie in (0, 1], got {relaxation}')

        y = y0
        for n in range(1, iterations + 1):
            y_step = step(y)
            y_flat = flatten_integrated(y)
            y_new = unflatten_integrated(
                y_flat + relaxation * (flatten_integrated(y_step) - y_flat), y_step
            )
            residual = norm(y_new, y) if norm is not None else None
            if tol is not None and residual is not None and residual < tol:
                return SolveResult(y_new, True, n, SolveDiagnostics(
                    residual=residual, rhs_evaluations=n, termination='tolerance'))
            y = y_new
        return SolveResult(y, tol is None, iterations, SolveDiagnostics(
            residual=residual if norm is not None else None,
            rhs_evaluations=iterations,
            termination='fixed_iterations' if tol is None else 'max_iterations',
        ))
