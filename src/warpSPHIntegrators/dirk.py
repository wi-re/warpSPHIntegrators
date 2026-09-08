"""Diagonally implicit Runge-Kutta driver (NOTES.md S3.1, S3.6, Phase 2).

Reuses the explicit-RK machinery `butcher.py` already has for the parts that do not
change: accumulating a stage's explicit contribution `sum_{j<i} a_ij k_j`
(`updateStateEuler`, the same helper the explicit path uses), the final b-weighted
update and finalize (`_weighted_update`, `finalizeSystem`), and embedded-pair error
estimation (`_error_estimate`). The only new piece is closing each stage's diagonal
term through a `NonlinearSolver` (`JFNKSolver` by default, with fixed-count Picard
available as an explicit low-overhead override): a DIRK stage equation
`Y_i = y^n + dt*sum_{j<=i} a_ij k_j(Y_j)`, with
`k_i = f(t_i, Y_i)`, is exactly the fixed-point form `NonlinearSolver.solve` expects
once the explicit part is folded into a `base_state` and the diagonal term is
expressed as a function of the still-unknown `Y_i`.

Only the four tableaus below are shipped. NOTES.md S3.6 also lists TR-BDF2 and
ESDIRK3(2)4L[2]SA as "tableau only" work, but both are embedded, higher-stage
tableaus whose published coefficients are easy to transcribe wrong in a way a smoke
test would not obviously catch (an order-2 measurement looks the same whether the
low-order embedded weights are exactly right or merely close) -- landing four
tableaus verified by hand against their order conditions beats landing six where two
are unverified.
"""

from typing import Optional

import numpy as np
from torch.profiler import record_function

from .butcher import _error_estimate, _weighted_update, butcherTableau
from .fields import get_reference_state, integrated_field_names, state_difference, state_norm
from .history import HistoryEntry
from .jfnk import JFNKSolver
from .solvers import FixedPointSolver, NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


def getDIRKTableau(scheme: str) -> butcherTableau:
    if scheme == 'backwardEuler':
        return butcherTableau(a=np.array([[1.0]]), b=np.array([1.0]), c=np.array([1.0]))
    elif scheme == 'implicitMidpoint':
        return butcherTableau(a=np.array([[0.5]]), b=np.array([1.0]), c=np.array([0.5]))
    elif scheme == 'trapezoidal':
        # Lobatto IIIA-2 / Crank-Nicolson: explicit first stage (a11 = 0, c1 = 0), one
        # implicit stage. Exercises the a_ii == 0 branch below on a real tableau.
        return butcherTableau(
            a=np.array([[0.0, 0.0], [0.5, 0.5]]),
            b=np.array([0.5, 0.5]),
            c=np.array([0.0, 1.0]),
        )
    elif scheme == 'SDIRK2':
        # Ellsiepen's L-stable 2-stage SDIRK2, gamma = 1 - sqrt(2)/2. Verified against
        # the order-2 condition sum(b*c) = 1/2 by hand: with this gamma,
        # 2*gamma - gamma**2 == 1/2 exactly (gamma solves gamma**2 - 2*gamma + 1/2 = 0).
        gamma = 1.0 - np.sqrt(2.0) / 2.0
        return butcherTableau(
            a=np.array([[gamma, 0.0], [1.0 - gamma, gamma]]),
            b=np.array([1.0 - gamma, gamma]),
            c=np.array([gamma, 1.0]),
        )
    else:
        raise ValueError(f"Unknown DIRK scheme {scheme}")


def _default_norm(rtol: float, atol: float):
    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), rtol, atol, reference=y_old)
    return norm


def DIRK(initialState, dt, f, tableau: butcherTableau, *args,
        name: str = 'DIRK', solver: Optional[NonlinearSolver] = None, **kwargs):
    """One DIRK step. `tableau.a`'s diagonal may be nonzero; those stages solve implicitly.

    Does not implement first-stage reuse (`priorStep`) yet -- a converged stage's `k`
    was evaluated one Picard iteration before the returned state (see the comment
    below), and whether that residual is small enough to reuse across a *different*
    `dt` at the next step has not been analysed. Rejects it the same way every other
    non-reuse scheme in this library does, with the same warning.

    `history=`, `rtol=`/`atol=` (the default Picard-convergence norm's tolerances, only
    used when `solver_opts={'tol': ...}` requests early exit -- the default fixed
    2-iteration schedule ignores them), and `solver_opts={}` (forwarded to
    `NonlinearSolver.solve` as `**opts`) are all optional.

    `norm` (below) is always this module's own Hairer-Wanner weighted-RMS
    (`fields.state_norm`, `_default_norm`) -- its own documented convention is
    "< 1.0 means converged", *not* whatever a given solver's own `tol` default
    means (`FixedPointSolver`'s `tol` is unset by default, opt-in only, so
    this never mattered for it; `JFNKSolver`'s `tol` is its GMRES linear-solve
    tolerance, always set, and a *different* convention -- see `jfnk.py`'s
    `JFNKSolver.solve` docstring on `newton_tol`). `newton_tol=1.0` is
    defaulted into `solver_opts` here, matching this norm's own scale, so a
    solver that reads `newton_tol` (`JFNKSolver`) gets a correctly-paired
    threshold without every caller needing to know this norm's convention;
    a caller's own explicit `solver_opts['newton_tol']` still wins.
    """
    verbose = bool(kwargs.get('verbose', False))
    solver = solver or JFNKSolver()
    history = kwargs.pop('history', None)
    reject_prior_step(name, kwargs.pop('priorStep', None))
    solver_opts = kwargs.get('solver_opts', {})
    norm = _default_norm(kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-6))

    with record_function("[Integration] DIRK"):
        initializeSystem(initialState, dt, *args, **kwargs)
        ks, rs = [], []
        stage_state = None

        for i, c_i in enumerate(tableau.c):
            with record_function(f"[Integration] DIRK: stage {i}"):
                a_row = tableau.a[i, :i]
                a_ii = tableau.a[i, i]
                t_i = float(initialState.t + c_i * dt)

                base_state = initialState.initializeNewState(*args, **kwargs)
                for j, a_ij in enumerate(a_row):
                    if a_ij != 0:
                        base_state = updateStateEuler(base_state, ks[j], a_ij * dt, copyState=False, **kwargs)
                base_state.t = t_i

                if a_ii == 0:
                    # An explicit stage inside an otherwise-implicit tableau (e.g.
                    # trapezoidal's first stage): no solve needed, k_i = f(t_i, base).
                    k_i, r_i = updateStep(initialState, base_state, dt, f, *args, **kwargs)
                    stage_state = base_state
                else:
                    box = {}

                    def step_fn(Y, base_state=base_state, a_ii=a_ii, t_i=t_i, box=box):
                        Y.t = t_i
                        k, r = updateStep(initialState, Y, dt, f, *args, **kwargs)
                        # Also remember `Y` itself, not just `(k, r)`: `updateStep`'s
                        # `preprocess` call ran *on this object*, in place, so it is the
                        # only thing downstream that is carrying this stage's `copied`
                        # fields (density, pressure, ...). The `y_new` this function
                        # returns is a fresh clone built by `updateStateEuler`
                        # (`copyState=True`) that never saw `preprocess` at all.
                        box['k'], box['r'], box['Y'] = k, r, Y
                        return updateStateEuler(base_state, k, a_ii * dt, copyState=True, **kwargs)

                    y0 = base_state.initializeNewState(*args, **kwargs)
                    y0.t = t_i
                    if verbose:
                        print(f"[Integrator] DIRK stage {i}: solving Y = base + {a_ii:.4f}*dt*f(Y) at t={t_i:.4f}")
                    solve_result = solver.solve(step_fn, y0, norm, **solver_opts)
                    # `box['Y']`/`box['k']` come from the *last* `step_fn` call, i.e. the
                    # last RHS evaluation of this stage -- the same "buffer the last
                    # evaluation ran on" convention `butcher.RungeKuttaB` uses for its
                    # own `lastStageState`. `stage_state = base + a_ii*dt*k_i` holds by
                    # construction (that is what `step_fn` just returned), so `(k_i,
                    # box['Y'])` satisfy the stage equation the way a converged Picard
                    # iterate is supposed to, not an approximation one iteration behind.
                    if isinstance(solver, JFNKSolver):
                        # Finite-difference Krylov probes use isolated cloned states,
                        # so re-evaluate the converged stage on this driver's buffer.
                        # That gives copied fields their standard last-stage lifecycle.
                        stage_state = base_state
                        solved_ref = get_reference_state(solve_result.y)
                        stage_ref = get_reference_state(stage_state)
                        for field_name in integrated_field_names(solve_result.y):
                            setattr(stage_ref, field_name, getattr(solved_ref, field_name))
                        stage_state.t = t_i
                        k_i, r_i = updateStep(initialState, stage_state, dt, f, *args, **kwargs)
                    else:
                        k_i, r_i = box['k'], box['r']
                        stage_state = box['Y']

                ks.append(k_i)
                rs.append(r_i)

        lastStageState = stage_state
        stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]

        def _next_history():
            if history is None:
                return None
            entry = HistoryEntry(t=float(initialState.t), dt=dt, update=ks[-1], aux=rs[-1])
            return history.pushed(entry)

        with record_function("[Integration] DIRK: Update"):
            if not isinstance(tableau.b, tuple):
                new_state = _weighted_update(initialState, ks, tableau.b, dt, *args, **kwargs)
                new_state.t = float(initialState.t + dt)
                finalizeSystem(new_state, initialState, dt, rs, ks, tableau.b,
                               *args, lastStageSystem=lastStageState, **kwargs)
                return IntegrationResult(state=new_state, stages=stages, history=_next_history())

            b_main, b_embedded = tableau.b
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(state=new_state, stages=stages, error=error, history=_next_history())


def dirkScheme(tableau_name: str, **tableau_kwargs):
    """Build a DIRK scheme callable that carries its tableau, mirroring `butcherScheme`.

    Deliberately does **not** expose `.butcherTableau` the way `butcherScheme` does:
    that attribute is what `reuse._tableau_of` looks for to run the *explicit*-scheme
    reuse analysis, whose substitution argument assumes a stage is a plain function of
    already-known states -- not true for an implicit stage, whose value depends on
    `dt` through the very solve reuse would try to skip. Leaving the attribute off
    makes `reuse.step_reuse_analysis` fall through to its "no tableau, no recorded
    reuse behaviour" answer, which is the correct one until DIRK reuse is analysed on
    its own terms.
    """
    tableau = getDIRKTableau(tableau_name)

    def scheme(state, dt, f, *args, **kwargs):
        return DIRK(state, dt, f, tableau, *args, name=tableau_name, **kwargs)

    scheme.__name__ = tableau_name
    scheme.dirkTableau = tableau
    return scheme


backwardEuler = dirkScheme('backwardEuler')
implicitMidpoint = dirkScheme('implicitMidpoint')
trapezoidal = dirkScheme('trapezoidal')
SDIRK2 = dirkScheme('SDIRK2')
