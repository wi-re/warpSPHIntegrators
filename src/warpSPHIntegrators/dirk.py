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

The tableaus below are all verified against their order conditions (symbolic
Taylor-model expansion of one step against the exact Taylor solution) and, for
the L-stable ones, their stability functions, not just smoke-tested: an embedded
pair's low-order weights are easy to transcribe wrong in a way an order
measurement would not obviously catch. The ESDIRK324/ESDIRK436 tableaus and
TR-BDF2's embedded pair are the Kennedy-Carpenter coefficients as published in
SUNDIALS ARKODE v7.9.0 (`src/arkode/arkode_butcher_dirk.def`); TR-BDF2's `a`/`b`
match that entry exactly, and its embedded `d` is the entry's published
third-order pair, so the estimate is O(dt^3) -- the propagated branch's own true
local error.
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
    elif scheme == 'TRBDF2':
        # TR-BDF2: a trapezoidal substep at gamma = 2-sqrt(2), followed by a
        # variable-step BDF2 endpoint solve. Written as an SDIRK tableau, both
        # implicit diagonals equal (1-gamma)/(2-gamma), making it L-stable and
        # stiffly accurate. sum(b*c) = 1/2 verifies the order-2 condition.
        # The embedded pair is SUNDIALS ARKODE's ARKODE_TRBDF2_3_3_2 published
        # (2, 3) pair (the entry's `d` vector): the estimate is O(dt^3), i.e.
        # the propagated branch's own true local error, and it satisfies
        # sum(d) = 1, d.c = 1/2, d.c^2 = 1/3 exactly.
        gamma = 2.0 - np.sqrt(2.0)
        diagonal = (1.0 - gamma) / (2.0 - gamma)
        first_weight = 1.0 / (2.0 * (2.0 - gamma))
        b_embed = np.array([
            (1.0 - np.sqrt(2.0) / 4.0) / 3.0,
            (1.0 + 3.0 * np.sqrt(2.0) / 4.0) / 3.0,
            gamma / 6.0,
        ])
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0],
                [gamma / 2.0, gamma / 2.0, 0.0],
                [first_weight, first_weight, diagonal],
            ]),
            b=(np.array([first_weight, first_weight, diagonal]), b_embed),
            c=np.array([0.0, gamma, 1.0]),
        )
    elif scheme == 'ESDIRK324L2SA':
        # Kennedy-Carpenter 4-stage ESDIRK3(2)4L[2]SA (order 3 / embedded 2,
        # A- and L-stable, stiffly accurate), as published in SUNDIALS ARKODE
        # v7.9.0, `src/arkode/arkode_butcher_dirk.def` (ARKODE_ESDIRK324L2SA).
        # Explicit first stage (a11 = 0, c1 = 0). Order, embedded order, and
        # A/L stability verified by symbolic Taylor expansion, numeric
        # local-error rates, and exact stability-function sweeps (NOTES.md S3.6).
        g = 0.4358665215084589994160194511935568425293
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [g, g, 0.0, 0.0],
                [0.2576482460664272457999960162840797092643,
                 -0.09351476757488624521601546747763655179361,
                 g, 0.0],
                [0.1876410243467238251612921441668043913795,
                 -0.5952974735769549480478230275858851737782,
                 0.9717899277217721234705114322255239398694,
                 g],
            ]),
            b=(np.array([0.1876410243467238251612921441668043913795,
                         -0.5952974735769549480478230275858851737782,
                         0.9717899277217721234705114322255239398694,
                         g]),
               np.array([0.1088966176158644541561307380704960821824,
                         -0.9153258118707127534816380978168183454991,
                         1.271273597302152167844715894135642876535,
                         0.5351555969526961314807914656106793867813])),
            c=np.array([0.0,
                        0.8717330430169179988320389023871136850586,
                        0.6,
                        1.0]),
        )
    elif scheme == 'ESDIRK436L2SA':
        # Kennedy-Carpenter 6-stage ESDIRK4(3)6L[2]SA (order 4 / embedded 3,
        # A- and L-stable, stiffly accurate), as published in SUNDIALS ARKODE
        # v7.9.0, `src/arkode/arkode_butcher_dirk.def` (ARKODE_ESDIRK436L2SA),
        # in its exact radical/rational form. Explicit first stage. The
        # off-diagonal A[i][0] entries are derived exactly as in that file:
        # A[i][0] = c[i] - sum_{j>=1} A[i][j]. Order, embedded order, and
        # A/L stability verified as for ESDIRK324L2SA (NOTES.md S3.6).
        sqrt2 = np.sqrt(2.0)
        b0 = (1181.0 - 987.0 * sqrt2) / 13782.0
        b2 = 47.0 * (-267.0 + 1783.0 * sqrt2) / 273343.0
        b3 = -16.0 * (-22922.0 + 3525.0 * sqrt2) / 571953.0
        b4 = -15625.0 * (97.0 + 376.0 * sqrt2) / 90749876.0
        b5 = 1.0 / 4.0
        b_main = np.array([b0, b0, b2, b3, b4, b5])
        b_embed = np.array([
            -480923228411.0 / 4982971448372.0,
            -480923228411.0 / 4982971448372.0,
            6709447293961.0 / 12833189095359.0,
            3513175791894.0 / 6748737351361.0,
            -498863281070.0 / 6042575550617.0,
            2077005547802.0 / 8945017530137.0,
        ])
        A11 = 1.0 / 4.0
        A21 = (1.0 - sqrt2) / 8.0
        A22 = 1.0 / 4.0
        A31 = (5.0 - 7.0 * sqrt2) / 64.0
        A32 = 7.0 * (1.0 + sqrt2) / 32.0
        A33 = 1.0 / 4.0
        A41 = -(13796.0 + 54539.0 * sqrt2) / 125000.0
        A42 = (506605.0 + 132109.0 * sqrt2) / 437500.0
        A43 = 166.0 * (-97.0 + 376.0 * sqrt2) / 109375.0
        A44 = 1.0 / 4.0
        c = np.array([0.0, 0.5, (2.0 - sqrt2) / 4.0, 5.0 / 8.0, 26.0 / 25.0, 1.0])
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [c[1] - A11, A11, 0.0, 0.0, 0.0, 0.0],
                [c[2] - A21 - A22, A21, A22, 0.0, 0.0, 0.0],
                [c[3] - A31 - A32 - A33, A31, A32, A33, 0.0, 0.0],
                [c[4] - A41 - A42 - A43 - A44, A41, A42, A43, A44, 0.0],
                [b_main[0], b_main[1], b_main[2], b_main[3], b_main[4], b_main[5]],
            ]),
            b=(b_main, b_embed),
            c=c,
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
        solver_diagnostics = []
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
                    solver_diagnostics.append(None)
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
                    solver_diagnostics.append(solve_result.diagnostics)
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
                return IntegrationResult(
                    state=new_state, stages=stages, history=_next_history(),
                    solver_diagnostics=solver_diagnostics)

            b_main, b_embedded = tableau.b
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(
                state=new_state, stages=stages, error=error, history=_next_history(),
                solver_diagnostics=solver_diagnostics)


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
TRBDF2 = dirkScheme('TRBDF2')
ESDIRK324L2SA = dirkScheme('ESDIRK324L2SA')
ESDIRK436L2SA = dirkScheme('ESDIRK436L2SA')
