"""Numerical stepping for dirk; configuration and coefficients are supplied separately."""
from typing import Optional

import numpy as np

from torch.profiler import record_function

from warpSPHIntegrators.drivers.runge_kutta import _error_estimate, _weighted_update
from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau

from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, state_difference, state_norm

from warpSPHIntegrators.state.history import HistoryEntry

from warpSPHIntegrators.solvers.krylov import JFNKSolver

from warpSPHIntegrators.solvers.nonlinear import record_solve_diagnostics, NonlinearSolver

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from warpSPHIntegrators.state.lifecycle import finalizeSystem, initializeSystem, reject_prior_step, unpack_prior_step, updateStateEuler, updateStep

def _dirk_reuses_first_stage(tableau: butcherTableau) -> bool:
    """True iff this DIRK tableau consumes a reused ``k0`` losslessly.

    Both halves of the (implicit) FSAL condition must hold:

    * the first stage is explicit (``a[0,0] == 0``, ``c[0] == 0``), so it needs no
      solve and can consume a ready-made derivative; and
    * the tableau is stiffly accurate (``c[-1] == 1``, ``a[-1] == b``), so the
      previous step's last stage IS ``y^{n+1}`` and its converged derivative is
      exactly ``f(t^{n+1}, y^{n+1})`` -- the value this step's first stage needs.

    Only then is the reuse exact (no order cost). This is derived from the tableau
    itself so the call-time decision cannot drift from the registered
    ``stiffly_accurate`` metadata; ``reuse.dirk_reuse_analysis`` reports the same
    property for callers that only have the scheme.
    """
    a = np.asarray(tableau.a, dtype=float)
    c = np.asarray(tableau.c, dtype=float)
    b = tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b
    b = np.asarray(b, dtype=float)
    explicit_first = abs(a[0, 0]) < 1e-12 and abs(c[0]) < 1e-12
    stiffly_accurate = abs(c[-1] - 1.0) < 1e-12 and np.allclose(a[-1], b, atol=1e-12)
    return bool(explicit_first and stiffly_accurate)

def _default_norm(rtol: float, atol: float):
    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), rtol, atol, reference=y_old)
    return norm

def DIRK(initialState, dt, f, tableau: butcherTableau, *args,
        name: str = 'DIRK', solver: Optional[NonlinearSolver] = None, warmStart=None,
        _initialize=True, _finalize=True, **kwargs):
    """One DIRK step. `tableau.a`'s diagonal may be nonzero; those stages solve implicitly.

    First-stage reuse (`priorStep`) is implemented for the tableaus that satisfy both
    halves of the (implicit) FSAL condition -- an explicit first stage
    (`a[0,0] == 0`, `c[0] == 0`) *and* stiff accuracy (`c[-1] == 1`, `a[-1] == b`):
    Trapezoidal, TR-BDF2, ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA. For those, the
    previous step's last stage IS `y^{n+1}`, so `stages[-1].update` is exactly the
    `f(t^{n+1}, y^{n+1})` this step's explicit first stage needs; the reuse costs no
    order and saves one (cheap, explicit) RHS evaluation per step. Every other DIRK
    tableau -- SDIRK2's implicit first stage, the single-stage backward Euler and
    implicit midpoint -- cannot consume a reused `k0`, so it rejects `priorStep` with
    the same warning every other non-reuse scheme in this library gives.

    `warmStart=StageResult` optionally seeds an implicit first stage with the
    previous first-stage derivative. It changes only the Newton initial guess,
    never reuses a derivative as a solved stage. Explicit-first tableaus refuse it.
    Preconditioner context includes dt, stage_index, stage_time and
    stage_coefficient=dt*a_ii; driver-owned values override caller context keys.

    `history=`, `rtol=`/`atol=` (the default Picard-convergence norm's tolerances, only
    used when `solver_opts={'tol': ...}` requests early exit -- the default fixed
    2-iteration schedule ignores them), and `solver_opts={}` (forwarded to
    `NonlinearSolver.solve` as `**opts`) are all optional.

    `norm` (below) is always this module's own Hairer-Wanner weighted-RMS
    (`fields.state_norm`, `_default_norm`) -- its own documented convention is
    "< 1.0 means converged". Note that this driver does *not* inject a
    `newton_tol` into `solver_opts`: `JFNKSolver`'s `newton_tol` falls back to
    its own `tol` (1e-8, the GMRES linear-solve tolerance), which float64
    solves reach and float32 ones exit through the stagnation floor instead
    (see `jfnk.py`'s `JFNKSolver.solve` docstring on `newton_tol`); a caller
    that wants the norm's own "< 1.0" scale as the Newton threshold passes
    `solver_opts={'newton_tol': 1.0}` explicitly.
    """
    verbose = bool(kwargs.get('verbose', False))
    solver = solver or JFNKSolver()
    if warmStart is not None:
        if tableau.a[0,0] == 0 or not isinstance(warmStart, StageResult) or warmStart.update is None:
            raise ValueError('DIRK warmStart needs a StageResult for an implicit first stage')
    history = kwargs.pop('history', None)
    priorStep = kwargs.pop('priorStep', None)
    # Reuse is valid for exactly the stiffly-accurate tableaus with an explicit first
    # stage (`_dirk_reuses_first_stage`). For every other DIRK tableau the reused k0
    # would be stale (last stage is not y^{n+1}) or unusable (the first stage is
    # implicit), so reject it the same way every other non-reuse scheme does.
    if priorStep is not None and not _dirk_reuses_first_stage(tableau):
        reject_prior_step(name, priorStep)
    solver_opts = kwargs.get('solver_opts', {})
    norm = _default_norm(kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-6))

    with record_function("[Integration] DIRK"):
        if _initialize:initializeSystem(initialState, dt, *args, **kwargs)
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
                    if i == 0 and priorStep is not None:
                        # Stiffly accurate + explicit first stage: the previous step's
                        # last stage IS f(t^{n+1}, y^{n+1}), exactly this stage's value,
                        # so consume it instead of re-evaluating. Saves one (cheap,
                        # explicit) RHS evaluation per step at no order cost.
                        k_i, r_i = unpack_prior_step(priorStep, verbose)
                    else:
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
                    if i == 0 and warmStart is not None:
                        y0 = updateStateEuler(y0,warmStart.update,a_ii*dt,copyState=False,**kwargs)
                    y0.t = t_i
                    if verbose:
                        print(f"[Integrator] DIRK stage {i}: solving Y = base + {a_ii:.4f}*dt*f(Y) at t={t_i:.4f}")
                    stage_opts = dict(solver_opts)
                    context = dict(stage_opts.get('preconditioner_context',getattr(solver,'preconditioner_context',None)) or {})
                    context.update(dt=dt,stage_index=i,stage_time=t_i,stage_coefficient=float(a_ii*dt))
                    stage_opts['preconditioner_context'] = context
                    solve_result = solver.solve(step_fn, y0, norm, **stage_opts)
                    solver_diagnostics.append(record_solve_diagnostics(solve_result))
                    # `box['Y']`/`box['k']` come from the *last* `step_fn` call, i.e. the
                    # last RHS evaluation of this stage -- the same "buffer the last
                    # evaluation ran on" convention `butcher.RungeKuttaB` uses for its
                    # own `lastStageState`. `stage_state = base + a_ii*dt*k_i` holds by
                    # construction (that is what `step_fn` just returned), so `(k_i,
                    # box['Y'])` satisfy the stage equation the way a converged Picard
                    # iterate is supposed to, not an approximation one iteration behind.
                    if isinstance(solver, JFNKSolver) or 'k' not in box:
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
                if _finalize:
                    finalizeSystem(new_state, initialState, dt, rs, ks, tableau.b,
                                   *args, lastStageSystem=lastStageState, **kwargs)
                return IntegrationResult(
                    state=new_state, stages=stages, history=_next_history(),
                    solver_diagnostics=solver_diagnostics)

            b_main, b_embedded = tableau.b
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            if _finalize:
                finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                               *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(
                state=new_state, stages=stages, error=error, history=_next_history(),
                solver_diagnostics=solver_diagnostics)
