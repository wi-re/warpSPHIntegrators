"""Additive (IMEX) Runge-Kutta driver for the Kennedy-Carpenter ARK pair (Phase 5).

Generalizes the single-stage ``imex.IMEXEuler`` split into a multi-stage additive
driver. A step is an *additive* (two-part) Runge-Kutta method: the right-hand side is
split ``f = f_exp + f_imp`` and each half has its own tableau. The stages are

    z_0 = y^n,   f_exp_j = f_exp(t_j, z_j),   f_imp_j = f_imp(t_j, z_j)
    z_i = y^n + dt * ( sum_{j<i} A_e[i][j] f_exp_j + sum_{j<i} A_i[i][j] f_imp_j )
    and, when the implicit half has a nonzero diagonal at stage i, z_i is the
    solution of  z_i = z_i + dt * A_i[i][i] f_imp(z_i)   (a JFNK fixed point).

The propagated update is the *additive b-weighted* combination

    y^{n+1} = y^n + dt * sum_j ( b_exp[j] f_exp_j + b_imp[j] f_imp_j ),

and the embedded estimate is the same combination with the ``d`` weights. This is
exactly the composition rule SUNDIALS ARKODE v7.9.0 uses for its ARK methods
(``arkode_arkstep.c``: ``arkStep_StageSetup`` for the stage base + diagonal solve,
``arkStep_ComputeSolutions`` for the non-stiffly-accurate additive update and the
``(b - d)`` error).

Why the update is the additive b-weighted combination and *not* the last stage: an
ARK method is a pair, and the two halves are not symmetric with respect to
stiff accuracy. For both pairs here the *implicit* (DIRK) half is stiffly accurate
(``b_imp == A_i[last]``), but the *explicit* (ERK) half is not (``b_exp !=
A_e[last]``). So the combined method is **not** FSAL -- ``y^{n+1} != z_last`` -- and
SUNDIALS's own ``IsStifflyAccurate`` check (which requires *both* halves) falls
through to the additive b-weighted update. Using the last stage as the solution
instead would drop the method to first order (verified numerically, docs/research/integrators/notes-3-6.md);
the additive b-weighted combination retains the full claimed order. The "SA" in the
published names therefore refers to the *implicit half's* property, not to the
combined method, and the schemes are registered with ``stiffly_accurate=False``
(matching this codebase's convention that the flag tracks FSAL).

RHS splitting. Pass ``IMEXRHS(explicit=..., implicit=...)`` as ``f`` to activate the
split. Passing an ordinary callable treats its complete update as the implicit part
(``f_exp = 0``), so the standard integrator call convention remains valid and the
step reduces to the *implicit half alone* -- a standalone ESDIRK of the same order
(the pure-implicit limit). That limit *is* FSAL, which is the only case in which
first-stage reuse would be exact.

Copied-field semantics when both callbacks are active. Each RHS evaluation runs
``updateStep`` on its **own** state object, so each gets exactly one
preprocess -> f -> postprocess cycle, the same as a single-RHS scheme: the implicit
callback *owns the stage buffer* (the JFNK solve and its post-convergence
re-evaluation run on it, exactly as in ``dirk.DIRK``), and the explicit callback is
evaluated on a throwaway clone of that buffer (``stage_state.initializeNewState()``,
which carries the converged integrated fields and lets preprocess recompute the
copied ones). The final state's copied fields come from the implicit buffer
(``lastStageSystem``), the ``dirk.DIRK`` convention.

Does not implement first-stage reuse (``priorStep``): the combined ARK method is not
FSAL, so a reused last stage is not ``f(t^{n+1}, y^{n+1})``. Rejects it the same way
``dirk.DIRK`` does.

The tableaus below are the Kennedy-Carpenter ARK(2,3)4L[2]SA and ARK(3,4)6L[2]SA
coefficients as published in SUNDIALS ARKODE v7.9.0, split across
``src/arkode/arkode_butcher_erk.def`` (explicit halves) and
``arkode_butcher_dirk.def`` (implicit halves). Order, embedded order, the
pure-explicit / pure-implicit component limits, and the two-parameter IMEX
stability region are verified in tests/ and docs/research/integrators/notes-3-6.md.
"""

from warpSPHIntegrators.schemes.coefficients.additive import AdditiveTableau

from typing import Optional


from torch.profiler import record_function

from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, state_difference, state_norm
from warpSPHIntegrators.state.history import HistoryEntry
from warpSPHIntegrators.solvers.krylov import JFNKSolver
from warpSPHIntegrators.adapters.rhs import resolve
from warpSPHIntegrators.solvers.nonlinear import record_solve_diagnostics, NonlinearSolver
from warpSPHIntegrators.state.specs import IntegrationResult, StageResult
from warpSPHIntegrators.state.lifecycle import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


def _default_norm(rtol: float, atol: float):
    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), rtol, atol, reference=y_old)
    return norm


def _additive_weighted_update(initialState, ks_exp, ks_imp, b_exp, b_imp, dt, *args, **kwargs):
    """y^{n+1} = y^n + dt * sum_i (b_exp[i] k_exp_i + b_imp[i] k_imp_i), one state.

    A half that is absent (its ``ks`` entries are ``None``) contributes nothing.
    Mirrors ``butcher._weighted_update`` for the additive two-part case.
    """
    new_state = initialState.initializeNewState(*args, **kwargs)
    for i in range(len(b_imp)):
        if ks_exp[i] is not None and b_exp[i] != 0:
            new_state = updateStateEuler(new_state, ks_exp[i], b_exp[i] * dt, copyState=False, **kwargs)
        if ks_imp[i] is not None and b_imp[i] != 0:
            new_state = updateStateEuler(new_state, ks_imp[i], b_imp[i] * dt, copyState=False, **kwargs)
    return new_state


def _additive_error_estimate(initialState, ks_exp, ks_imp, b_exp, b_imp, d_exp, d_imp, dt,
                             *args, **kwargs):
    """Propagated-minus-embedded difference of an additive pair, as a state.

    Computes both the ``b``-weighted and the ``d``-weighted solutions in full (the
    additive analogue of ``butcher._error_estimate``) and differences them
    field-by-field, so a nonlinear ``apply_state_update`` is handled correctly.
    """
    y_main = _additive_weighted_update(initialState, ks_exp, ks_imp, b_exp, b_imp, dt, *args, **kwargs)
    y_embed = _additive_weighted_update(initialState, ks_exp, ks_imp, d_exp, d_imp, dt, *args, **kwargs)
    error = state_difference(y_main, y_embed)
    error.t = float(initialState.t + dt)
    return error


def ARK(initialState, dt, f, tableau: AdditiveTableau, *args,
        name: str = 'ARK', solver: Optional[NonlinearSolver] = None, **kwargs):
    """One additive (IMEX) Runge-Kutta step. See the module docstring for the rule.

    ``f`` is either an ``IMEXRHS(explicit=..., implicit=...)`` (activates the split)
    or an ordinary RHS callable (treated as fully implicit; the step reduces to the
    implicit half). ``history=``, ``rtol=``/``atol=``, and ``solver_opts={}`` are
    optional and mean the same things as in ``dirk.DIRK``; ``norm`` is this module's
    own Hairer-Wanner weighted-RMS (``_default_norm``). ``priorStep`` is rejected.
    """
    verbose = bool(kwargs.get('verbose', False))
    solver = solver or JFNKSolver()
    history = kwargs.pop('history', None)
    reject_prior_step(name, kwargs.pop('priorStep', None))
    solver_opts = kwargs.get('solver_opts', {})
    norm = _default_norm(kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-6))

    # The additive split is read by capability (NOTES S3.12), not by type: a plain
    # callable resolves to (explicit=None, implicit=f) -- the pure-implicit limit --
    # exactly as the old isinstance dispatch did, so bare-callable trajectories are
    # unchanged.
    resolved = resolve(f, scheme_name=name)
    explicit_rhs, implicit_rhs = resolved.explicit, resolved.implicit

    with record_function(f"[Integration] {name}"):
        initializeSystem(initialState, dt, *args, **kwargs)
        s = len(tableau.c)
        ks_exp = [None] * s
        ks_imp = [None] * s
        rs_exp = [None] * s
        rs_imp = [None] * s
        solver_diagnostics = []
        stage_state = None

        for i, c_i in enumerate(tableau.c):
            with record_function(f"[Integration] {name}: stage {i}"):
                a_ii = tableau.a_implicit[i, i]
                t_i = float(initialState.t + c_i * dt)

                base_state = initialState.initializeNewState(*args, **kwargs)
                for j in range(i):
                    a_ej = tableau.a_explicit[i, j]
                    if a_ej != 0 and ks_exp[j] is not None:
                        base_state = updateStateEuler(base_state, ks_exp[j], a_ej * dt,
                                                      copyState=False, **kwargs)
                    a_ij = tableau.a_implicit[i, j]
                    if a_ij != 0 and ks_imp[j] is not None:
                        base_state = updateStateEuler(base_state, ks_imp[j], a_ij * dt,
                                                      copyState=False, **kwargs)
                base_state.t = t_i

                if a_ii == 0:
                    # Explicit stage (stage 0): no solve. The implicit callback owns
                    # the buffer; the explicit one runs on a throwaway clone.
                    k_imp_i, r_imp_i = updateStep(initialState, base_state, dt, implicit_rhs,
                                                  *args, **kwargs)
                    stage_state = base_state
                    solver_diagnostics.append(None)
                    if explicit_rhs is not None:
                        exp_state = base_state.initializeNewState(*args, **kwargs)
                        exp_state.t = t_i
                        k_exp_i, r_exp_i = updateStep(initialState, exp_state, dt, explicit_rhs,
                                                      *args, **kwargs)
                    else:
                        k_exp_i, r_exp_i = None, None
                else:
                    box = {}

                    def step_fn(Y, base_state=base_state, a_ii=a_ii, t_i=t_i, box=box):
                        Y.t = t_i
                        k, r = updateStep(initialState, Y, dt, implicit_rhs, *args, **kwargs)
                        # Remember `Y` itself, not just (k, r): updateStep's preprocess
                        # ran on it in place, so it is what carries this stage's copied
                        # fields (see the dirk.DIRK comment for the full rationale).
                        box['k'], box['r'], box['Y'] = k, r, Y
                        return updateStateEuler(base_state, k, a_ii * dt, copyState=True, **kwargs)

                    y0 = base_state.initializeNewState(*args, **kwargs)
                    y0.t = t_i
                    if verbose:
                        print(f"[Integrator] {name} stage {i}: solving "
                              f"Y = base + {a_ii:.4f}*dt*f_imp(Y) at t={t_i:.4f}")
                    solve_result = solver.solve(step_fn, y0, norm, **solver_opts)
                    solver_diagnostics.append(record_solve_diagnostics(solve_result))

                    if isinstance(solver, JFNKSolver) or 'k' not in box:
                        # Finite-difference Krylov probes use isolated clones, so
                        # re-evaluate the converged stage on this driver's buffer to
                        # give copied fields their standard last-stage lifecycle.
                        stage_state = base_state
                        solved_ref = get_reference_state(solve_result.y)
                        stage_ref = get_reference_state(stage_state)
                        for field_name in integrated_field_names(solve_result.y):
                            setattr(stage_ref, field_name, getattr(solved_ref, field_name))
                        stage_state.t = t_i
                        k_imp_i, r_imp_i = updateStep(initialState, stage_state, dt, implicit_rhs,
                                                      *args, **kwargs)
                    else:
                        k_imp_i, r_imp_i = box['k'], box['r']
                        stage_state = box['Y']

                    if explicit_rhs is not None:
                        # Throwaway clone of the converged stage: carries the
                        # integrated fields, preprocess recomputes the copied ones.
                        exp_state = stage_state.initializeNewState(*args, **kwargs)
                        exp_state.t = t_i
                        k_exp_i, r_exp_i = updateStep(initialState, exp_state, dt, explicit_rhs,
                                                      *args, **kwargs)
                    else:
                        k_exp_i, r_exp_i = None, None

                ks_exp[i] = k_exp_i
                ks_imp[i] = k_imp_i
                rs_exp[i] = r_exp_i
                rs_imp[i] = r_imp_i

        lastStageState = stage_state
        stages = [StageResult(aux=(rs_exp[i], rs_imp[i]), update=ks_imp[i]) for i in range(s)]

        def _next_history():
            if history is None:
                return None
            entry = HistoryEntry(t=float(initialState.t), dt=dt, update=ks_imp[-1], aux=rs_imp[-1])
            return history.pushed(entry)

        with record_function(f"[Integration] {name}: Update"):
            new_state = _additive_weighted_update(
                initialState, ks_exp, ks_imp, tableau.b_explicit, tableau.b_implicit,
                dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _additive_error_estimate(
                initialState, ks_exp, ks_imp,
                tableau.b_explicit, tableau.b_implicit,
                tableau.d_explicit, tableau.d_implicit, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs_imp, ks_imp, tableau.b_implicit,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(
                state=new_state, stages=stages, error=error, history=_next_history(),
                solver_diagnostics=solver_diagnostics)
