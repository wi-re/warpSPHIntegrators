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
instead would drop the method to first order (verified numerically, NOTES.md S3.6);
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
stability region are verified in tests/ and NOTES.md S3.6.
"""

from typing import NamedTuple, Optional

import numpy as np
from torch.profiler import record_function

from .fields import get_reference_state, integrated_field_names, state_difference, state_norm
from .history import HistoryEntry
from .jfnk import JFNKSolver
from .rhs import resolve
from .solvers import NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


class AdditiveTableau(NamedTuple):
    """An additive (IMEX) Runge-Kutta tableau: one explicit + one implicit half.

    Both halves share the node vector ``c``. ``a_explicit`` is strictly lower
    triangular (its stages are explicit), ``a_implicit`` is lower triangular with a
    possibly nonzero diagonal (those stages are solved implicitly). ``b_*`` propagate,
    ``d_*`` are the embedded (lower-order) weights for the error estimate.
    """

    c: np.ndarray
    a_explicit: np.ndarray
    a_implicit: np.ndarray
    b_explicit: np.ndarray
    b_implicit: np.ndarray
    d_explicit: np.ndarray
    d_implicit: np.ndarray


def getARKTableau(scheme: str) -> AdditiveTableau:
    if scheme == 'ARK324L2SA':
        # Kennedy-Carpenter ARK(2,3)4L[2]SA: 4 stages, implicit half order 3,
        # explicit half order 2, L[2]-stable. SUNDIALS ARKODE v7.9.0
        # (ARKODE_ARK324L2SA_{ERK,DIRK}_4_2_3). The implicit half (a_implicit,
        # b_implicit) is identical in a/b/c to the registered ESDIRK3(2)4L[2]SA;
        # only the embedded d differs (the ARK pair shares one d across both halves).
        g = 0.4358665215084589994160194511935568425293
        c = np.array([0.0,
                      0.8717330430169179988320389023871136850586,
                      0.6,
                      1.0])
        a_explicit = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.8717330430169179988320389023871136850586, 0.0, 0.0, 0.0],
            [0.52758901197630041156180797140291790433,
             0.07241098802369958843819202859708209566999, 0.0, 0.0],
            [0.3990960076760701320627260736092142797856,
             -0.437557654613519443722846363831022571942,
             1.038461646937449311660120290221808292156, 0.0],
        ])
        a_implicit = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [g, g, 0.0, 0.0],
            [0.2576482460664272457999960162840797092643,
             -0.09351476757488624521601546747763655179361, g, 0.0],
            [0.1876410243467238251612921441668043913795,
             -0.5952974735769549480478230275858851737782,
             0.9717899277217721234705114322255239398694, g],
        ])
        b = np.array([0.1876410243467238251612921441668043913795,
                      -0.5952974735769549480478230275858851737782,
                      0.9717899277217721234705114322255239398694,
                      g])
        d = np.array([0.2147402862233891404862383406484193714659,
                      -0.4851622638849390928209050808398155895845,
                      0.86872500252038755116621237682951240796,
                      0.4016969751411624011684543633618838101586])
        # Both halves carry the same propagation and embedded weights (verified in
        # the .def entries), but they are stored separately: the update applies b_exp
        # to f_exp and b_implicit to f_imp independently.
        return AdditiveTableau(c=c, a_explicit=a_explicit, a_implicit=a_implicit,
                               b_explicit=b, b_implicit=b.copy(),
                               d_explicit=d, d_implicit=d.copy())
    elif scheme == 'ARK436L2SA':
        # Kennedy-Carpenter ARK(3,4)6L[2]SA: 6 stages, implicit half order 4,
        # explicit half order 3, L[2]-stable. SUNDIALS ARKODE v7.9.0
        # (ARKODE_ARK436L2SA_{ERK,DIRK}_6_3_4), in exact rational form. The
        # implicit half is a *different* 6-stage ESDIRK design from the registered
        # standalone ESDIRK4(3)6L[2]SA (different nodes: 83/250, 31/50, 17/20 vs
        # (2-sqrt(2))/4, 5/8, 26/25), so its coefficients are carried here in full.
        c = np.array([0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0])
        a_explicit = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0],
            [-116923316275.0 / 2393684061468.0,
             -2731218467317.0 / 15368042101831.0,
             9408046702089.0 / 11113171139209.0, 0.0, 0.0, 0.0],
            [-451086348788.0 / 2902428689909.0,
             -2682348792572.0 / 7519795681897.0,
             12662868775082.0 / 11960479115383.0,
             3355817975965.0 / 11060851509271.0, 0.0, 0.0],
            [647845179188.0 / 3216320057751.0,
             73281519250.0 / 8382639484533.0,
             552539513391.0 / 3454668386233.0,
             3354512671639.0 / 8306763924573.0,
             4040.0 / 17871.0, 0.0],
        ])
        a_implicit = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
            [8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
            [5012029.0 / 34652500.0, -654441.0 / 2922500.0,
             174375.0 / 388108.0, 1.0 / 4.0, 0.0, 0.0],
            [15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
             730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 1.0 / 4.0, 0.0],
            [82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
             -2260.0 / 8211.0, 1.0 / 4.0],
        ])
        b = np.array([82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
                      -2260.0 / 8211.0, 1.0 / 4.0])
        d = np.array([4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0,
                      814220225.0 / 1159782912.0, -3700637.0 / 11593932.0,
                      61727.0 / 225920.0])
        return AdditiveTableau(c=c, a_explicit=a_explicit, a_implicit=a_implicit,
                               b_explicit=b, b_implicit=b.copy(),
                               d_explicit=d, d_implicit=d.copy())
    else:
        raise ValueError(f"Unknown ARK scheme {scheme}")


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
                    solver_diagnostics.append(solve_result.diagnostics)

                    if isinstance(solver, JFNKSolver):
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


def arkScheme(tableau_name: str, **tableau_kwargs):
    """Build an ARK scheme callable that carries its tableau, mirroring ``dirkScheme``.

    Deliberately exposes ``.arkTableau`` and **not** ``.butcherTableau``: the latter is
    what ``reuse._tableau_of`` looks for to run the *explicit*-scheme reuse analysis,
    which is the wrong model for an implicit stage. Leaving it off makes
    ``reuse.step_reuse_analysis`` report "no recorded reuse behaviour", which is the
    correct answer for a scheme that rejects ``priorStep``.
    """
    tableau = getARKTableau(tableau_name)

    def scheme(state, dt, f, *args, **kwargs):
        return ARK(state, dt, f, tableau, *args, name=tableau_name, **kwargs)

    scheme.__name__ = tableau_name
    scheme.arkTableau = tableau
    return scheme


ARK324L2SA = arkScheme('ARK324L2SA')
ARK436L2SA = arkScheme('ARK436L2SA')
