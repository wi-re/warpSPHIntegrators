"""Coupled (block) fully implicit Runge-Kutta driver (docs/research/integrators/notes-3-18.md, Phase 6).

A DIRK solves its ``s`` stage equations as ``s`` *sequential* solves, because
``a_ij == 0`` for ``j > i`` makes stage ``i`` a function of already-known stages
alone. A *coupled* fully implicit tableau -- Gauss-Legendre 2, Radau IIA s=2 --
has nonzero off-diagonal entries, so stage ``i`` references the still-unknown
stage ``j != i`` and the ``s`` stage equations form one ``s``-by-``s`` coupled
system. This driver solves that system with a single ``NonlinearSolver`` call
(``JFNKSolver`` by default): the unknown is a ``BlockState`` (one substate per
stage, ``fields.py``), the residual map ``step_fn`` evaluates ``f`` at every
stage in one call (``s`` RHS evaluations per map evaluation), and the solver's
``norm``/``flatten_integrated``/``replace_integrated_fields`` machinery recurses
over the substate block (``fields.py``, substate-major ``'i:name'`` layout).

Diagnostics are one per step, scaled by the stage count ``s`` (a map evaluation
is ``s`` RHS evaluations), mirroring how the DIRK driver appends one entry per
stage. First-stage reuse is refused for every coupled tableau (there is no
explicit first stage to splice a previous step's derivative into); instead the
``warmStart=`` kwarg takes the previous step's converged stage *states*
(``IntegrationResult.stages``) as the block solve's initial guess.

The tableaus below are hand-derived and verified (docs/research/integrators/notes-3-18.md): Radau IIA s=2
is the collocation method at the Radau points ``{1/3, 1}`` (right endpoint
pinned, so s=1 is exactly backward Euler), order 3, L-stable, stiffly accurate;
Gauss-Legendre 2 is the collocation method at the Gauss points ``1/2 +- sqrt(3)/6``,
order 4, A-stable, symplectic for separable Hamiltonians (pinned by measurement,
``tests/test_hamiltonian.py`` -- its tableau is *not* symmetric: ``a_12 != a_21``),
not L-stable. Neither
exists in SUNDIALS ARKODE v7.9.0's published tableaus, so the coefficients come
from the collocation derivation with the order conditions and stability
functions checked symbolically and numerically (the SDIRK2 "verified by hand"
precedent, ``dirk.py``). Each carries a null-stage order-2 companion -- an
order-2 quadrature rule over ``(k0 = f(t^n, y^n), k1, k2)`` -- because the
2-stage-only order-2 pair is degenerate (it is just ``b`` itself) for both
schemes.

The scale-qualified three-stage additions use the pinned OrdinaryDiffEq
monorepo source: Gauss (order 6) has a same-node order-2 companion; Radau IIA
(order 5) has a derived unsmoothed order-3 companion. The latter does not
reproduce upstream's shifted-Jacobian error filter or adaptive controller.
"""

from warpSPHIntegrators.schemes.coefficients.coupled import BlockTableau

import dataclasses
from typing import Optional


from torch.profiler import record_function

from warpSPHIntegrators.drivers.runge_kutta import _weighted_update
from warpSPHIntegrators.state.fields import BlockState, get_reference_state, integrated_field_names, state_difference, state_norm
from warpSPHIntegrators.state.history import HistoryEntry
from warpSPHIntegrators.solvers.krylov import JFNKSolver
from warpSPHIntegrators.solvers.nonlinear import record_solve_diagnostics, NonlinearSolver
from warpSPHIntegrators.state.specs import IntegrationResult, StageResult
from warpSPHIntegrators.state.lifecycle import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


def _default_norm(rtol: float, atol: float):
    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), rtol, atol, reference=y_old)
    return norm


def fullyImplicitBlock(initialState, dt, f, tableau: BlockTableau, *args,
                       name: str = 'FullyImplicitBlock',
                       solver: Optional[NonlinearSolver] = None,
                       warmStart: Optional[list] = None, **kwargs):
    """One step of a coupled fully implicit RK method.

    ``tableau`` is a ``BlockTableau``. The ``s`` stage states are solved as one
    coupled system (``JFNKSolver`` by default); the solver's per-map-evaluation
    diagnostics are scaled by ``s`` (one map evaluation is ``s`` RHS
    evaluations) and returned as one entry in ``IntegrationResult.solver_diagnostics``.

    ``warmStart=`` takes the previous step's ``IntegrationResult.stages`` (a list
    of ``StageResult`` of length ``s``) and seeds the block solve with
    ``Y_i ~= y^n + dt * sum_j a_ij k_j(previous)`` instead of the cold guess
    ``y^n`` -- exposed as a plain flag, since warm starting helps on stiff
    problems with similar steps and costs nothing when it does not.
    Preconditioner context includes dt, tableau_a and stage_times; driver-owned
    values override caller context keys. A tableau with companion_b=None emits
    no error estimate, supporting unregistered research prototypes.
    ``priorStep`` (first-stage reuse) is always rejected: a coupled tableau has
    no explicit first stage to splice a previous step's derivative into.
    ``history=``, ``rtol=``/``atol=`` (the solve's norm tolerances), and
    ``solver_opts={}`` (forwarded to ``NonlinearSolver.solve``) work as in
    ``dirk.DIRK``.
    """
    verbose = bool(kwargs.get('verbose', False))
    solver = solver or JFNKSolver()
    history = kwargs.pop('history', None)
    priorStep = kwargs.pop('priorStep', None)
    # A coupled tableau has no explicit first stage, so there is nothing for
    # first-stage reuse to splice a previous step's derivative into.
    reject_prior_step(name, priorStep)
    if warmStart is not None:
        s = len(tableau.c)
        if len(warmStart) != s:
            raise ValueError(
                f"{name}: warmStart must have one StageResult per stage ({s}), got {len(warmStart)}")
    solver_opts = dict(kwargs.get('solver_opts', {}))
    context = dict(solver_opts.get('preconditioner_context',getattr(solver,'preconditioner_context',None)) or {})
    context.update(dt=dt,tableau_a=tableau.a,stage_times=tuple(float(initialState.t+c*dt) for c in tableau.c))
    solver_opts['preconditioner_context'] = context
    norm = _default_norm(kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-6))

    with record_function("[Integration] FullyImplicitBlock"):
        initializeSystem(initialState, dt, *args, **kwargs)
        a = tableau.a
        s = len(tableau.c)
        t_i = [float(initialState.t + c_i * dt) for c_i in tableau.c]
        box = {}

        def step_fn(Y, box=box):
            """The block residual map: Y -> (Y_i - (y^n + dt * sum_j a_ij k_j)).

            All ``s`` stage states are evaluated in one call, so the solver sees
            the coupled system as a single map on a ``BlockState``. ``box``
            remembers the last evaluation's ``(k, r, Y)`` for the Picard (non-JFNK)
            path, the same "buffer the last evaluation ran on" convention
            ``dirk.DIRK`` and ``butcher.RungeKuttaB`` use for ``lastStageState``.
            """
            ks, rs = [], []
            for j in range(s):
                Y_j = Y.states[j]
                Y_j.t = t_i[j]
                k_j, r_j = updateStep(initialState, Y_j, dt, f, *args, **kwargs)
                ks.append(k_j)
                rs.append(r_j)
            outs = []
            for i in range(s):
                out_i = initialState.initializeNewState(*args, **kwargs)
                for j in range(s):
                    a_ij = float(a[i, j])
                    if a_ij != 0:
                        out_i = updateStateEuler(out_i, ks[j], a_ij * dt, copyState=False, **kwargs)
                out_i.t = t_i[i]
                outs.append(out_i)
            box['ks'], box['rs'], box['Y'] = ks, rs, Y
            return BlockState(tuple(outs))

        if warmStart is not None:
            # Warm guess: the previous step's converged stage derivatives in place
            # of this step's unknowns, Y_i ~= y^n + dt * sum_j a_ij k_j(previous).
            warm_updates = [st.update for st in warmStart]
            guesses = []
            for i in range(s):
                guess_i = initialState.initializeNewState(*args, **kwargs)
                for j in range(s):
                    a_ij = float(a[i, j])
                    if a_ij != 0:
                        guess_i = updateStateEuler(guess_i, warm_updates[j], a_ij * dt, copyState=False, **kwargs)
                guess_i.t = t_i[i]
                guesses.append(guess_i)
        else:
            # Cold guess: the known part only, Y_i ~= y^n.
            guesses = []
            for i in range(s):
                guess_i = initialState.initializeNewState(*args, **kwargs)
                guess_i.t = t_i[i]
                guesses.append(guess_i)

        if verbose:
            print(f"[Integrator] {name}: solving the {s}-stage coupled block at "
                  f"t = {', '.join(f'{t:.4f}' for t in t_i)}"
                  f"{' (warm start)' if warmStart is not None else ''}")
        solve_result = solver.solve(step_fn, BlockState(tuple(guesses)), norm, **solver_opts)

        # One map evaluation is `s` RHS evaluations, so the solver's counters
        # count map evaluations, not force evaluations: scale them up.
        diag = record_solve_diagnostics(solve_result)
        if diag is not None:
            diag = dataclasses.replace(
                diag,
                rhs_evaluations=diag.rhs_evaluations * s,
                gmres_iterations=diag.gmres_iterations * s,
            )
        solver_diagnostics = [diag]

        # Null-stage companion (docs/research/integrators/notes-3-18.md): the 2-stage-only order-2 pair is
        # degenerate for both shipped tableaus (it is b itself), so the estimate
        # quadratures k0 = f(t^n, y^n) at c0 = 0 in addition to the two stages.
        # It is evaluated here, before the converged block is re-evaluated below,
        # so the step's last RHS evaluation is the last *stage*: the copied()
        # field contract (tests/test_copied_fields.py) is "copied from the last
        # substep in finalize", and the companion is an estimator detail that
        # must not reorder the method's stages. (k0 = f(t^n, y^n) depends only
        # on the initial state, so its value is order-independent.)
        companion_b = tableau.companion_b
        k0 = None
        if companion_b is not None and float(companion_b[0]) != 0.0:
            k0_buf = initialState.initializeNewState(*args, **kwargs)
            k0_buf.t = float(initialState.t)
            k0, _r0 = updateStep(initialState, k0_buf, dt, f, *args, **kwargs)

        if isinstance(solver, JFNKSolver) or 'ks' not in box:
            # Finite-difference / forward-mode Krylov probes ran on isolated (and,
            # for the JVP, dual) clones, so re-evaluate the converged block on this
            # driver's own buffers: one fresh buffer per substate, seeded with the
            # converged integrated fields, so copied fields get their standard
            # last-stage lifecycle (the same re-evaluation dirk.DIRK does per stage).
            ks, rs = [], []
            stage_states = []
            for i in range(s):
                buffer_i = initialState.initializeNewState(*args, **kwargs)
                solved_ref = get_reference_state(solve_result.y.states[i])
                stage_ref = get_reference_state(buffer_i)
                for field_name in integrated_field_names(solve_result.y.states[i]):
                    setattr(stage_ref, field_name, getattr(solved_ref, field_name))
                buffer_i.t = t_i[i]
                k_i, r_i = updateStep(initialState, buffer_i, dt, f, *args, **kwargs)
                ks.append(k_i)
                rs.append(r_i)
                stage_states.append(buffer_i)
        else:
            # Fixed-count Picard: the box holds the last step_fn evaluation, i.e.
            # the final iterate's stages, which satisfy the block equation to the
            # Picard residual (one iteration behind, the documented convention).
            ks, rs = box['ks'], box['rs']
            stage_states = list(box['Y'].states)

        lastStageState = stage_states[-1]
        k_all = ([k0] + ks) if k0 is not None else ks

        y_main = _weighted_update(initialState, ks, tableau.b, dt, *args, **kwargs)
        error = None
        if companion_b is not None:
            weights = companion_b if k0 is not None else companion_b[1:]
            y_embed = _weighted_update(initialState, k_all, weights, dt, *args, **kwargs)
            error = state_difference(y_main, y_embed)
            error.t = float(initialState.t + dt)
        y_main.t = float(initialState.t + dt)

        stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]

        def _next_history():
            if history is None:
                return None
            entry = HistoryEntry(t=float(initialState.t), dt=dt, update=ks[-1], aux=rs[-1])
            return history.pushed(entry)

        finalizeSystem(y_main, initialState, dt, rs, ks, tableau.b,
                       *args, lastStageSystem=lastStageState, **kwargs)
        return IntegrationResult(
            state=y_main, stages=stages, error=error, history=_next_history(),
            solver_diagnostics=solver_diagnostics)
