"""Numerical stepping for runge_kutta; configuration and coefficients are supplied separately."""
import inspect


from warpSPHIntegrators.state.lifecycle import finalizeSystem, hostTime, initializeSystem, unpack_prior_step, updateStateEuler, updateStep


from warpSPHIntegrators.state.fields import state_difference

from warpSPHIntegrators.state.history import HistoryEntry

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

from torch.profiler import record_function

def RungeKuttaB(initialState, dt, f, butcherTableau, *args, **kwargs):
    """Explicit Runge-Kutta step (see `_RungeKuttaBOnce`), with one optional second pass.

    A system whose `finalize` cannot complete from what the stages stored (the compatible-energy update of
    CompSPH / CRKSPH when the pair list changed between stages) sets `rerunRequest` on the new state. The whole
    step is then run again from the untouched `initialState`, and the request is handed to
    `initialState.initialize(dt, rerun=request)` so the second pass can use what the first one found out (the
    half-step velocity). The stage trajectory must not depend on it; the system is responsible for that."""
    result = _RungeKuttaBOnce(initialState, dt, f, butcherTableau, *args, **kwargs)
    request = getattr(result.state, 'rerunRequest', None)
    if request is None:
        return result
    return _RungeKuttaBOnce(initialState, dt, f, butcherTableau, *args, _initExtra={'rerun': request}, **kwargs)

def _RungeKuttaBOnce(initialState, dt, f, butcherTableau, *args, _initExtra=None, **kwargs):
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False

    with record_function("[Integration] Butcher"):
        history = kwargs.pop('history', None)
        priorStep = kwargs.pop('priorStep', None)
        # `history=` is bookkeeping only: it records stages for the *next* call and is
        # returned on `IntegrationResult.history`, but it must never silently seed
        # `priorStep` here. Reuse costs order for every non-FSAL tableau (`reuse.py`),
        # and `integration._with_reuse_guard` only warns about that cost when
        # `priorStep` itself is in `kwargs` -- if this function derived `priorStep`
        # from `history` internally, reuse would switch on with no warning at all for
        # any caller that merely wanted history threaded, not reuse. A caller that
        # wants reuse asks for it the same way as always: pass
        # `priorStep=history.as_prior_step()` explicitly.
        # Optional stage-index injection, opt-in only: `f` (the scheme's RHS)
        # receives `stageIndex=0, 1, 2, ...` ONLY if its signature explicitly
        # declares a `stageIndex` parameter -- e.g. `schemes/deltaSPH.py`'s
        # frozen-diffusion support (Sun et al. 2017 Sec. 2), which needs to
        # know "is this the first stage of this real step" to decide whether
        # to recompute or reuse its cached diffusive terms. Every other
        # scheme's step function is completely unaffected: nothing is added
        # to `kwargs` unless `f` asks for it, so no existing caller's
        # signature or behaviour changes.
        try:
            _fParams = inspect.signature(f).parameters
        except (TypeError, ValueError):
            _fParams = {}
        _wantsStageIndex = 'stageIndex' in _fParams

        def _stageKwargs(stageIndex):
            return {**kwargs, 'stageIndex': stageIndex} if _wantsStageIndex else kwargs

        if verbose:
            print(f"[Integrator] Running Runge-Kutta with dt={dt:.4f} and scheme={butcherTableau}")
        # `stageWeights`: the weights the propagated solution combines the stages with, for a system whose
        # stage function folds its stage into a running sum (CompSPH's compatible energy). Only to `initialize`,
        # never to the stage function.
        _bMain = butcherTableau.b[0] if isinstance(butcherTableau.b, tuple) else butcherTableau.b
        initializeSystem(initialState, dt, *args, stageWeights=tuple(float(b) for b in _bMain),
                         **(_initExtra or {}), **kwargs)

        if verbose:
            print(f"[Integrator] Butcher: Starting with initial state at t={initialState.t:.4f}")
        currentState = initialState.initializeNewState(*args, **kwargs)
        # Stage 0 sits at t^n. The user's initializeNewState is not required to
        # forward `t`, so set it explicitly rather than inheriting whatever came
        # back (see NOTES.md 2.3).
        currentState.t = hostTime(initialState.t)
        if verbose:
            if priorStep is not None:
                print(f"[Integrator] Using prior step as k0")
            else:
                print(f"[Integrator] Running first step to compute k0")
        if priorStep is None:
            k0, r0 = updateStep(initialState, currentState, dt, f, *args, **_stageKwargs(0))
        else:
            k0, r0 = unpack_prior_step(priorStep, verbose)
        ks = [k0]
        rs = [r0]
        # if verbose:
            # print('[Integrator] Computed k0 with auxiliary value:', r0, 'and update:', k0)
        for ic, c in enumerate(butcherTableau.c[1:]):
            with record_function(f"[Integration] Butcher: k{ic+1} prep"):
                current_as = butcherTableau.a[ic+1,:ic+1]
                if verbose:
                    print(f"[Integrator] Preparing k{ic+1} with c={c:.4f} and a={current_as}")
                currentState = initialState.initializeNewState(*args, **kwargs)
                for i, a in enumerate(current_as):
                    if a != 0:
                        if verbose:
                            print(f"[Integrator] Updating state for k{ic+1} with a={a:.4f} and dt={dt:.4f} using k{i}")
                        currentState = updateStateEuler(currentState, ks[i], a * dt, copyState = False, **kwargs)
                currentState.t = hostTime(initialState.t + c * dt)
            with record_function(f"[Integration] Butcher: k{ic+1}"):
                if verbose:
                    print(f"[Integrator] Computing k{ic+1} with c={c:.4f} and a={current_as}")
                k, r = updateStep(initialState, currentState, dt, f, *args, **_stageKwargs(ic + 1))
                ks.append(k)
                rs.append(r)

        # The buffer the last right-hand-side evaluation ran on. `copied` fields are
        # taken from here when the final state is assembled.
        lastStageState = currentState

        with record_function("[Integration] Butcher: Update"):
            stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]

            def _next_history(new_state):
                if history is None:
                    return None
                entry = HistoryEntry(t=hostTime(initialState.t), dt=dt, update=ks[-1], aux=rs[-1])
                return history.pushed(entry)

            if not isinstance(butcherTableau.b, tuple):
                if verbose:
                    print(f"[Integrator] Updating state with b={butcherTableau.b} and dt={dt:.4f}")
                new_state = _weighted_update(initialState, ks, butcherTableau.b, dt, *args, **kwargs)
                new_state.t = hostTime(initialState.t + dt)
                if verbose:
                    print(f"[Integrator] Finalizing state at t={new_state.t:.4f} with b={butcherTableau.b} and dt={dt:.4f}")
                finalizeSystem(new_state, initialState, dt, rs, ks, butcherTableau.b,
                               *args, lastStageSystem=lastStageState, **kwargs)
                return IntegrationResult(state=new_state, stages=stages, history=_next_history(new_state))

            # Embedded pair: b[0] is the propagated solution, b[1] the lower-order
            # estimate. Only b[0] is finalized and returned as `state`; the pair is
            # reported as a state-shaped `error` (see _error_estimate).
            b_main, b_embedded = butcherTableau.b[0], butcherTableau.b[1]
            if verbose:
                print(f"[Integrator] Embedded pair: propagating b={b_main}, error estimate against b={b_embedded}")
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = hostTime(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(state=new_state, stages=stages, error=error, history=_next_history(new_state))

def _weighted_update(initialState, ks, weights, dt, *args, **kwargs):
    """y^{n+1} = y^n + dt * sum_i weights[i] * k_i, allocating one state.

    `verbose` is read out of kwargs rather than taken as a parameter: the schemes
    forward the caller's kwargs wholesale and never pop `verbose`, so a named
    parameter ahead of *args would be bound twice.
    """
    verbose = bool(kwargs.get('verbose', False))
    new_state = initialState.initializeNewState(*args, **kwargs)
    for i, b in enumerate(weights):
        if b != 0:
            if verbose:
                print(f"[Integrator] Updating state with b={b:.4f} and dt={dt:.4f} using k{i}")
            # copyState=False: accumulate in place. The clone already happened above.
            new_state = updateStateEuler(new_state, ks[i], b * dt, copyState=False, **kwargs)
    return new_state

def _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs):
    """Difference between the two solutions of an embedded pair, as a state.

    Computes ``y_main`` and ``y_embedded`` in full, the same way the propagated state
    itself is computed, then differences them field-by-field with
    ``fields.state_difference`` (NOTES.md S1) -- equivalent to the old hand-rolled
    linear combination of raw stage derivatives ``dt * sum_i (b_main[i] -
    b_embedded[i]) * k_i`` for the additive update semantics every system here uses,
    and correct even for a system whose ``apply_state_update`` is not purely linear,
    since both solutions go through it properly before being compared.
    """
    y_main = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
    y_embedded = _weighted_update(initialState, ks, b_embedded, dt, *args, **kwargs)
    error = state_difference(y_main, y_embedded)
    error.t = hostTime(initialState.t + dt)
    return error
