import inspect
from typing import Union, Tuple, NamedTuple
from .util import (
    finalizeSystem,
    hostTime,
    initializeSystem,
    unpack_prior_step,
    updateStateEuler,
    updateStep,
)
import numpy as np
from .fields import state_difference
from .history import HistoryEntry
from .specs import IntegrationResult, StageResult
from torch.profiler import record_function


class butcherTableau(NamedTuple):
    a: np.array
    b: Union[np.array,Tuple[np.array, np.array]]
    c: np.array


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


def getButcherTableau(scheme, alpha = 1/2, beta = 2/3):
    if scheme == 'forwardEuler':
        return butcherTableau(
            a = np.array([[0]]),
            b = np.array([1]),
            c = np.array([0])
        )
    elif scheme == 'generic2nd':
        return butcherTableau(
            a = np.array([[0, 0], [alpha, 0]]),
            b = np.array([1 - 1/(2*alpha), 1/(2*alpha)]),
            c = np.array([0, alpha])
        )
    elif scheme == 'midpoint':
        return butcherTableau(
            a = np.array([[0, 0], [1/2, 0]]),
            b = np.array([0, 1]),
            c = np.array([0, 1/2])
        )
    elif scheme == 'heunsMethod':
        return butcherTableau(
            a = np.array([[0, 0], [1, 0]]),
            b = np.array([1/2, 1/2]),
            c = np.array([0, 1])
        )
    elif scheme == 'ralston':
        return butcherTableau(
            a = np.array([[0, 0], [2/3, 0]]),
            b = np.array([1/4, 3/4]),
            c = np.array([0, 2/3])
        )
    elif scheme == 'generic3rd':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [alpha, 0, 0],
                          [beta / alpha * (beta - 3 * alpha * (1-alpha)) / (3 * alpha - 2), - beta/alpha *(beta - alpha) / (3 *alpha - 2), 0]]),
            b = np.array([1 - (3 * alpha + 3 * beta -2) / ( 6 * alpha * beta), (3 * beta - 2) / (6 * alpha * (beta - alpha)), (2 -3 *alpha / (6 * beta * (beta - alpha)))]),
            c = np.array([0, alpha, beta])
        )
    elif scheme == 'RK3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/2, 0, 0],
                          [-1, 2, 0]]),
            b = np.array([1/6, 2/3, 1/6]),
            c = np.array([0, 1/2, 1])
        )
    elif scheme == 'Heun3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/3, 0, 0],
                          [0, 2/3, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 1/3, 2/3])
        )
    elif scheme == 'ralston3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [1/2, 0, 0],
                          [0, 3/4, 0]]),
            b = np.array([2/9, 1/3, 4/9]),
            c = np.array([0, 1/2, 3/4])
        )
    elif scheme == 'Wray3':
        return butcherTableau(
            a = np.array([[0, 0, 0], 
                          [8/15, 0, 0],
                          [1/4, 5/12, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 8/15, 2/3])
        )
    elif scheme == 'SSPRK3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1/4, 1/4, 0]]),
            b = np.array([1/6, 1/6, 2/3]),
            c = np.array([0, 1, 1/2])
        )
    elif scheme == 'SSPRK104':
        # Shu's 10-stage order-4 SSP method (Shu, "Total variation diminishing
        # Runge-Kutta time discretizations", J. Comput. Phys. 169 (2001)
        # 208-228, Sec. 4.2): a 5+5 stage cascade, full step (c[-1] = 1).
        # Measured: order 4 on all three test problems, exact SSP coefficient
        # 6.0 (stage 2's constant coefficient 1 - mu/6 is the binding
        # constraint), and a negative real-axis stability interval of
        # [-13.916, 0] (OrdinaryDiffEq.jl records 13.917). Row sums == c;
        # tests/test_tvd.py pins the SSP boundary and the real-axis interval.
        return butcherTableau(
            a = np.array([[0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     0,       0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     1/6,     0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     1/6,     1/6,     0,       0,       0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    0,       0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     1/6,     0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     1/6,     1/6,     0]]),
            b = np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]),
            c = np.array([0, 1/6, 1/3, 1/2, 2/3, 1/3, 1/2, 2/3, 5/6, 1])
        )
    # The OrdinaryDiffEq wave-1 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12):
    # Ruuth's globally optimized methods SSPRK53/63/73/83 (order 3) and SSPRK54
    # (order 4) -- Ruuth, "Global optimization of explicit strong-stability-
    # preserving Runge-Kutta methods", Mathematics of Computation 75.253 (2006)
    # 183-207 -- plus the Kubatko-Yeager-Ketcheson DG-optimized methods
    # KYKSSPRK42 and KYK2014DGSSPRK_3S2 (order 2) -- "Optimal strong-stability-
    # preserving Runge-Kutta time discretizations for discontinuous Galerkin
    # methods", J. Sci. Comput. 60 (2014) 313-344. Coefficients transcribed from
    # OrdinaryDiffEq v7.8.1 (OrdinaryDiffEqSSPRK/ssprk_caches.jl alphas/betas,
    # ssprk_perform_step.jl recurrences), converted from the Shu-Osher form to
    # Butcher form (stage states as (u_i - u^n)/dt; c = row sums of a, which
    # match ODE's stored c to 1e-13). All are plain single-step methods, none
    # FSAL, so no first-stage reuse. tests/test_tvd.py pins each SSP boundary
    # and real-axis interval.
    elif scheme == 'SSPRK53':
        # Measured: order 3, SSP coefficient 2.6506 (ODE publishes 2.65),
        # real-axis interval [-6.214, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.377268915331368, 0.0, 0.0, 0.0],
                          [0.24299522053739583, 0.24299522053739583, 0.242995220537396, 0.0, 0.0],
                          [0.15358906769512648, 0.15358906769512648, 0.1535890676951266, 0.23845893284629, 0.0]]),
            b = np.array([0.20673402086480447, 0.20673402086480447, 0.11709725184184414, 0.18180256012013943, 0.287632146308408]),
            c = np.array([0.0, 0.377268915331368, 0.754537830662736, 0.7289856616121877, 0.6992261359316696])
        )
    elif scheme == 'SSPRK63':
        # Measured: order 3, SSP coefficient 3.5184 (ODE publishes 3.518),
        # real-axis interval [-7.213, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.284220721334261, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.284220721334261, 0.284220721334261, 0.0, 0.0, 0.0],
                          [0.1487128616603831, 0.12071378576592966, 0.12071378576592966, 0.12071378576593, 0.0, 0.0],
                          [0.1487128616603831, 0.12071378576592966, 0.12071378576592966, 0.12071378576593, 0.284220721334261, 0.0]]),
            b = np.array([0.16974662234923632, 0.14609361068522916, 0.10197638641686799, 0.10197638641686826, 0.24010349706589984, 0.2401034970659]),
            c = np.array([0.0, 0.284220721334261, 0.568441442668522, 0.852662164002783, 0.5108542189581724, 0.7950749402924334])
        )
    elif scheme == 'SSPRK73':
        # Measured: order 3, SSP coefficient 4.2879 (ODE publishes 4.2879; ODE
        # stores alpha73/alpha76 at 6 digits, so its coefficients carry that
        # slack), real-axis interval [-8.888, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.233213863663009, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.233213863663009, 0.233213863663009, 0.0, 0.0, 0.0, 0.0],
                          [0.19007802386584471, 0.19007802386584471, 0.19007802386584471, 0.190078023865845, 0.0, 0.0, 0.0],
                          [0.16930787981247397, 0.0958849178781437, 0.0958849178781437, 0.09588491787814384, 0.117644805593912, 0.0, 0.0],
                          [0.16930787981247397, 0.0958849178781437, 0.0958849178781437, 0.09588491787814384, 0.117644805593912, 0.233213863663009, 0.0]]),
            b = np.array([0.17698931516532443, 0.11239171983253872, 0.11239171983253872, 0.08435964663410883, 0.10350401760632937, 0.205181790464579, 0.205181790464579]),
            c = np.array([0.0, 0.233213863663009, 0.466427727326018, 0.699641590989027, 0.7603120954633792, 0.5746074390408172, 0.8078213027038262])
        )
    elif scheme == 'SSPRK83':
        # Measured: order 3, SSP coefficient 5.1071 (ODE publishes 5.107),
        # real-axis interval [-10.986, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0],
                          [0.1132986712473457, 0.11213375462167291, 0.11213375462167291, 0.11213375462167291, 0.112133754621673, 0.0, 0.0, 0.0],
                          [0.11364964986110604, 0.11165673643345277, 0.11165673643345277, 0.11165673643345277, 0.11165673643345285, 0.194971062960412, 0.0, 0.0],
                          [0.14221023579187003, 0.14091014951805214, 0.12047209837964422, 0.07283978741985271, 0.07283978741985275, 0.12719027290864166, 0.127733653231944, 0.0]]),
            b = np.array([0.14221023579187003, 0.14091014951805214, 0.12047209837964422, 0.07283978741985271, 0.07283978741985275, 0.12719027290864166, 0.127733653231944, 0.195804015330143]),
            c = np.array([0.0, 0.195804015330143, 0.391608030660286, 0.587412045990429, 0.783216061320572, 0.5618336897340375, 0.7552476585553292, 0.8041959846698575])
        )
    elif scheme == 'SSPRK54':
        # Measured: order 4, SSP coefficient 1.7693 (all-stage upwind model; the
        # final map alone is convex to 1.8611), real-axis interval [-5.331, 0].
        # ODE's published 1.508 is the paper's coefficient, which the library's
        # stage-map convexity measurement does not reproduce. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.39175222657189, 0.0, 0.0, 0.0, 0.0],
                          [0.2176690962611688, 0.368410593050371, 0.0, 0.0, 0.0],
                          [0.0826920866578106, 0.13995850219189537, 0.251891774271694, 0.0, 0.0],
                          [0.06796628363711477, 0.11503469850463158, 0.20703489859738564, 0.544974750228521, 0.0]]),
            b = np.array([0.1468118760847866, 0.2484829094449762, 0.10425883033198097, 0.2744389009013507, 0.226007483236906]),
            c = np.array([0.0, 0.39175222657189, 0.5860796893115398, 0.47454236312139997, 0.935010630967653])
        )
    elif scheme == 'KYKSSPRK42':
        # Measured: order 2, SSP coefficient 2.2838 (all-stage upwind model),
        # real-axis interval [-4.954, 0]. ODE's published 2.459 matches the
        # FINAL-map-only convexity CFL (2.4595), not the stage maps, which are
        # the binding constraint here. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.406584463657504, 0.0, 0.0, 0.0],
                          [0.24606229845682182, 0.246062298456822, 0.0, 0.0],
                          [0.25901120328686006, 0.24537398664540927, 0.405447122055692, 0.0]]),
            b = np.array([0.20997096222437225, 0.1833284972503433, 0.30292539407757624, 0.303775146447707]),
            c = np.array([0.0, 0.406584463657504, 0.4921245969136438, 0.9098323119879613])
        )
    elif scheme == 'KYK2014DGSSPRK_3S2':
        # Measured: order 2, SSP coefficient 1.8939, real-axis interval
        # [-4.267, 0]. ODE's published 0.8417 is the paper's DG-context
        # coefficient, not the upwind-model stage-map convexity CFL. Row sums
        # == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0],
                          [0.528005024856522, 0.0, 0.0],
                          [0.4818821386339936, 0.481882138633993, 0.0]]),
            b = np.array([0.3384803991135379, 0.3156535616530465, 0.345866039233415]),
            c = np.array([0.0, 0.528005024856522, 0.9637642772679866])
        )
    # The OrdinaryDiffEq wave-2 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12):
    # the embedded 3(2) pairs SSPRK43 and SSPRK432 (order-3 main / order-2
    # estimate, 4 stages) and the high-stage SSPRK932 (order 3, 9 stages,
    # fixed-step only here). Coefficients transcribed from OrdinaryDiffEq v7.8.1
    # (OrdinaryDiffEqSSPRK/ssprk_perform_step.jl recurrences), which are already
    # in Butcher form here (exact rationals; c = row sums of a).
    #   SSPRK43   -- Kraaijevanger, "Contractivity of Runge-Kutta methods", BIT
    #                31.3 (1991) 482-528 (main); Conde, Fekete & Shadid,
    #                arXiv:1806.08693 (embedded pair).
    #   SSPRK432  -- Gottlieb, Ketcheson & Shu, "Strong Stability Preserving
    #                Runge-Kutta and Multistep Time Discretizations", World
    #                Scientific (2011), Example 6.1 (low-storage 3(2) variant).
    #   SSPRK932  -- same GKS (2011) book; ODE's docstring suggests SSPRK43
    #                instead. Registered FIXED-STEP only: ODE's adaptive
    #                estimate (uprev + 6*u6 + 6*dt*k)/7 - u_final has leading
    #                term (5/7)*dt*u'(t) (the 1/7, 6/7, 6/7 coefficients sum
    #                to 13/7 -- utilde is not an embedded solution), i.e.
    #                O(dt) instead of the O(dt^3) a 3(2) pair would need, so
    #                the library does not register it as an error estimate.
    # All three are non-FSAL (no first-stage reuse). tests/test_tvd.py pins each
    # SSP boundary and real-axis interval; the 3(2) pairs get estimate-rate and
    # adaptive tests via tests/test_embedded.py.
    elif scheme == 'SSPRK43':
        # 3(2) pair. Measured: order 3 (estimate O(h^3)), SSP coefficient 2.0
        # (ODE publishes 2.0), real-axis interval [-5.149, 0]. b = (b_main,
        # b_embed) with b_embed = (1/4, 1/4, 1/4, 1/4). Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [1/2, 0.0, 0.0, 0.0],
                          [1/2, 1/2, 0.0, 0.0],
                          [1/6, 1/6, 1/6, 0.0]]),
            b = (np.array([1/6, 1/6, 1/6, 1/2]),
                 np.array([1/4, 1/4, 1/4, 1/4])),
            c = np.array([0.0, 1/2, 1.0, 1/2])
        )
    elif scheme == 'SSPRK432':
        # 3(2) pair, low-storage variant of SSPRK43 (same a, c, b_main).
        # Measured: order 3 (estimate O(h^3)), SSP coefficient 2.0 (ODE
        # publishes 2.0), real-axis interval [-5.149, 0]. b_embed =
        # (1/3, 1/3, 1/3, 0). Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [1/2, 0.0, 0.0, 0.0],
                          [1/2, 1/2, 0.0, 0.0],
                          [1/6, 1/6, 1/6, 0.0]]),
            b = (np.array([1/6, 1/6, 1/6, 1/2]),
                 np.array([1/3, 1/3, 1/3, 0.0])),
            c = np.array([0.0, 1/2, 1.0, 1/2])
        )
    elif scheme == 'SSPRK932':
        # Order-3, 9 stages, FIXED-STEP (no embedded b: see group comment).
        # Measured: order 3, SSP coefficient 6.0 (ODE publishes 6.0), real-axis
        # interval [-13.290, 0]. A 6-stage dt/6 cascade followed by three
        # recovery stages. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 0.0,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 1/6,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 1/6,  1/6,  0.0]]),
            b = np.array([1/6, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 1/6, 1/6]),
            c = np.array([0.0, 1/6, 1/3, 1/2, 2/3, 5/6, 1/2, 2/3, 5/6])
        )
    # The OrdinaryDiffEq wave-5 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13):
    # Higueras & Roldan's new third-order low-storage SSP methods (arXiv:1809.04807).
    # Screened before registration: all three are algebraically DISTINCT one-step
    # maps from SSPRK53 (different stages and final b-row; one-step-map diff
    # O(dt^4) ~ 1.5e-4..4.6e-3 per unit dt^4, not ~1e-16), so they are their own
    # methods, not storage rearrangements of it (NOTES.md S3.25). ODE's storage
    # saving (1-2 temporaries) does not apply to this driver, which builds each
    # stage state fresh from the a-row. Coefficients transcribed from
    # OrdinaryDiffEq v7.8.1 (ssprk_caches.jl alphas/betas,
    # ssprk_perform_step.jl recurrences), converted to Butcher form as in wave 1
    # (row sums == ODE's stored c to 1e-15; order-3 conditions to <=5e-15).
    # None are FSAL.
    elif scheme == 'SSPRK53_2N1':
        # Measured: order 3, SSP coefficient 2.1808 (ODE publishes 2.18),
        # real-axis interval [-5.527, 0]. Row sums == c. c_3 = 1.0053 > 1: stage
        # 3 evaluates just past the step end, as ODE does.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.0, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.291111420073766, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.291111420073766, 0.270612601278217, 0.0, 0.0],
                          [0.1901117921952908, 0.12476933240758091, 0.11598361065328995, 0.110577759392786, 0.0]]),
            b = np.array([0.1901117921952908, 0.12476933240758091, 0.11598361065328995, 0.110577759392786, 0.458557505351052]),
            c = np.array([0.0, 0.443568244942995, 0.734679665016762, 1.005292266294979, 0.541442494648948])
        )
    elif scheme == 'SSPRK53_2N2':
        # Measured: order 3, SSP coefficient 2.1487 (ODE publishes 2.148),
        # real-axis interval [-7.263, 0] -- the largest of the four despite the
        # smallest SSP coefficient. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.465388589249323, 0.0, 0.0, 0.0, 0.0],
                          [0.465388589249323, 0.465388589249323, 0.0, 0.0, 0.0],
                          [0.14783400776685549, 0.14783400776685549, 0.124745797313998, 0.0, 0.0],
                          [0.14783400776685549, 0.14783400776685549, 0.124745797313998, 0.465388589249323, 0.0]]),
            b = np.array([0.14114733153392192, 0.14114733153392192, 0.11910342333890192, 0.4443386098445868, 0.154263303748666]),
            c = np.array([0.0, 0.465388589249323, 0.930777178498646, 0.42041381284771, 0.885802402097033])
        )
    elif scheme == 'SSPRK53_H':
        # Measured: order 3, SSP coefficient 2.6506 (ODE publishes 2.65),
        # real-axis interval [-6.213, 0] (the same as SSPRK53's to measurement
        # tolerance). Shares SSPRK53's first two nodes (c1, c2) but has distinct
        # later stages. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.377268915331368, 0.0, 0.0, 0.0],
                          [0.26081197914449766, 0.26081197914449766, 0.260811979144498, 0.0, 0.0],
                          [0.21915343633198697, 0.11709725184184361, 0.11709725184184376, 0.169383144652957, 0.0]]),
            b = np.array([0.21915343633198697, 0.11709725184184361, 0.11709725184184376, 0.169383144652957, 0.377268915331368]),
            c = np.array([0.0, 0.377268915331368, 0.754537830662737, 0.782435937433493, 0.622731084668631])
        )
    elif scheme == 'RK4':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0],
                          [1/2, 0, 0, 0],
                          [0, 1/2, 0, 0],
                          [0, 0, 1, 0]]),
            b = np.array([1/6, 1/3, 1/3, 1/6]),
            c = np.array([0, 1/2, 1/2, 1])
        )
    elif scheme == 'RK4alt':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0], 
                          [1/3, 0, 0, 0],
                          [-1/3, 1, 0, 0],
                          [1, -1, 1, 0]]),
            b = np.array([1/8, 3/8, 3/8, 1/8]),
            c = np.array([0, 1/3, 2/3, 1])
        )
    elif scheme == 'Nystrom5':
        return butcherTableau(
            a = np.array([[0,       0,      0,     0, 0,0],
                          [1/3,     0,      0,     0, 0, 0],
                          [4/25, 6/25,      0,     0, 0, 0],
                          [1/4,     -3,  15/4,     0, 0, 0],
                          [2/27,  10/9, -50/81, 8/81, 0, 0],
                          [2/25, 12/25,   2/15, 8/75, 0 ,0]]),
            b = np.array([23/192, 0, 125/192,  0, -27/64, 125/192]),
            c = np.array([0, 1/3, 2/5, 1, 2/3, 4/5])
        )
    # The OrdinaryDiffEq wave-6 tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): the
    # fixed-step workhorses of LowOrderRK (the embedded pairs follow below).
    # Coefficients are the exact Float64s ODE stores in its ConstantCache
    # (extracted with a local Julia probe, transcribed by script, zero hand
    # entry), and every tableau reproduces ODE's own perform_step bit-for-bit in
    # a one-step cross-check (max|diff| = 0.0 vs solve(..., adaptive=false)).
    # Plain tableaus on the generic RK driver.
    elif scheme == 'Alshina3':
        # Alshina et al. (2008) "3rd order, 3-stage method with optimal
        # parameters". Registered fixed-step: ODE's b2tilde estimate is not a
        # valid embedded branch (its implied map is not a solution). Measured:
        # order 3, SSP coefficient 1.0, real-axis interval [-2.512, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0],
                          [0.0, 0.75, 0.0],
            ]),
            b = np.array([0.2222222222222222, 0.3333333333333333, 0.4444444444444444]),
            c = np.array([0.0, 0.5, 0.75])
        )
    elif scheme == 'Alshina6':
        # Alshina et al. (2008) "6th order, 7-stage method with optimal
        # parameters". Non-monotone but in-range nodes (c5 = 0.27639, c6 =
        # 0.72361). Measured: order 6, SSP coefficient 0.0, real-axis interval
        # [-2.904, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0267857142857142, -0.3125, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.9349206349206349, 0.2777777777777778, -0.35555555555555557, 0.0, 0.0, 0.0, 0.0],
                          [0.18002567144208434, 0.14737940683961614, 0.016070163293314288, -0.0670820393249937, 0.0, 0.0, 0.0],
                          [-0.07979765856603138, 0.025290591998992584, -0.3516326202266813, 0.3207294901687516, 0.8090169943749475, 0.0, 0.0],
                          [0.4988599356197352, -0.8633499941930429, 1.6778122846668349, -1.2682372542187894, -0.42705098312484235, 1.381966011250105, 0.0],
            ]),
            b = np.array([0.08333333333333333, 0.0, 0.0, 0.0, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333]),
            c = np.array([0.0, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 0.276393202250021, 0.7236067977499789, 1.0])
        )
    elif scheme == 'Ralston4':
        # Ralston (1962) "4th order Ralston method with minimum truncation
        # error for 4-stage explicit Runge-Kutta". Measured: order 4, SSP
        # coefficient 0.7209, real-axis interval [-2.784, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.4, 0.0, 0.0, 0.0],
                          [0.29697760924775357, 0.15875964497103584, 0.0, 0.0],
                          [0.21810038822592004, -3.0509651486929306, 3.8328647604670105, 0.0],
            ]),
            b = np.array([0.1747602822626904, -0.551480662878733, 1.2055355993965235, 0.17118478121951902]),
            c = np.array([0.0, 0.4, 0.4557372542187894, 1.0])
        )
    elif scheme == 'RKO65':
        # Tsitouras' Runge-Kutta-Oliver 6-stage 5th-order method, with a FREE
        # first stage: k1 = f(t + (2/3) dt, y^n), so the first row of a is all
        # zero and the row sums are NOT c (structural, as in ODE). Measured:
        # order 5, SSP coefficient 0.0, real-axis interval [-4.164, 0].
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.875, 2.625, 0.0, 0.0, 0.0, 0.0],
                          [-9.0, 10.714285714285714, -0.7142857142857143, 0.0, 0.0, 0.0],
                          [-3.0, 3.9150857142857145, -0.13028571428571428, 0.0152, 0.0, 0.0],
                          [0.0, 0.3236842105263158, 2.5, 0.15, -1.9736842105263157, 0.0],
            ]),
            b = np.array([0.0, 0.40601503759398494, 1.5238095238095237, 0.05555555555555555, -1.0964912280701755, 0.1111111111111111]),
            c = np.array([0.6666666666666666, 0.16666666666666666, 0.75, 1.0, 0.8, 1.0])
        )
    elif scheme == 'RKM':
        # Mead-Renaut 6-stage 1-2-1-1-1 low-storage method, "designed to have
        # good stability properties when applied to pseudospectral
        # discretizations". ODE's docstring targets that setting; on general
        # problems the order conditions give order 3, which is what registers.
        # Measured: order 3, SSP coefficient 0.7531, real-axis interval
        # [-3.224, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.167266187050662, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.484574582244783, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.536909403373491, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.082069535961948, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.853923000035347, 0.0],
            ]),
            b = np.array([-0.028289441132839, 0.46396891856471, -0.434414348751899, 0.693796229087598, 0.0, 0.30493864223243]),
            c = np.array([0.0, 0.167266187050662, 0.484574582244783, 0.536909403373491, 0.082069535961948, 0.853923000035347])
        )
    elif scheme == 'MSRK5':
        # Stepanov (arXiv:2202.08443, Fig. 3) 5th-order method, 9 stages =
        # 8 + FSAL (fixed-step). Measured: order 5, SSP coefficient 0.3226,
        # real-axis interval [-4.366, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03333333333333333, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.5, 0.0, -1.875, 1.875, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.08148148148148149, 0.0, 0.5111111111111111, -0.07407407407407407, 0.17777777777777778, 0.0, 0.0, 0.0, 0.0],
                          [0.046296296296296294, 0.0, 0.4861111111111111, -0.27314814814814814, -0.9259259259259259, 1.5, 0.0, 0.0, 0.0],
                          [0.2421875, 0.0, -1.6881696428571429, 2.080357142857143, 1.5450892857142857, -1.7075892857142858, 0.478125, 0.0, 0.0],
                          [0.06359649122807018, 0.0, 0.0, 0.2894736842105263, 0.07407407407407407, 0.275, 0.21052631578947367, 0.08732943469785576, 0.0],
            ]),
            b = np.array([0.06359649122807018, 0.0, 0.0, 0.2894736842105263, 0.07407407407407407, 0.275, 0.21052631578947367, 0.08732943469785576, 0.0]),
            c = np.array([0.0, 0.08888888888888889, 0.13333333333333333, 0.2, 0.5, 0.5333333333333333, 0.8333333333333334, 0.95, 1.0])
        )
    elif scheme == 'MSRK6':
        # Stepanov (arXiv:2202.08443, Table 4) 6th-order method, 9 stages =
        # 8 + FSAL (fixed-step). Measured: order 6, SSP coefficient 0.1009,
        # real-axis interval [-4.164, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05357142857142857, 0.0, 0.16071428571428573, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.4027777777777778, 0.0, -1.4583333333333333, 1.5555555555555556, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.30357142857142855, 0.0, 1.6607142857142858, -1.1428571428571428, 0.42857142857142855, 0.0, 0.0, 0.0, 0.0],
                          [0.14504373177842567, 0.0, -0.9948979591836735, 1.6058673469387754, -0.7024416909620991, 0.8035714285714286, 0.0, 0.0, 0.0],
                          [0.19155336771370526, 0.0, 1.5777074542897327, -2.491610017190186, 2.9752988748241913, -1.842573058290358, 0.5896233786529145, 0.0, 0.0],
                          [0.06584362139917696, 0.0, 0.0, 0.31438458660680885, 0.17666666666666667, 0.19761316872427984, 0.19761316872427984, 0.04787878787878788, 0.0],
            ]),
            b = np.array([0.06584362139917696, 0.0, 0.0, 0.31438458660680885, 0.17666666666666667, 0.19761316872427984, 0.19761316872427984, 0.04787878787878788, 0.0]),
            c = np.array([0.0, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.5, 0.6428571428571429, 0.8571428571428571, 1.0, 1.0])
        )
    elif scheme == 'SIR54':
        # Kovalnogov et al. "5th order method suited for SIR-type epidemic
        # models", 8 stages = 7 + FSAL (fixed-step): ODE's btilde estimate is
        # not a valid embedded branch (sums to 7.85, not 0), so only the
        # propagated weights register. Measured: order 5, SSP coefficient 0.0,
        # real-axis interval [-3.512, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.22499185714559425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08915336103206711, 0.23896021111688856, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.3050223986305804, -5.501775490886399, 5.1411313952931605, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.7623401292754435, -7.4480042847020265, 6.708799990212466, -0.03413773303571631, 0.0, 0.0, 0.0, 0.0],
                          [1.8741291161981875, -7.9146797771177075, 7.079854670922641, -0.024494533823827536, -0.01480947617929343, 0.0, 0.0, 0.0],
                          [0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0],
                          [0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0],
            ]),
            b = np.array([0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0]),
            c = np.array([0.0, 0.22499185714559425, 0.32811357214895565, 0.9443783030373425, 0.9889981017501664, 1.0, 1.0, 1.0])
        )
    elif scheme == 'PSRK4p7q6':
        # Aubry-Chartier (1998) pseudo-symplectic method, 6 stages (fixed-
        # step); the largest real-axis interval of the wave, [-11.41, 0].
        # Measured: order 4, SSP coefficient 0.0. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.23593376536652, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.34750735658424, -0.13561935398346, 0.0, 0.0, 0.0, 0.0],
                          [-0.20592852403227, 1.89179076622108, -0.89775024478958, 0.0, 0.0, 0.0],
                          [-0.09435493281455, 1.75617141223762, -0.96707850476948, 0.06932825997989, 0.0, 0.0],
                          [0.14157883255197, -1.17039696277833, 1.30579112376331, -2.20354136855289, 2.92656837501595, 0.0],
            ]),
            b = np.array([0.07078941627598, 0.87808570611881, -0.44887512239479, -0.44887512239479, 0.87808570611881, 0.07078941627598]),
            c = np.array([0.0, 0.23593376536652, 0.21188800260078, 0.78811199739923, 0.76406623463348, 1.0])
        )
    elif scheme == 'PSRK3p6q5':
        # Aubry-Chartier (1998) pseudo-symplectic method, 5 stages (fixed-
        # step). Measured: order 3, SSP coefficient 0.0, real-axis interval
        # [-3.548, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.13502027922909, 0.0, 0.0, 0.0, 0.0],
                          [-0.47268213605237, 1.05980250415419, 0.0, 0.0, 0.0],
                          [-1.21650460595689, 2.16217630216753, -0.37234592426536, 0.0, 0.0],
                          [0.33274443036387, -0.20882668296587, 1.87865617737921, -1.00257392477721, 0.0],
            ]),
            b = np.array([0.04113894457092, 0.26732123194414, 0.86700906289955, -0.30547139552036, 0.13000215610576]),
            c = np.array([0.0, 0.13502027922909, 0.58712036810182, 0.57332577194528, 1.0])
        )
    elif scheme == 'PSRK3p5q4':
        # Aubry-Chartier (1998) pseudo-symplectic method, 4 stages (fixed-
        # step). Measured: order 3, SSP coefficient 0.0, real-axis interval
        # [-2.784, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.375, 0.0, 0.0, 0.0],
                          [0.9166666666666666, -0.6666666666666666, 0.0, 0.0],
                          [-0.08333333333333333, 1.8333333333333333, -0.75, 0.0],
            ]),
            b = np.array([0.1111111111111111, 0.8888888888888888, -0.2222222222222222, 0.2222222222222222]),
            c = np.array([0.0, 0.375, 0.25, 1.0])
        )
    # ---- Embedded pairs -------------------------------------------------- #
    # `b` is a tuple (propagated weights, embedded lower-order weights). The first
    # entry advances the solution; the difference between the two is returned as
    # IntegrationResult.error for step size control.
    elif scheme == 'BogackiShampine':
        # Bogacki-Shampine 3(2), scipy's RK23. FSAL: c[-1] == 1 and a[-1] == b[0],
        # so the last stage of step n IS f(t^{n+1}, y^{n+1}) and can be reused as k0
        # of step n+1 at no cost in order -- 4 stages, 3 effective evaluations.
        return butcherTableau(
            a = np.array([[  0,   0,   0, 0],
                          [1/2,   0,   0, 0],
                          [  0, 3/4,   0, 0],
                          [2/9, 1/3, 4/9, 0]]),
            b = (np.array([2/9, 1/3, 4/9, 0]),
                 np.array([7/24, 1/4, 1/3, 1/8])),
            c = np.array([0, 1/2, 3/4, 1])
        )
    elif scheme == 'DormandPrince':
        # Dormand-Prince 5(4), scipy's RK45 / MATLAB's ode45. Also FSAL:
        # 7 stages, 6 effective evaluations under reuse.
        return butcherTableau(
            a = np.array([
                [          0,            0,           0,         0,            0,       0, 0],
                [       1/5,             0,           0,         0,            0,       0, 0],
                [      3/40,          9/40,           0,         0,            0,       0, 0],
                [     44/45,        -56/15,        32/9,         0,            0,       0, 0],
                [19372/6561,   -25360/2187,  64448/6561,  -212/729,            0,       0, 0],
                [ 9017/3168,       -355/33,  46732/5247,    49/176,  -5103/18656,       0, 0],
                [    35/384,             0,    500/1113,   125/192,   -2187/6784,   11/84, 0]]),
            b = (np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
                 np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])),
            c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        )
    elif scheme == 'CashKarp':
        # Cash-Karp 5(4). Not FSAL (c[-1] = 7/8), but a well-conditioned embedded
        # pair in 6 stages when Dormand-Prince is more than is needed.
        return butcherTableau(
            a = np.array([
                [          0,        0,          0,             0,        0, 0],
                [        1/5,        0,          0,             0,        0, 0],
                [       3/40,     9/40,          0,             0,        0, 0],
                [       3/10,    -9/10,        6/5,             0,        0, 0],
                [     -11/54,      5/2,     -70/27,         35/27,        0, 0],
                [1631/55296,  175/512, 575/13824, 44275/110592, 253/4096, 0]]),
            b = (np.array([37/378, 0, 250/621, 125/594, 0, 512/1771]),
                 np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4])),
            c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])
        )
    # The OrdinaryDiffEq wave-6 embedded pairs (SCHEME_EXPANSION_PLAN, 2026-10-13):
    # b_embed = b_main - btilde (or - bhat for BS5), the difference convention of
    # DiffEqBase's calculate_residuals. Coefficients and the one-step
    # bit-for-bit cross-check as for the fixed-step wave-6 tableaus above.
    elif scheme == 'Tsit5':
        # Tsitouras 5(4), ODE's recommended default for non-stiff problems
        # (free 4th order interpolant). 7 stages = 6 + FSAL. Measured: main
        # order 5, embedded 4, SSP coefficient 0.0, real-axis interval
        # [-3.506, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.008480655492356989, 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0.0, 0.0, 0.0, 0.0],
                          [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0.0, 0.0, 0.0],
                          [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0.0, 0.0],
                          [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0],
            ]),
            b = (np.array([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0]),
                 np.array([0.09824077787029101, 0.010816434459656746, 0.4720087724042376, 1.5237195812770048, -3.872426680888636, 2.7827926300289607, -0.015151515151515152])),
            c = np.array([0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0])
        )
    elif scheme == 'OwrenZen3':
        # Owren-Zennaro (1992) optimized-interpolation 3(2) (free 3rd order
        # interpolant), 4 stages = 3 + FSAL. Measured: main order 3, embedded
        # 2, SSP coefficient 0.7813, real-axis interval [-2.512, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.5217391304347826, 0.0, 0.0, 0.0],
                          [-0.18133333333333335, 0.9813333333333333, 0.0, 0.0],
                          [0.2152777777777778, 0.4592013888888889, 0.3255208333333333, 0.0],
            ]),
            b = (np.array([0.2152777777777778, 0.4592013888888889, 0.3255208333333333, 0.0]),
                 np.array([0.3888888888888889, -0.039930555555555525, 0.6510416666666666, 0.0])),
            c = np.array([0.0, 0.5217391304347826, 0.8, 1.0])
        )
    elif scheme == 'OwrenZen4':
        # Owren-Zennaro (1992) optimized-interpolation 4(3) (free 4th order
        # interpolant), 6 stages = 5 + FSAL. Measured: main order 4, embedded
        # 3, SSP coefficient 0.6444, real-axis interval [-2.872, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03214024835646457, 0.26515704894083275, 0.0, 0.0, 0.0, 0.0],
                          [0.6895990230002036, -1.6993690209647874, 1.6568288214939955, 0.0, 0.0, 0.0],
                          [-0.09002509947964493, 0.6817777777777778, -0.2402791551882461, 0.5151931435567799, 0.0, 0.0],
                          [0.08990252172070354, 0.0, 0.4360623278236915, 0.1842858372687918, 0.2897493131868132, 0.0],
            ]),
            b = (np.array([0.08990252172070354, 0.0, 0.4360623278236915, 0.1842858372687918, 0.2897493131868132, 0.0]),
                 np.array([-0.0984318711591439, 0.0, 0.9664084022038568, -0.4474751574183393, 0.5794986263736264, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.2972972972972973, 0.6470588235294118, 0.8666666666666667, 1.0])
        )
    elif scheme == 'OwrenZen5':
        # Owren-Zennaro (1992) optimized-interpolation 5(4) (free 5th order
        # interpolant), 8 stages = 7 + FSAL. ODE's perform_step applies a
        # literal k3 to stage 4 (a43 = 1) that its cache does not store; the
        # tableau includes it. Measured: main order 5, embedded 4, SSP
        # coefficient 0.0, real-axis interval [-3.192, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0625, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.25, -0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.75, 3.75, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                          [0.26895043731778423, -0.7084548104956269, 0.8658892128279884, 0.15462307371928363, 0.06184922948771345, 0.0, 0.0, 0.0],
                          [-0.02947695035460993, 0.18500664893617022, 0.4802345261121857, -0.5337849069148937, -0.013090093085106383, 0.7861107753062541, 0.0, 0.0],
                          [0.08783068783068783, 0.0, 0.3006060606060606, 0.22777777777777777, 0.027777777777777776, 0.06218596218596219, 0.2938217338217338, 0.0],
            ]),
            b = (np.array([0.08783068783068783, 0.0, 0.3006060606060606, 0.22777777777777777, 0.027777777777777776, 0.06218596218596219, 0.2938217338217338, 0.0]),
                 np.array([0.2867724867724868, 0.0, -0.6109090909090908, 2.2055555555555557, 0.1388888888888889, -1.607951307951308, 0.5876434676434676, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.25, 0.5, 0.5, 0.6428571428571429, 0.875, 1.0])
        )
    elif scheme == 'BS5':
        # Bogacki-Shampine (1996) 5(4), "an efficient Runge-Kutta (4,5) pair",
        # 8 stages = 7 + FSAL. The embedded branch is ODE's bhat (the 4th order
        # companion); ODE's btilde is a separate 5th order estimate for the
        # step controller, not a solution. Measured: main order 5, embedded
        # 4, SSP coefficient 0.5237, real-axis interval [-3.986, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07407407407407407, 0.14814814814814814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.13338192419825073, -0.47230320699708456, 0.7674927113702624, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.22895622895622897, -0.36363636363636365, 0.2937062937062937, 0.5076405076405076, 0.0, 0.0, 0.0, 0.0],
                          [0.026500355113636364, 0.23011363636363635, 0.10772747760052448, 0.1602190777972028, 0.225439453125, 0.0, 0.0, 0.0],
                          [0.18159821692916506, -0.38707982536247293, 0.41288268560074054, 0.6409913191253497, -1.0121439383514517, 1.1637515420586695, 0.0, 0.0],
                          [0.07279265873015874, 0.0, 0.28662437773692473, 0.19513621794871794, 0.008638392857142857, 0.3595655806182122, 0.07724277210884353, 0.0],
            ]),
            b = (np.array([0.07279265873015874, 0.0, 0.28662437773692473, 0.19513621794871794, 0.008638392857142857, 0.3595655806182122, 0.07724277210884353, 0.0]),
                 np.array([0.07513640873015874, 0.0, 0.27624830233206643, 0.21162660256410254, -0.010345982142857145, 0.37009189640768586, 0.07724277210884353, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.2222222222222222, 0.42857142857142855, 0.6666666666666666, 0.75, 1.0, 1.0])
        )
    elif scheme == 'Stepanov5':
        # Stepanov (Stepanov2021Embedded5) 5(4), 7 stages = 6 + FSAL.
        # Measured: main order 5, embedded 4, SSP coefficient 0.0, real-axis
        # interval [-3.496, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0621301775147929, 0.26094674556213016, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.6301020408163265, -5.207142857142857, 4.4770408163265305, 0.0, 0.0, 0.0, 0.0],
                          [2.8811124102418746, -9.263950892857142, 7.442850056689342, -0.08501157407407407, 0.0, 0.0, 0.0],
                          [3.2522024583580915, -10.435540069686411, 8.27662100358286, -0.05432098765432099, -0.03896240460022027, 0.0, 0.0],
                          [0.09774793108126441, 0.0, 0.48629574293291106, 1.4567901234567902, -2.904470161107329, 1.8636363636363635, 0.0],
            ]),
            b = (np.array([0.09774793108126441, 0.0, 0.48629574293291106, 1.4567901234567902, -2.904470161107329, 1.8636363636363635, 0.0]),
                 np.array([0.10141093474426807, 0.0, 0.4736707572990759, 1.7234567901234568, -3.775811209439528, 2.5037752314463497, -0.026502504173622706])),
            c = np.array([0.0, 0.2, 0.3230769230769231, 0.9, 0.975, 1.0, 1.0])
        )
    elif scheme == 'Alshina2':
        # Alshina et al. (2008) "2nd order, 2-stage method with optimal
        # parameters", 2(1). The embedded branch is exactly forward Euler
        # (b_main - b_embed = (-3/4, 3/4) = ODE's btilde). Measured: main
        # order 2, embedded 1, SSP coefficient 1.0, real-axis interval
        # [-2.0, 0]. Not FSAL.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0],
                          [0.6666666666666666, 0.0],
            ]),
            b = (np.array([0.25, 0.75]),
                 np.array([1.0, 0.0])),
            c = np.array([0.0, 0.666666666666666])
        )
    else:
        raise ValueError(f"Unknown scheme {scheme}")


def butcherScheme(tableau_name: str, **tableau_kwargs):
    """Build a scheme callable that carries its tableau.

    The tableau is built once here rather than on every step, and is exposed as
    ``scheme.butcherTableau`` so the reuse analysis (``integrators.reuse``) can
    inspect it without a name-to-tableau lookup table.
    """
    tableau = getButcherTableau(tableau_name, **tableau_kwargs)

    def scheme(state, dt, f, *args, **kwargs):
        return RungeKuttaB(state, dt, f, tableau, *args, **kwargs)

    scheme.__name__ = tableau_name
    scheme.butcherTableau = tableau
    return scheme



# Evaluate-Predict-Evaluate-Correct (EPEC) scheme based on pySPH code
# is equivalent to the traditional explicit midpoint method 
EPEC = butcherScheme('midpoint')
# Modified EPEC scheme based on pySPH code, uses $y^{n+1} = y^n + \frac{\Delta t}{2}\left( F(y^n) + F(y^{n+\frac{1}{2}}) \right)$
EPECmodified = butcherScheme('heunsMethod')


forwardEuler    = butcherScheme('forwardEuler')
RungeKutta2     = butcherScheme('midpoint')
midPoint        = butcherScheme('midpoint')
heunsMethod     = butcherScheme('heunsMethod')
ralston2nd      = butcherScheme('ralston')
RungeKutta3     = butcherScheme('RK3')
heunsMethod3rd  = butcherScheme('Heun3')
ralston3rd      = butcherScheme('ralston3')
Wray3rd         = butcherScheme('Wray3')
SSPRK3           = butcherScheme('SSPRK3')
SSPRK104         = butcherScheme('SSPRK104')
SSPRK53          = butcherScheme('SSPRK53')
SSPRK63          = butcherScheme('SSPRK63')
SSPRK73          = butcherScheme('SSPRK73')
SSPRK83          = butcherScheme('SSPRK83')
SSPRK54          = butcherScheme('SSPRK54')
KYKSSPRK42       = butcherScheme('KYKSSPRK42')
KYK2014DGSSPRK3S2 = butcherScheme('KYK2014DGSSPRK_3S2')
SSPRK43          = butcherScheme('SSPRK43')
SSPRK432         = butcherScheme('SSPRK432')
SSPRK932         = butcherScheme('SSPRK932')
SSPRK53_2N1      = butcherScheme('SSPRK53_2N1')
SSPRK53_2N2      = butcherScheme('SSPRK53_2N2')
SSPRK53_H        = butcherScheme('SSPRK53_H')
RungeKutta4      = butcherScheme('RK4')
RungeKutta4alt  = butcherScheme('RK4alt')
Nystrom5th      = butcherScheme('Nystrom5')
BogackiShampine = butcherScheme('BogackiShampine')
DormandPrince   = butcherScheme('DormandPrince')
CashKarp        = butcherScheme('CashKarp')
# The OrdinaryDiffEq wave-6 tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13):
# Tsit5 and the LowOrderRK workhorses (embedded pairs first, then fixed-step).
Tsit5           = butcherScheme('Tsit5')
OwrenZen3       = butcherScheme('OwrenZen3')
OwrenZen4       = butcherScheme('OwrenZen4')
OwrenZen5       = butcherScheme('OwrenZen5')
BS5             = butcherScheme('BS5')
Stepanov5       = butcherScheme('Stepanov5')
Alshina2        = butcherScheme('Alshina2')
Alshina3        = butcherScheme('Alshina3')
Alshina6        = butcherScheme('Alshina6')
MSRK5           = butcherScheme('MSRK5')
MSRK6           = butcherScheme('MSRK6')
SIR54           = butcherScheme('SIR54')
Ralston4        = butcherScheme('Ralston4')
RKO65           = butcherScheme('RKO65')
RKM             = butcherScheme('RKM')
PSRK4p7q6       = butcherScheme('PSRK4p7q6')
PSRK3p6q5       = butcherScheme('PSRK3p6q5')
PSRK3p5q4       = butcherScheme('PSRK3p5q4')
