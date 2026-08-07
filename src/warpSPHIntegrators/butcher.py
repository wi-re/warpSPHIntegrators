from typing import Union, Tuple, NamedTuple
from .util import (
    applyStateUpdate,
    finalizeSystem,
    initializeSystem,
    unpack_prior_step,
    updateStateEuler,
    updateStep,
)
import numpy as np
from .specs import IntegrationResult, StageResult, blend_state, explicit_step
from torch.profiler import record_function


class butcherTableau(NamedTuple):
    a: np.array
    b: Union[np.array,Tuple[np.array, np.array]]
    c: np.array


def RungeKuttaB(initialState, dt, f, butcherTableau, *args, **kwargs):
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False

    with record_function("[Integration] Butcher"):
        priorStep = kwargs.pop('priorStep', None)
        if verbose:
            print(f"[Integrator] Running Runge-Kutta with dt={dt:.4f} and scheme={butcherTableau}")
        initializeSystem(initialState, dt, *args, **kwargs)
        
        if verbose:
            print(f"[Integrator] Butcher: Starting with initial state at t={initialState.t:.4f}")
        currentState = initialState.initializeNewState(*args, **kwargs)
        # Stage 0 sits at t^n. The user's initializeNewState is not required to
        # forward `t`, so set it explicitly rather than inheriting whatever came
        # back (see NOTES.md 2.3).
        currentState.t = float(initialState.t)
        if verbose:
            if priorStep is not None:
                print(f"[Integrator] Using prior step as k0")
            else:
                print(f"[Integrator] Running first step to compute k0")
        if priorStep is None:
            k0, r0 = updateStep(initialState, currentState, dt, f, *args, **kwargs)
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
                currentState.t = float(initialState.t + c * dt)
            with record_function(f"[Integration] Butcher: k{ic+1}"):
                if verbose:
                    print(f"[Integrator] Computing k{ic+1} with c={c:.4f} and a={current_as}")
                k, r = updateStep(initialState, currentState, dt, f, *args, **kwargs)
                ks.append(k)
                rs.append(r)
        
        # The buffer the last right-hand-side evaluation ran on. `copied` fields are
        # taken from here when the final state is assembled.
        lastStageState = currentState

        with record_function("[Integration] Butcher: Update"):
            stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]

            if not isinstance(butcherTableau.b, tuple):
                if verbose:
                    print(f"[Integrator] Updating state with b={butcherTableau.b} and dt={dt:.4f}")
                new_state = _weighted_update(initialState, ks, butcherTableau.b, dt, *args, **kwargs)
                new_state.t = float(initialState.t + dt)
                if verbose:
                    print(f"[Integrator] Finalizing state at t={new_state.t:.4f} with b={butcherTableau.b} and dt={dt:.4f}")
                finalizeSystem(new_state, initialState, dt, rs, ks, butcherTableau.b,
                               *args, lastStageSystem=lastStageState, **kwargs)
                return IntegrationResult(state=new_state, stages=stages)

            # Embedded pair: b[0] is the propagated solution, b[1] the lower-order
            # estimate. Only b[0] is finalized and returned as `state`; the pair is
            # reported as a state-shaped `error` (see _error_estimate).
            b_main, b_embedded = butcherTableau.b[0], butcherTableau.b[1]
            if verbose:
                print(f"[Integrator] Embedded pair: propagating b={b_main}, error estimate against b={b_embedded}")
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(state=new_state, stages=stages, error=error)


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

    Returns a state of the same type as the integrated one whose integrated fields
    hold ``y_main - y_embedded = dt * sum_i (b_main[i] - b_embedded[i]) * k_i``.
    The initial state cancels out, so it is scaled away with ``self_scale=0`` on the
    first contribution rather than being subtracted afterwards.
    """
    db = np.asarray(b_main, dtype=float) - np.asarray(b_embedded, dtype=float)
    error = initialState.initializeNewState(*args, **kwargs)
    error.t = float(initialState.t + dt)
    zeroed = False
    for i, d in enumerate(db):
        if d == 0:
            continue
        blend = blend_state(self_scale=0.0) if not zeroed else None
        applyStateUpdate(error, ks[i], explicit_step(float(d) * dt, blend=blend), **kwargs)
        zeroed = True
    if not zeroed:
        # The two weight vectors agree; the estimate is identically zero.
        applyStateUpdate(error, ks[0], explicit_step(0.0, blend=blend_state(self_scale=0.0)), **kwargs)
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
SSPRK3          = butcherScheme('SSPRK3')
RungeKutta4     = butcherScheme('RK4')
RungeKutta4alt  = butcherScheme('RK4alt')
Nystrom5th      = butcherScheme('Nystrom5')
BogackiShampine = butcherScheme('BogackiShampine')
DormandPrince   = butcherScheme('DormandPrince')
CashKarp        = butcherScheme('CashKarp')
