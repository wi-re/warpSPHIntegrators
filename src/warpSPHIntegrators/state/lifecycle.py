

from typing import Optional

from warpSPHIntegrators.state.fields import copy_finalized_fields
from warpSPHIntegrators.state.specs import ComponentUpdateSpec, PositionUpdateSpec, StageResult, StateBlend, explicit_step, semi_implicit_position_step


def verbosePrint(verbose, *args):
    if verbose:
        print(*args)


def reject_prior_step(scheme_name: str, priorStep) -> None:
    """Warn that a scheme cannot reuse a prior stage, having popped it from kwargs.

    Schemes that do not implement reuse must still ``pop`` ``priorStep`` -- otherwise
    it flows through into the user's right-hand side and lifecycle hooks and breaks
    any strict signature (see NOTES.md 2.9).
    """
    if priorStep is not None:
        import warnings
        warnings.warn(
            f"{scheme_name} does not support first-stage reuse; the supplied priorStep is ignored. "
            f"Use integrators.supports_step_reuse(scheme) to check before opting in.",
            RuntimeWarning,
            stacklevel=3,
        )


def unpack_prior_step(priorStep, verbose: bool = False):
    """Unpack a reused stage into the ``(k, r)`` pair the schemes expect.

    ``StageResult`` stores ``(aux, update)`` while every scheme works with
    ``(update, aux)``, so the two are deliberately swapped here. This lives in one
    place because three schemes used to open-code it and one of them got the order
    backwards (see NOTES.md 2.4).
    """
    if isinstance(priorStep, StageResult):
        verbosePrint(verbose, f"[Integrator] Reusing prior stage {priorStep}")
        return priorStep.update, priorStep.aux
    if isinstance(priorStep, tuple):
        verbosePrint(verbose, f"[Integrator] Reusing prior stage tuple {priorStep}")
        k, r = priorStep
        return k, r
    raise ValueError(
        f"Invalid priorStep format: expected StageResult or (k, r) tuple, got {type(priorStep).__name__}"
    )

def updateStateEuler(systemState_, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    return applyStateUpdate(systemState, systemUpdate, explicit_step(dt), **kwargs)

def updateStateSemiImplicitEuler(systemState_, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    applyVelocityUpdate(systemState, systemUpdate, explicit_step(dt), semiImplicit = True, **kwargs)
    applyPositionUpdate(systemState, systemUpdate, semi_implicit_position_step(dt), **kwargs)
    applyQuantityUpdate(systemState, systemUpdate, explicit_step(dt), **kwargs)
    # Time is owned by the integrator, not by the update helpers (see NOTES.md 2.15).
    # `updateStateEuler` never advanced `t`; this one used to, so composing the two
    # double-advanced time.
    return systemState


def _normalize_blend(blend: Optional[StateBlend]) -> StateBlend:
    if blend is None:
        return StateBlend()
    return blend


def applyStateUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_state_update'):
        return systemState.apply_state_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrate(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


def applyPositionUpdate(systemState, systemUpdate, spec: PositionUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_position_update'):
        return systemState.apply_position_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    legacy_kwargs = {}
    if spec.current_velocity_dt is not None:
        legacy_kwargs['semiImplicitScale'] = spec.current_velocity_dt
    if spec.update_velocity_dt is not None:
        legacy_kwargs['verletScale'] = spec.update_velocity_dt

    return systemState.integratePosition(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **legacy_kwargs,
        **kwargs,
    )


def applyVelocityUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_velocity_update'):
        return systemState.apply_velocity_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrateVelocity(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


def applyQuantityUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_quantity_update'):
        return systemState.apply_quantity_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrateQuantities(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


def split_return(rv):
    if isinstance(rv, tuple):
        return rv[0], rv[1:]
    return rv, None

from torch.profiler import record_function

def initializeSystem(currentSystem, dt, *args, **kwargs):
    # print('Arguments: ', args)
    # print('Kwargs: ', kwargs)
    with record_function("[Integration] Initialize"):
        if kwargs.get('_composition_substep',False):
            hook=getattr(currentSystem,'initialize_substep',None)
            return currentSystem if hook is None else hook(dt,*args,**kwargs)
        return currentSystem.initialize(dt, *args, **kwargs)

def preprocessSystem(currentSystem , initialSystem , dt, *args, **kwargs):
    with record_function("[Integration] Preprocess"):
        return currentSystem.preprocess(initialSystem, dt, *args, **kwargs)

def postprocessSystem(initialSystem , currentSystem , dt, r, *args, **kwargs):
    with record_function("[Integration] Postprocess"):
        return initialSystem.postprocess(currentSystem, dt, r, *args, **kwargs)

def finalizeSystem(currentSystem , initialSystem , dt, *args, lastStageSystem=None, **kwargs):
    """Assemble the final state: carry `copied` fields over, then run the user's hook.

    `lastStageSystem` is the stage buffer the final right-hand-side evaluation ran on.
    Fields declared `copied()` are recomputed per stage rather than integrated, so the
    final state takes them from there; everything else is left alone. Passing None
    skips the copy, which is what schemes whose final state *is* the last stage buffer
    (PEFRL) should do.
    """
    with record_function("[Integration] Finalize"):
        copy_finalized_fields(currentSystem, lastStageSystem)
        if kwargs.get('_composition_substep',False):
            return currentSystem
        return currentSystem.finalize(initialSystem, dt, *args, **kwargs)


def updateStep(initialState, currentState, dt, f, *args, **kwargs):
    with record_function("[Integration] Update Step"):
        preprocessSystem(currentState, initialState, dt, *args, **kwargs)
        k, r = split_return(f(currentState, dt, *args, **kwargs))
        postprocessSystem(initialState, currentState, dt, r, *args, **kwargs)
        return k, r
    

# from warpSPHIntegrators.integration import *


# --- host-time deferral (CUDA-graph capture of a whole step) -----------------
# A step captured into a CUDA graph must not read device values back to the
# host. The integrators keep stage times as Python floats (`float(state.t +
# dt)`), which syncs whenever `dt` is a device tensor. Inside
# `deferHostTime()` those conversions are skipped (the time stays a tensor);
# the capturing caller restores the host-side time itself after each replay.
# Outside the context behaviour is unchanged.
import contextlib as _contextlib

_DEFER_HOST_TIME = [False]


@_contextlib.contextmanager
def deferHostTime():
    prev = _DEFER_HOST_TIME[0]
    _DEFER_HOST_TIME[0] = True
    try:
        yield
    finally:
        _DEFER_HOST_TIME[0] = prev


def hostTime(value):
    """`float(value)`, unless inside `deferHostTime()` (then `value` as is)."""
    return value if _DEFER_HOST_TIME[0] else float(value)


def hostTimeDeferred() -> bool:
    return _DEFER_HOST_TIME[0]
