import copy
import torch
from typing import NamedTuple, Optional, List, Tuple, Union

from .fields import copy_finalized_fields
from .specs import (
    ComponentUpdateSpec,
    PositionUpdateSpec,
    StageResult,
    StateBlend,
    blend_state,
    explicit_step,
    semi_implicit_position_step,
    verlet_position_step,
)


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


from enum import Enum
import torch

from .enums import IntegrationSchemeType, SCHEME_FAMILY
from typing import Callable
class IntegrationScheme(NamedTuple):
    function: Callable
    name: str
    identifier: IntegrationSchemeType
    order: int

    dissipation: bool = False

    #: Convergence order retained when `priorStep` reuse is used, or None when the
    #: scheme does not implement reuse at all. Filled in at registration time from
    #: `integrators.reuse.step_reuse_analysis`; see that module for the derivation.
    reuse_order: Optional[int] = None
    #: True when the tableau is FSAL, so reuse is exact rather than merely lossless.
    fsal: bool = False

    # ------------------------------------------------------------------ #
    # NOTES.md S3: metadata a driver needs to tell schemes apart without    #
    # special-casing them by name. All default to the "plain explicit RK"  #
    # answer, so every scheme registered before this landed is unaffected. #
    # ------------------------------------------------------------------ #

    #: True for a scheme whose stage equations require solving for an unknown that
    #: appears on both sides (DIRK, fully implicit RK, BDF/Adams-Moulton). False for
    #: every explicit scheme and for linear multistep predictors (Adams-Bashforth).
    implicit: bool = False
    #: How many past-step states/derivatives a step needs beyond the current one: 1
    #: for every one-step method (RK, DIRK), k for a k-step linear multistep method.
    steps: int = 1
    #: True when the last stage IS the step (`a[-1] == b`), the implicit analogue of
    #: FSAL (`reuse.tableau_reuse_analysis` already detects this for explicit
    #: tableaus; this field is what lets an implicit driver ask the same question
    #: without re-deriving it from `a`/`b` at call time).
    stiffly_accurate: bool = False
    #: Linear stability region, or None for a scheme with no useful one to name
    #: (mixed-order embedded pairs, symplectic-but-not-A/L-stable methods). 'A':
    #: stable for all Re(lambda*dt) <= 0. 'L': A-stable and additionally damps
    #: infinitely stiff modes to zero in one step. 'A(alpha)': A-stable only within a
    #: cone of half-angle alpha from the negative real axis (BDF3+).
    stability: Optional[str] = None
    #: Convergence order actually reached from a cold start (history length < steps),
    #: or None for a one-step method, where the question does not apply. A k-step
    #: multistep method needs k-1 prior derivatives it does not have yet at t=0;
    #: self-starting with a lower-order method caps the *whole run* at this order
    #: unless a separate high-order starter is used (NOTES.md S3.7 pain point 1).
    startup_order: Optional[int] = None

    @property
    def supports_reuse(self) -> bool:
        """True iff `priorStep` reuse costs this scheme no convergence order."""
        return self.reuse_order is not None and self.reuse_order >= self.order

    @property
    def family(self):
        """The family enum class this scheme belongs to (e.g. `ExplicitRK`, `DIRK`).

        One of the `enums.FAMILY_ENUMS` views of `IntegrationSchemeType` — the
        same fact as `SCHEME_FAMILY[self.identifier]`, on the registered
        scheme object.
        """
        return SCHEME_FAMILY[self.identifier]

    def __call__(self, state, dt, f, *args, **kwargs):
        return self.function(state, dt, f, *args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def split_return(rv):
    if isinstance(rv, tuple):
        return rv[0], rv[1:]
    return rv, None

from torch.profiler import record_function

def initializeSystem(currentSystem, dt, *args, **kwargs):
    # print('Arguments: ', args)
    # print('Kwargs: ', kwargs)
    with record_function("[Integration] Initialize"):
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
        return currentSystem.finalize(initialSystem, dt, *args, **kwargs)


def updateStep(initialState, currentState, dt, f, *args, **kwargs):
    with record_function("[Integration] Update Step"):
        preprocessSystem(currentState, initialState, dt, *args, **kwargs)
        k, r = split_return(f(currentState, dt, *args, **kwargs))
        postprocessSystem(initialState, currentState, dt, r, *args, **kwargs)
        return k, r
    

# from warpSPHIntegrators.integration import *
