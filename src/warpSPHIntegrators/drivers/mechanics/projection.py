"""Temporal primitives for pressure-projected particle maps.

Spatial callbacks supply accelerations at fixed geometry. These recurrences
preserve their historical arithmetic order; masking a kick is an explicit
choice because some projection operators already zero constrained rows.
No SPH package or pressure normalization is assumed here.
"""

import torch

__all__ = ["kick", "drift", "symplecticProjection"]


def kick(velocities, dt, acceleration, *, mask=None):
    if mask is not None:
        acceleration = torch.where(mask, acceleration, torch.zeros_like(acceleration))
    return velocities + dt * acceleration


def drift(positions, dt, velocities, *, mask=None):
    if mask is not None:
        velocities = torch.where(mask, velocities, torch.zeros_like(velocities))
    return positions + dt * velocities


def symplecticProjection(state, dt, acceleration, *, mask=None, damping=0.0, beforeDrift=None):
    """One kick followed by drift; optional shift is evaluated after the kick.

    `beforeDrift(state)` returns an optional displacement computed using the
    kicked velocity at pre-drift geometry. The displacement is added after
    the ordinary drift, preserving the former omni/Fickian update order.
    """
    state.velocities = kick(state.velocities, dt, acceleration, mask=mask)
    if damping != 0.0:
        state.velocities = state.velocities * (1.0 - damping)
    shift = None if beforeDrift is None else beforeDrift(state)
    state.positions = drift(state.positions, dt, state.velocities, mask=mask)
    if shift is not None:
        state.positions = state.positions + shift
    return state


def step(method, state, dt, rhs, *args, **kwargs):
    """One full dt map; never repeat/blend pressure-projected full steps."""
    from warpSPHIntegrators.state.lifecycle import (
        initializeSystem,
        updateStep,
        finalizeSystem,
        hostTime,
        reject_prior_step,
    )
    from warpSPHIntegrators.state.specs import IntegrationResult, StageResult

    reject_prior_step(method.name, kwargs.pop("priorStep", None))
    if not getattr(state, "projected_map", False):
        raise TypeError(
            "ProjectedStep requires a projected_map system with stage-state finalization"
        )
    initializeSystem(state, dt, *args, **kwargs)
    stage = state.initializeNewState(*args, **kwargs)
    stage.t = hostTime(state.t)
    stage._projectedStepActive = True
    update, aux = updateStep(state, stage, dt, rhs, *args, **kwargs)
    output = state.initializeNewState(*args, **kwargs)
    output.t = hostTime(state.t + dt)
    finalizeSystem(
        output, state, dt, [aux], [update], [1.0], *args, lastStageSystem=stage, **kwargs
    )
    return IntegrationResult(output, stages=[StageResult(aux=aux, update=update)])
