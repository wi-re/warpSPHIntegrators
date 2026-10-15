"""Numerical stepping for generalized_alpha; configuration and coefficients are supplied separately."""
from dataclasses import replace

import math


import torch


from warpSPHIntegrators.state.fields import BlockState, find_tagged_field, get_reference_state, flatten_integrated, unflatten_integrated, state_difference, state_norm

from warpSPHIntegrators.state.history import HistoryEntry, StepHistory

from warpSPHIntegrators.solvers.krylov import JFNKSolver

from warpSPHIntegrators.drivers.mechanics.newmark import _acceleration_of

from warpSPHIntegrators.solvers.nonlinear import record_solve_diagnostics

from warpSPHIntegrators.state.specs import IntegrationResult, StageResult, explicit_step

from warpSPHIntegrators.adapters.tensor import TensorState

from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, applyQuantityUpdate, reject_prior_step, verbosePrint


from warpSPHIntegrators.drivers.mechanics._generalized import _layout
from warpSPHIntegrators.schemes.families.generalized_alpha import GeneralizedAlphaMemory

def step(self, state, dt, f, *args, solver=None, history=None, **kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt, bool) or not math.isfinite(dt) or dt == 0:
        raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name, kwargs.pop('priorStep', None))
    history = StepHistory(1) if history is None else history
    if not isinstance(history, StepHistory):
        raise TypeError('history must be a StepHistory')
    base = snapshot_state(state)
    ref = get_reference_state(base)
    position = find_tagged_field(ref, tag='position').name
    velocity = find_tagged_field(ref, tag='velocity').name
    q0, v0 = getattr(ref, position), getattr(ref, velocity)
    if (not isinstance(q0, torch.Tensor) or not isinstance(v0, torch.Tensor)
            or not q0.is_floating_point() or q0.shape != v0.shape
            or q0.dtype != v0.dtype or q0.device != v0.device):
        raise ValueError('Matching floating-point position and velocity tensors required')
    layout = _layout(base)
    uid = getattr(ref, 'uid', None)
    memory = history.latest.state if history.latest is not None else None
    if history.latest is not None:
        if (not isinstance(memory, GeneralizedAlphaMemory) or memory.rho_inf != self.rho_inf
                or memory.layout != layout or not math.isclose(memory.t, float(base.t), rel_tol=0, abs_tol=1e-12)
                or (memory.uid is None) != (uid is None)
                or uid is not None and not torch.equal(memory.uid, uid)):
            raise ValueError('Generalized-alpha history does not match state, parameters or particle layout')
        if (memory.acceleration.shape != v0.shape or memory.acceleration.dtype != v0.dtype
                or memory.acceleration.device != v0.device or not torch.isfinite(memory.acceleration).all()):
            raise ValueError('Invalid algorithmic acceleration memory')
    before_initialize = flatten_integrated(base).clone() if memory is not None else None
    initializeSystem(base, dt, *args, **kwargs)
    if memory is not None and not torch.equal(flatten_integrated(base), before_initialize):
        raise ValueError('Generalized-alpha initialize must preserve integrated fields when resuming memory')
    ref = get_reference_state(base)
    q0, v0 = getattr(ref, position), getattr(ref, velocity)
    layout = _layout(base)
    uid = getattr(ref, 'uid', None)
    if memory is not None and ((memory.uid is None) != (uid is None)
            or uid is not None and not torch.equal(memory.uid, uid)):
        raise ValueError('Generalized-alpha initialize changed the particle layout')
    start = base.initializeNewState(*args, **kwargs)
    start.t = float(base.t)
    initial_update, initial_aux = updateStep(base, start, dt, f, *args, **kwargs)
    drift = getattr(initial_update, find_tagged_field(ref, tag='position').metadata['update_key'])
    if not torch.allclose(drift, v0, rtol=1e-10, atol=1e-12):
        raise ValueError('Generalized-alpha requires position rate equal to velocity')
    physical = _acceleration_of(initial_update)
    if (not isinstance(physical, torch.Tensor) or physical.shape != v0.shape
            or physical.dtype != v0.dtype or physical.device != v0.device):
        raise ValueError('Acceleration must match the velocity tensor layout')
    a0 = physical if memory is None else memory.acceleration
    am, af, gamma, beta = self.parameters
    end_time = float(base.t+dt)

    def step(block):
        guess, acc = block.states
        guess.t = end_time
        endpoint_update, _ = updateStep(base, guess, dt, f, *args, **kwargs)
        weighted = unflatten_integrated(af*flatten_integrated(base)+(1-af)*flatten_integrated(guess), guess)
        weighted.t = float(base.t+(1-af)*dt)
        force_update, _ = updateStep(base, weighted, dt, f, *args, **kwargs)
        a1 = (_acceleration_of(force_update)-am*a0)/(1-am)
        output = base.initializeNewState(*args, **kwargs)
        out = get_reference_state(output)
        setattr(out, position, q0+dt*v0+dt*dt*((.5-beta)*a0+beta*acc.y))
        setattr(out, velocity, v0+dt*((1-gamma)*a0+gamma*acc.y))
        applyQuantityUpdate(output, initial_update, explicit_step(.5*dt), **kwargs)
        applyQuantityUpdate(output, endpoint_update, explicit_step(.5*dt), **kwargs)
        output.t = end_time
        return BlockState((output, TensorState(a1)))

    def norm(new, old):
        return state_norm(state_difference(new, old), 1e-3, 1e-6, reference=old)

    guess = base.initializeNewState(*args, **kwargs)
    guessed_ref = get_reference_state(guess)
    setattr(guessed_ref, position, q0+dt*v0+.5*dt*dt*a0)
    setattr(guessed_ref, velocity, v0+dt*a0)
    guess.t = end_time
    verbosePrint(kwargs.get('verbose', False), '[Integrator] Generalized-alpha: solving weighted equilibrium')
    result = (solver or JFNKSolver()).solve(step, BlockState((guess, TensorState(a0))), norm,
                                          **kwargs.get('solver_opts', {}))
    output, acceleration = result.y.states
    output.t = end_time
    diagnostics = record_solve_diagnostics(result)
    diagnostics = replace(diagnostics, rhs_evaluations=2*diagnostics.rhs_evaluations+1)
    if not torch.isfinite(acceleration.y).all():
        diagnostics = replace(diagnostics, converged=False, termination='invalid_residual')
    candidate = IntegrationResult(output, history=history, solver_diagnostics=diagnostics)
    if not assess_step(candidate).successful:
        # A failed trial has neither an accepted memory value nor a finalize.
        return candidate
    endpoint = output.initializeNewState(*args, **kwargs)
    endpoint.t = end_time
    update, aux = updateStep(base, endpoint, dt, f, *args, **kwargs)
    diagnostics = replace(diagnostics, rhs_evaluations=diagnostics.rhs_evaluations+1)
    stages = [StageResult(initial_aux, initial_update), StageResult(aux, update)]
    candidate = IntegrationResult(output, stages=stages, history=history,
                                  solver_diagnostics=diagnostics)
    if not assess_step(candidate).successful:
        return candidate
    before = flatten_integrated(output).clone()
    # The endpoint trapezoid supplies the quantity-update weights. Mechanics
    # instead follows the weighted acceleration equation above.
    finalizeSystem(output, base, dt, [initial_aux, aux], [initial_update, update], [.5, .5],
                   lastStageSystem=endpoint, **kwargs)
    if not torch.equal(flatten_integrated(output), before):
        raise ValueError('Generalized-alpha finalize must preserve integrated fields')
    accepted = GeneralizedAlphaMemory(acceleration.y.clone(), end_time, self.rho_inf, layout,
                                     snapshot_state(uid))
    history = history.pushed(HistoryEntry(float(base.t), dt, update, aux, accepted))
    return IntegrationResult(output, stages=stages, history=history,
                             solver_diagnostics=diagnostics)
