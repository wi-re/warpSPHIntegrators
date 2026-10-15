# Runs, acceptance and restart

`solve` owns a fixed or adaptive trajectory over the existing registered single-step
drivers. It works with `TensorState` and the tagged integration-system protocol.
Existing `scheme(state, dt, rhs, ...)` calls retain their single-step behavior.

This complete float64 CPU example samples an adaptive decay trajectory and restarts
it with a terminal event. It is also saved as `examples/run_ode.py`.

[example:include examples/run_ode.py]: #
```python
"""Controlled tensor ODE run: python examples/run_ode.py."""

import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import Event


def run_example():
    initial = TensorState(torch.ones(2, dtype=torch.float64))
    rhs = tensor_rhs(lambda t, y: -y)
    result = solve('Dormand-Prince 5(4)', initial, 1., rhs, dt=.3,
                   adaptive=True, rtol=1e-8, atol=1e-10,
                   sample_times=[0., .25, .5, .75, 1.])
    assert result.state.t == 1.
    assert torch.allclose(result.state.y, initial.y * math.exp(-1), rtol=3e-8)
    assert torch.equal(initial.y, torch.ones_like(initial.y))
    resumed = solve('Dormand-Prince 5(4)', None, 2., rhs,
                    restart=result.checkpoint,
                    events=[Event(lambda t, s: s.y[0].item() - .2,
                                  name='threshold', terminal=True)])
    assert resumed.termination == 'event'
    assert abs(resumed.state.t - math.log(5)) < 1e-7
    return result, resumed


if __name__ == '__main__':
    result, resumed = run_example()
    print(f'accepted={result.accepted_steps}, rejected={result.rejected_steps}, '
          f'event={resumed.events[-1].name} at t={resumed.state.t:.8f}')
```

## Fixed and adaptive steps

Pass a scheme record, name or enum, a state, `t_end`, RHS and positive `dt`.
`adaptive=False` is the default. Time direction comes from the endpoints, so
backward runs also use positive `dt`. The last step is clipped to end exactly at
`t_end`; a zero-length interval returns the owned initial state. Fixed multistep
history is threaded automatically and cleared **before** consuming it when the
step changes, including a clipped endpoint. That endpoint uses the method's
startup and can reduce its accuracy. Reuse uses FSAL only at unchanged steps;
`reuse=False` disables it. Callbacks that alter external RHS parameters also need
reuse disabled.

Variable-coefficient `VariableAdams` retains compatible history across step
changes, including adaptive steps; large ratio changes can restart it safely.

Adaptive mode requires an estimate emitter with one-step or variable-step
history support. Scalar `rtol` / `atol` or
field-name maps control a weighted RMS norm over all integrated entries:
`error / (atol + rtol * max(abs(old), abs(new)))`. Missing map fields retain
`rtol=1e-3`, `atol=1e-6`; unknown fields, nonfinite or negative tolerances, and
fields with both tolerances zero are rejected. Accept when that norm is at most
one and the step status permits acceptance. The controller uses a per-result
`estimate_order` during ramped startup, otherwise the scheme's steady metadata,
including exponent three for TR-BDF2 and Gauss-Legendre 2.
Rejected attempts shrink the step. Nonfinite states, derivatives or estimates
are rejected. Constant-coefficient multistep methods remain fixed-step.

`dt_min=1e-14`, `dt_max` (initially the larger of `dt` and interval length),
`max_steps=100000`, `max_attempts=200000` and `max_rejections=1000` bound runs.
A shorter final endpoint step is permitted. Limits count the current invocation,
including retries for terminal events. A numerical failure raises `RunError` with
its partial `result`; `raise_on_failure=False` returns that result instead.
RHS, option, hook and callback exceptions propagate as caller errors. The
`termination` identifies the stop, and `successful` includes normal completion,
a terminal event and a callback-requested stop.

## Step and solver status

`IntegrationResult.step_status` (also `assess_step(result)`) exposes `successful`,
`finite`, `tolerance_converged`, `precision_floor` and `termination`.
Native implicit drivers retain the solver's `converged` flag in diagnostics,
including custom solvers that return no detailed diagnostics. A false flag,
`max_iterations` or `invalid_residual` fails the step. Fixed-iteration Picard and
linear/exponential methods report their scheduled completion; this is not a
claim that a nonlinear tolerance was met. Explicit steps have no nonlinear
convergence claim (`tolerance_converged=None`).

Newton `stagnation` remains distinct from `tolerance`: the solver may declare a
precision-floor result usable, so step status can be successful while
`tolerance_converged=False`. The runner rejects it by default. Fixed runs stop;
adaptive runs retry within their limits. Set `allow_stagnation=True` deliberately
to accept those finite, solver-approved results. Direct single-step calls still
return their candidate even when the solve failed: check their status explicitly.

## Ownership, lifecycle and callbacks

Each attempt receives an isolated snapshot of the last accepted state and history.
Rejected candidates, copied/ephemeral buffers and lifecycle changes are discarded.
The input state, history, returned samples, checkpoint and callback records own
separate copies. Tensor clones retain autograd connections. `callback(record)`
runs after acceptance, with `record.dt`, `result`, `status` and `error_norm`;
return exactly `False` to stop cleanly. `save_steps=True` stores accepted endpoints.
The default stores initial/final states plus requested samples, not every stage
or an unbounded list of full step results.

`snapshot_state` handles tensor/NumPy buffers, containers, dataclasses, ordinary
attribute/slot objects and registered clone handlers, preserving repeated aliases.
Pass `snapshot=your_snapshot` for application-owned resource types; it must isolate
all mutable state and preserve the protocol. Functions/callables are shared.
External closure variables, file writes and device services cannot be undone by
cloning state. Keep RHS/lifecycle effects state-owned or supply a transaction with
`begin(state) -> token`, `commit(token, result)` and `rollback(token)`. Commit
receives an isolated result; rollback runs once on rejection or an exception
before commit. Hooks must themselves implement reliable rollback/commit. After
commit, output/callback exceptions cannot undo that accepted step's external effects.
Event functions and interpolation update hooks must treat external resources as
read-only; dense output is a numerical query, not a new physical lifecycle step.

## Samples and events

`sample_times` must be finite, unique, ordered in the run direction and inside the
interval. Sampling never changes the step grid or controller. Methods without
dense output allow only fixed-grid endpoints (and the final clipped endpoint).
Off-grid samples and event localization currently require Dormand-Prince 5(4).
Its quartic interpolates integrated fields; auxiliary/copied fields are not
recomputed or finalized for a sample. Query samples as numerical output, not as
restart states with refreshed physical buffers.

`Event(function, name='event', terminal=False, direction=0, time_tol=1e-10)`
evaluates `function(t, isolated_state)` as a finite scalar. Continuous sign changes
are localized by bisection of DP5 dense output. Direction `+1` / `-1` refers to
increasing physical time, including backward integration. Initial zeros are
reported once, and endpoint zeros are not repeated on the next step. Tangencies
and multiple crossings within one step can be missed; decrease `dt_max` when
needed. Event functions must be pure. Root tolerance is a time tolerance on the
interpolant, not a guarantee of exact physical event time.

A terminal crossing rolls back the full trial and recomputes a step to the root,
so accepted history and lifecycle effects end at the returned state. Localization
may need further retries because the shortened step has its own numerical root.
Nonterminal hits are output records and do not split the integration grid.

## Restart checkpoints

History methods may expose `next_step_size(result, h, proposal)` to constrain
the runner's accepted adaptive proposal. The result owns trial memory; the hook
returns a positive finite magnitude before the external transaction commits.
NDF uses this to hold quasi-constant blocks. Rejection still shrinks immediately.

An order-selecting driver can instead expose
`adapt_step(result, h, proposal, *, error_norm, norm_estimate)`. The scalar
`error_norm` and `norm_estimate(error_state)` use the same old/new field scales
as acceptance. AdaptiveNDF uses these to compare neighboring defects and update
owned trial history before commit. This hook may not mutate external state.
Its returned magnitude is validated and capped by `dt_max` and method ratios.

Portable history entries may have `update=None, aux=None` when the method stores
its state entirely in `history.context`. These serialize with a null stage;
older checkpoint entries with tensor updates continue to load. NDF's difference
tensors and accepted-step counter are included in this weights-only payload.

Every result contains an isolated `RunCheckpoint`: accepted state, history, FSAL
stage, last/next step and controller configuration. Resume with `state=None` and
`restart=checkpoint`; the scheme and stepping mode must match. Tolerances, bounds,
reuse and stagnation policy carry forward unless explicitly overridden. A fixed
run keeps its original nominal step. For exact trajectory continuation, stop via
callback on an existing accepted endpoint; choosing a new intermediate `t_end`
can clip a step and restart multistep history.

The caller supplies the same RHS, parameters, `rhs_args`, driver/solver options,
transaction and callbacks again. Event definitions, previous hits and sampling
requests are invocation-local; zeros at a resumed initial endpoint are suppressed.
Application state with external context requires matching application-owned
restart data. In-memory checkpoints retain gradients; use
`snapshot_state(checkpoint, detach=True)` to truncate the graph explicitly.

`save_checkpoint(checkpoint, path)` and `load_checkpoint(path, map_location=...)`
provide a versioned tensor-only file format. They support plain `TensorState`,
tensor updates, histories and empty auxiliary tuples/lists. Save detaches tensors;
load uses PyTorch's weights-only deserializer. Files do not serialize functions,
solvers, custom systems or external resources. Full SPH systems use in-memory
checkpoints or application-specific serialization. A checkpoint at a terminal
event can be resumed; decide which events to install for the resumed invocation.
