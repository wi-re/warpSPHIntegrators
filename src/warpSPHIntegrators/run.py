"""Transactional fixed/adaptive run orchestration over registered single steps.

State-owned buffers are isolated. External effects require explicit transaction
hooks; callbacks/events must otherwise treat inputs as read-only. No solver,
coefficient, or existing single-step acceptance semantics are replaced here.
"""

from dataclasses import dataclass, fields, is_dataclass
import copy
from enum import Enum
import math
from numbers import Real
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .adaptive import propose_dt, dormand_prince_dense_output
from .fields import _maybe_reference_state, integrated_field_names, _find_handler
from .history import StepHistory
from .integration import getIntegrator
from .specs import IntegrationResult
from .util import IntegrationScheme


@dataclass(frozen=True)
class StepStatus:
    successful: bool
    finite: bool
    tolerance_converged: bool | None
    precision_floor: bool
    termination: str


def _tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensor_values(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_values(item)


def _finite_integrated(state):
    reference = _maybe_reference_state(state)
    return all(bool(torch.isfinite(getattr(reference, name)).all())
               for name in integrated_field_names(state))


def assess_step(result):
    finite = (math.isfinite(float(result.state.t)) and _finite_integrated(result.state)
              and (result.error is None or _finite_integrated(result.error))
              and all(bool(torch.isfinite(value).all()) for stage in result.stages
                      for value in _tensor_values(stage.update)))
    diagnostics = result.solver_diagnostics
    if diagnostics is None:
        diagnostics = ()
    elif not isinstance(diagnostics, (list, tuple)):
        diagnostics = (diagnostics,)
    diagnostics = tuple(d for d in diagnostics if d is not None)
    failed = next((d for d in diagnostics if d.converged is False or
                   d.termination in ('max_iterations', 'invalid_residual')), None)
    floor = any(d.termination == 'stagnation' for d in diagnostics)
    met = all(d.termination == 'tolerance' for d in diagnostics) if diagnostics else None
    reason = ('nonfinite' if not finite else failed.termination if failed else
              'precision_floor' if floor else 'tolerance' if met else
              'fixed_iterations' if diagnostics else 'completed')
    return StepStatus(finite and failed is None, finite, met, floor, reason)


def snapshot_state(value, *, detach=False, _memo=None):
    """Clone the state-owned graph, preserving aliases and tensor gradients.

    Dataclasses, ordinary attribute/slot objects, containers and registered buffer
    types are supported. Functions are shared; closed-over or external resources
    require a transaction. Unsupported opaque resources need a custom snapshot.
    """
    memo = {} if _memo is None else _memo
    if value is None or isinstance(value, (str, bytes, bool, Real, complex, Enum, torch.dtype, torch.device)):
        return value
    if callable(value) or isinstance(value, type):
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    clone = lambda x: snapshot_state(x, detach=detach, _memo=memo)
    if isinstance(value, torch.Tensor):
        result = value.detach().clone() if detach else value.clone()
    elif isinstance(value, np.ndarray):
        result = value.copy()
    elif isinstance(value, dict):
        result = type(value)(); memo[key] = result
        result.update((clone(k), clone(v)) for k,v in value.items())
    elif isinstance(value, list):
        result = []; memo[key] = result
        result.extend(clone(v) for v in value)
    elif isinstance(value, tuple):
        items = [clone(v) for v in value]
        result = type(value)(*items) if hasattr(value, '_fields') else tuple(items)
    elif isinstance(value, (set, frozenset)):
        result = type(value)(clone(v) for v in value)
    elif (handler := _find_handler(value)) is not None:
        result = handler.clone(value, detach)
    elif is_dataclass(value) or hasattr(value, '__dict__') or hasattr(type(value), '__slots__'):
        result = copy.copy(value); memo[key] = result
        attributes = dict(vars(value)) if hasattr(value, '__dict__') else {}
        if is_dataclass(value):
            attributes.update((f.name, getattr(value, f.name)) for f in fields(value))
        for owner in type(value).__mro__:
            slots = getattr(owner, '__slots__', ())
            for name in (slots,) if isinstance(slots, str) else slots:
                if name not in ('__dict__', '__weakref__') and hasattr(value, name):
                    attributes[name] = getattr(value, name)
        for name, item in attributes.items():
            object.__setattr__(result, name, clone(item))
    else:
        handler = _find_handler(value)
        if handler is None:
            raise TypeError(f'No transactional snapshot for {type(value).__name__}; pass snapshot=')
        result = handler.clone(value, detach)
    memo[key] = result
    return result


@dataclass(frozen=True)
class RunSample:
    t: float
    state: Any


@dataclass(frozen=True)
class StepRecord:
    dt: float
    result: IntegrationResult
    status: StepStatus
    error_norm: float | None


@dataclass(frozen=True)
class RunCheckpoint:
    scheme: str
    state: Any
    history: StepHistory
    prior_step: Any
    last_dt: float | None
    next_dt: float
    configuration: dict


@dataclass(frozen=True)
class RunResult:
    state: Any
    history: StepHistory
    prior_step: Any
    last_dt: float | None
    next_dt: float
    accepted_steps: int
    rejected_steps: int
    attempts: int
    samples: tuple
    events: tuple
    termination: str
    checkpoint: RunCheckpoint

    @property
    def successful(self):
        return self.termination in ('finished', 'event', 'callback')


class RunError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'Run stopped at t={result.state.t}: {result.termination}')


def _positive(name, value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return float(value)


def _tolerances(value, names, name, default):
    if isinstance(value, Mapping):
        if not set(value) <= set(names):
            raise ValueError(f'Unknown {name} fields: {set(value)-set(names)}')
        return {field: _positive(name, value.get(field, default), zero=True) for field in names}
    return {field: _positive(name, value, zero=True) for field in names}


def _error_norm(result, previous, rtol, atol):
    error = _maybe_reference_state(result.error)
    new = _maybe_reference_state(result.state)
    old = _maybe_reference_state(previous)
    total, count = 0., 0
    for name in integrated_field_names(result.state):
        value = getattr(error, name)
        scale = atol[name] + rtol[name] * torch.maximum(getattr(old, name).abs(), getattr(new, name).abs())
        weighted = torch.where(scale > 0, value/scale, torch.where(value == 0, 0., float('inf')))
        total += float(weighted.square().sum())
        count += weighted.numel()
    return math.sqrt(total/count) if count else 0.


def solve(scheme, state, t_end, f, *, dt=None, adaptive=None, rtol=None, atol=None,
          dt_min=None, dt_max=None, max_steps=100000, max_attempts=200000,
          max_rejections=1000, history=None, reuse=None, allow_stagnation=None,
          sample_times=(), save_steps=False, callback=None, transaction=None,
          snapshot=snapshot_state, restart=None, raise_on_failure=True,
          rhs_args=(), **step_options):
    """Integrate to t_end, committing only accepted state/history/side effects.

    The callback receives an isolated StepRecord after commit; False stops cleanly.
    External transactions implement begin(state), commit(token, result), rollback(token).
    Without those hooks, RHS/lifecycle effects must stay inside the owned state.
    Sampling never changes the numerical grid: off-grid samples require DP5 dense
    output. Events/checkpoints extend this contract through the helpers below.
    """
    scheme = scheme if isinstance(scheme, IntegrationScheme) else getIntegrator(scheme)
    if restart is not None:
        if restart.scheme != scheme.identifier.name:
            raise ValueError('Checkpoint scheme does not match')
        if state is not None:
            raise ValueError('Pass state=None when restarting')
        state = restart.state
        adaptive = restart.configuration['adaptive'] if adaptive is None else adaptive
        if adaptive != restart.configuration['adaptive']:
            raise ValueError('Checkpoint stepping mode does not match')
        dt = restart.next_dt if dt is None else dt
        rtol = restart.configuration['rtol'] if rtol is None else rtol
        atol = restart.configuration['atol'] if atol is None else atol
        history = restart.history if history is None else history
        dt_min = restart.configuration['dt_min'] if dt_min is None else dt_min
        dt_max = restart.configuration['dt_max'] if dt_max is None else dt_max
        reuse = restart.configuration['reuse'] if reuse is None else reuse
        allow_stagnation = restart.configuration['allow_stagnation'] if allow_stagnation is None else allow_stagnation
    adaptive = False if adaptive is None else adaptive
    reuse = True if reuse is None else reuse
    allow_stagnation = False if allow_stagnation is None else allow_stagnation
    for name,value in (('adaptive',adaptive),('reuse',reuse),('allow_stagnation',allow_stagnation),('save_steps',save_steps),('raise_on_failure',raise_on_failure)):
        if type(value) is not bool:
            raise ValueError(f'{name} must be a bool')
    t0 = float(state.t); end = float(t_end)
    if not math.isfinite(t0) or not math.isfinite(end):
        raise ValueError('Run times must be finite')
    direction = 1 if end >= t0 else -1
    step_size = _positive('dt', dt if dt is not None else 1. if t0 == end else float('nan'))
    minimum = _positive('dt_min', 1e-14 if dt_min is None else dt_min)
    maximum = _positive('dt_max', dt_max if dt_max is not None else max(step_size, abs(end-t0)))
    if maximum < minimum:
        raise ValueError('dt_max must be >= dt_min')
    for name,value in (('max_steps',max_steps),('max_attempts',max_attempts),('max_rejections',max_rejections)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if adaptive and (not scheme.has_error_estimate or scheme.metadata.history_length):
        raise ValueError('Adaptive runs require a one-step estimate-emitting scheme')
    if 'priorStep' in step_options:
        raise ValueError('The runner owns priorStep; use a checkpoint to resume reuse')
    current = snapshot(state)
    names = integrated_field_names(current)
    if not names:
        raise ValueError('Runner requires integrated tensor fields')
    rel = _tolerances(1e-3 if rtol is None else rtol, names, 'rtol', 1e-3)
    absolute = _tolerances(1e-6 if atol is None else atol, names, 'atol', 1e-6)
    if any(rel[name] == absolute[name] == 0 for name in names):
        raise ValueError('Each field needs a positive absolute or relative tolerance')
    history = snapshot(history) if history is not None else StepHistory(maxlen=max(1,scheme.metadata.history_length))
    if history.maxlen < scheme.metadata.history_length:
        raise ValueError('History capacity is too small for the chosen method')
    prior = snapshot(restart.prior_step) if restart is not None else None
    last_dt = restart.last_dt if restart is not None else None
    nominal = restart.configuration['nominal_dt'] if restart is not None and not adaptive else step_size
    configuration = dict(adaptive=adaptive, rtol=rel, atol=absolute, nominal_dt=nominal,
                         dt_min=minimum, dt_max=maximum, reuse=reuse, allow_stagnation=allow_stagnation)
    accepted = rejected = attempts = 0
    samples = []; hits = []
    requested = tuple(float(t) for t in sample_times)
    if any(not math.isfinite(t) or direction*(t-t0) < 0 or direction*(end-t) < 0 for t in requested):
        raise ValueError('sample_times must be finite and within the run interval')
    if any(direction*(b-a) <= 0 for a,b in zip(requested,requested[1:])):
        raise ValueError('sample_times must be strictly ordered in the run direction')
    if not scheme.metadata.dense_output:
        for t in requested:
            if t in (t0,end):
                continue
            if adaptive or not math.isclose((t-t0)/ (direction*nominal), round((t-t0)/(direction*nominal)), rel_tol=0, abs_tol=1e-10):
                raise ValueError('Off-grid sampling requires Dormand-Prince dense output')
    if transaction is not None and not all(callable(getattr(transaction, key, None)) for key in ('begin','commit','rollback')):
        raise TypeError('transaction needs begin, commit and rollback hooks')
    def finish(reason):
        saved = snapshot(current)
        cp = RunCheckpoint(scheme.identifier.name, snapshot(current), snapshot(history),
                           snapshot(prior), last_dt, step_size, snapshot(configuration))
        result = RunResult(saved, snapshot(history), snapshot(prior), last_dt, step_size,
                           accepted, rejected, attempts, tuple(samples), tuple(hits), reason, cp)
        if not result.successful and raise_on_failure:
            raise RunError(result)
        return result
    def sample(t, value):
        if not samples or samples[-1].t != t:
            samples.append(RunSample(t, snapshot(value)))
    if not _finite_integrated(current):
        return finish('nonfinite_input')
    sample(t0,current)
    index = int(bool(requested) and requested[0] == t0)
    while direction*(end-float(current.t)) > 0:
        if accepted >= max_steps:
            return finish('max_steps')
        if attempts >= max_attempts:
            return finish('max_attempts')
        if rejected >= max_rejections:
            return finish('max_rejections')
        remaining = abs(end-float(current.t))
        h = min(step_size, maximum, remaining)
        if math.isclose(h,remaining,rel_tol=0,abs_tol=4*math.ulp(end)):
            h = remaining
        if (h < minimum and remaining > minimum) or float(current.t)+direction*h == float(current.t):
            return finish('min_dt')
        signed_h = direction*h
        trial = snapshot(current)
        trial_history = snapshot(history)
        same_dt = last_dt is not None and math.isclose(last_dt,signed_h,rel_tol=1e-12,abs_tol=0)
        if trial_history.latest is not None and not math.isclose(trial_history.latest.dt,signed_h,rel_tol=1e-12,abs_tol=0):
            trial_history = StepHistory(history.maxlen)  # reset BEFORE the formula consumes history
        options = dict(step_options, history=trial_history)
        if reuse and scheme.fsal and same_dt and prior is not None:
            options['priorStep'] = snapshot(prior)
        token = transaction.begin(snapshot(current)) if transaction is not None else None
        active = transaction is not None
        attempts += 1
        try:
            result = scheme(trial, signed_h, f, *rhs_args, **options)
            status = assess_step(result)
            if status.finite and not math.isclose(float(result.state.t), float(current.t)+signed_h,rel_tol=0,abs_tol=4*math.ulp(float(current.t)+signed_h)):
                raise ValueError('Step returned an incorrect final time')
            error = (_error_norm(result,current,rel,absolute) if adaptive and status.successful else None)
            acceptable = status.successful and (allow_stagnation or not status.precision_floor)
            acceptable = acceptable and (not adaptive or (math.isfinite(error if error is not None else float('inf')) and error <= 1))
            if not acceptable:
                if transaction is not None:
                    transaction.rollback(token)
                    active = False
                rejected += 1
                if not adaptive:
                    return finish(status.termination if status.termination != 'completed' else 'step_failure')
                new_size = propose_dt(error,h,scheme.estimate_order) if error is not None and math.isfinite(error) else h*.2
                step_size = min(h*.9,new_size)
                continue
            if transaction is not None:
                transaction.commit(token,snapshot(result))
                active = False
        except BaseException:
            if active:
                transaction.rollback(token)
            raise
        previous = current
        current = result.state
        if h == remaining:
            current.t = end  # avoid accumulating representation drift at the requested endpoint
        history = result.history if result.history is not None else trial_history
        prior = result.stages[-1] if result.stages else None
        last_dt = signed_h
        accepted += 1
        while index < len(requested) and (direction*(requested[index]-float(current.t)) <= 0 or
                                        math.isclose(requested[index],float(current.t),rel_tol=0,abs_tol=4*math.ulp(float(current.t)))):
            t = requested[index]
            value = snapshot(current) if math.isclose(t,float(current.t),rel_tol=0,abs_tol=4*math.ulp(float(current.t))) else dormand_prince_dense_output(
                snapshot(previous), result.stages, signed_h, (t-float(previous.t))/signed_h,
                *rhs_args, **options)
            value.t = t
            sample(t,value); index += 1
        if save_steps:
            sample(float(current.t),current)
        step_size = min(maximum,propose_dt(error,h,scheme.estimate_order)) if adaptive else nominal
        if callback is not None and callback(snapshot(StepRecord(signed_h,result,status,error))) is False:
            sample(float(current.t),current)
            return finish('callback')
    sample(end,current)
    return finish('finished')

