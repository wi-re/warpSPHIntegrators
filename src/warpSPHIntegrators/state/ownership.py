from dataclasses import dataclass, fields, is_dataclass

import copy

from enum import Enum

import math

from numbers import Real


import numpy as np

import torch


from warpSPHIntegrators.state.fields import _maybe_reference_state, integrated_field_names, _find_handler


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
