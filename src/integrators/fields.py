
from dataclasses import MISSING, dataclass, field as dc_field
import dataclasses
from typing import Any
import torch

BEHAVIOR_KEY = 'behavior'
ROLE_KEY = 'role'
TAGS_KEY = 'tags'


def _normalize_tags(tags=None):
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,)
    return tuple(dict.fromkeys(tags))


def _field_metadata(*, behavior=None, role=None, tags=None, metadata=None, **extra):
    merged = dict(metadata or {})
    if behavior is not None:
        merged[BEHAVIOR_KEY] = behavior
    if role is not None:
        merged[ROLE_KEY] = role

    existing_tags = _normalize_tags(merged.get(TAGS_KEY))
    merged[TAGS_KEY] = tuple(dict.fromkeys((*existing_tags, *_normalize_tags(tags))))
    merged.update(extra)
    return merged


def tagged(*, default=MISSING, default_factory=MISSING, role=None, tags=None, metadata=None, **extra):
    kwargs = {'metadata': _field_metadata(role=role, tags=tags, metadata=metadata, **extra)}
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def integrated(
    update_key: str,
    fluid_only: bool = True,
    default=MISSING,
    *,
    default_factory=MISSING,
    role=None,
    tags=None,
    metadata=None,
):
    """Updated via integrateQ. Cloned in initializeNewState."""
    kwargs = {
        'metadata': _field_metadata(
            behavior='integrated',
            role=role,
            tags=tags,
            metadata=metadata,
            update_key=update_key,
            fluid_only=fluid_only,
        )
    }
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def constant(default=MISSING, *, default_factory=MISSING, role=None, tags=None, metadata=None):
    """Never touched. Cloned in initializeNewState."""
    kwargs = {
        'metadata': _field_metadata(
            behavior='constant',
            role=role,
            tags=tags,
            metadata=metadata,
        )
    }
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def copied(default=MISSING, *, default_factory=MISSING, role=None, tags=None, metadata=None):
    """Not integrated; copied from last substep in finalize. None in initializeNewState."""
    kwargs = {
        'metadata': _field_metadata(
            behavior='copied',
            role=role,
            tags=tags,
            metadata=metadata,
        )
    }
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def ephemeral(default=MISSING, *, default_factory=MISSING, role=None, tags=None, metadata=None):
    """Stage-local only. None everywhere."""
    kwargs = {
        'metadata': _field_metadata(
            behavior='ephemeral',
            role=role,
            tags=tags,
            metadata=metadata,
        )
    }
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def custom(default=MISSING, *, default_factory=MISSING, role=None, tags=None, metadata=None):
    """No default behavior. System overrides finalize/initializeNewState explicitly."""
    kwargs = {
        'metadata': _field_metadata(
            behavior='custom',
            role=role,
            tags=tags,
            metadata=metadata,
        )
    }
    if default is not MISSING:
        kwargs['default'] = default
    if default_factory is not MISSING:
        kwargs['default_factory'] = default_factory
    return dc_field(**kwargs)


def reference_state(*, default=MISSING, default_factory=MISSING, tags=None, metadata=None):
    """Marks the system field that holds the primary state for generic helpers."""
    return tagged(
        default=default,
        default_factory=default_factory,
        role='reference_state',
        tags=('reference_state', *_normalize_tags(tags)),
        metadata=metadata,
    )


def field_has_tag(field, tag: str) -> bool:
    return tag in _normalize_tags(field.metadata.get(TAGS_KEY))


def find_tagged_field(instance_or_type, *, role=None, tag=None):
    owner = instance_or_type if isinstance(instance_or_type, type) else type(instance_or_type)
    matches = []
    for f in dataclasses.fields(owner):
        if role is not None and f.metadata.get(ROLE_KEY) != role:
            continue
        if tag is not None and not field_has_tag(f, tag):
            continue
        matches.append(f)

    if not matches:
        raise LookupError(f'No dataclass field found for role={role!r}, tag={tag!r}')
    if len(matches) > 1:
        names = ', '.join(f.name for f in matches)
        raise LookupError(f'Ambiguous dataclass fields for role={role!r}, tag={tag!r}: {names}')
    return matches[0]


def get_tagged_attr(obj: Any, *, role=None, tag=None):
    field = find_tagged_field(obj, role=role, tag=tag)
    return getattr(obj, field.name)


def set_tagged_attr(obj: Any, value: Any, *, role=None, tag=None):
    field = find_tagged_field(obj, role=role, tag=tag)
    setattr(obj, field.name, value)
    return obj


def get_reference_state(system):
    return get_tagged_attr(system, role='reference_state')


def _op(value, detach=False):
    return value.detach().clone() if detach else value.clone() if isinstance(value, torch.Tensor) else value


def _empty_like(value):
    return None if isinstance(value, (torch.Tensor, type(None))) else value


def _clone_state(state, *, include_ephemeral=False, detach=False):
    """Generic clone driven by behavior metadata. No per-field manual code."""
    kwargs = {}
    for f in dataclasses.fields(state):
        behavior = f.metadata.get(BEHAVIOR_KEY, 'constant')
        value = getattr(state, f.name)
        if behavior == 'ephemeral' and not include_ephemeral:
            kwargs[f.name] = _empty_like(value)
        else:
            kwargs[f.name] = _op(value, detach)
    return type(state)(**kwargs)


def _state_initialize(state, detach=False):
    kwargs = {}
    for f in dataclasses.fields(state):
        behavior = f.metadata.get(BEHAVIOR_KEY, 'copied')
        value = getattr(state, f.name)
        if behavior in ('constant', 'integrated'):
            kwargs[f.name] = _op(value, detach)
        else:
            kwargs[f.name] = _empty_like(value)
    return type(state)(**kwargs)


def _state_finalize(live_state, last_substep_state):
    """Copy 'copied' fields from last substep. Leave everything else alone."""
    for f in dataclasses.fields(live_state):
        if f.metadata.get(BEHAVIOR_KEY, 'copied') == 'copied':
            setattr(live_state, f.name, getattr(last_substep_state, f.name))


@dataclass
class BaseState:
    def initializeNewState(self, **kwargs):
        return _state_initialize(self)

    def clone(self):
        return _clone_state(self, include_ephemeral=True)

    def nograd(self):
        return _clone_state(self, detach=True)

    def to(self, device):
        kwargs = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                kwargs[f.name] = value.to(device)
            else:
                kwargs[f.name] = value
        return type(self)(**kwargs)
    

from .specs import ComponentUpdateSpec, PositionUpdateSpec, StateBlend
import numpy as np
from dataclasses import dataclass
def verbosePrint(verbose, *args):
    if verbose:
        print(*args)
def _resolve_state(system, system_role='reference_state'):
    return get_tagged_attr(system, role=system_role)
def _resolve_delta(update, derivative_tag):
    if isinstance(update, list):
        return [get_tagged_attr(item, tag=derivative_tag) for item in update]
    return get_tagged_attr(update, tag=derivative_tag)
def _accumulate(base_value, delta, step):
    if isinstance(step, list):
        result = base_value
        for step_i, delta_i in zip(step, delta):
            result = result + step_i * delta_i
        return result
    return base_value + step * delta
def update_component(system, update, spec: ComponentUpdateSpec, state_tag: str, derivative_tag: str, system_role='reference_state'):
    if isinstance(update, list) and len(update) == 0:
        return system
    state = _resolve_state(system, system_role)
    blend = spec.blend
    value = get_tagged_attr(state, tag=state_tag)
    if blend.self_scale is not None:
        value = value * blend.self_scale
    if blend.reference_state is not None:
        reference_state = _resolve_state(blend.reference_state, system_role)
        value = value + blend.reference_weight * get_tagged_attr(reference_state, tag=state_tag)
    delta = _resolve_delta(update, derivative_tag)
    value = _accumulate(value, delta, spec.derivative_dt)
    set_tagged_attr(state, value, tag=state_tag)
    return system

def update_position(
    system,
    update,
    spec: PositionUpdateSpec,
    position_tag: str,
    position_derivative_tag: str,
    velocity_tag: str,
    velocity_derivative_tag: str = None,
    system_role='reference_state',
):
    has_derivative = update is not None and (not isinstance(update, list) or len(update) > 0)
    state = _resolve_state(system, system_role)
    blend = spec.blend
    value = get_tagged_attr(state, tag=position_tag)
    if blend.self_scale is not None:
        value = value * blend.self_scale
    if blend.reference_state is not None:
        reference_state = _resolve_state(blend.reference_state, system_role)
        value = value + blend.reference_weight * get_tagged_attr(reference_state, tag=position_tag)
    if has_derivative and spec.derivative_dt:
        delta = _resolve_delta(update, position_derivative_tag)
        value = _accumulate(value, delta, spec.derivative_dt)
    if spec.current_velocity_dt is not None:
        value = value + spec.current_velocity_dt * get_tagged_attr(state, tag=velocity_tag)
    if spec.update_velocity_dt is not None and has_derivative and velocity_derivative_tag is not None:
        value = value + spec.update_velocity_dt * _resolve_delta(update, velocity_derivative_tag)
    set_tagged_attr(state, value, tag=position_tag)
    return system

