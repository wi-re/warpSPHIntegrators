
from dataclasses import MISSING, dataclass, field as dc_field
import dataclasses
import math
from typing import Any, Callable, List, NamedTuple, Optional
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


#: Behavior assumed for a field that carries no ``behavior`` metadata. ``constant``
#: is the conservative choice: an untagged field survives both ``clone()`` and
#: ``initializeNewState()`` rather than being silently nulled by one of them.
DEFAULT_BEHAVIOR = 'constant'


def field_behavior(f) -> str:
    """The behavior of a dataclass field. Single source of truth for the default."""
    return f.metadata.get(BEHAVIOR_KEY, DEFAULT_BEHAVIOR)


# --------------------------------------------------------------------------- #
# Value dispatch: how a single field value is cloned, emptied, and moved       #
# --------------------------------------------------------------------------- #
#
# These used to be two functions that special-cased `torch.Tensor` and passed every
# other type through *by reference*. That meant a state holding `wp.array` fields --
# or a plain list or dict of tensors -- was not cloned at all: every RK stage wrote
# into the same buffer as the initial state, stage 2 read stage-1-corrupted data, and
# nothing raised anywhere (NOTES.md 4.1).
#
# Adding a backend is now one `register_clone_handler` call rather than an edit to two
# functions. Handlers are tried newest-first, so a registration overrides a built-in.


class CloneHandler(NamedTuple):
    """How to copy one kind of value."""

    name: str
    #: Does this handler apply to `value`?
    matches: Callable[[Any], bool]
    #: (value, detach) -> an independent copy.
    clone: Callable[[Any, bool], Any]
    #: (value, device) -> the value on `device`. None means "leave it alone".
    to_device: Optional[Callable[[Any, Any], Any]] = None
    #: True if a stage-local (`ephemeral`) field of this kind should be nulled rather
    #: than carried over. Buffers yes; plain values no.
    is_buffer: bool = True


_CLONE_HANDLERS: List[CloneHandler] = []

#: Types that are immutable, so sharing one is safe and cloning it is pointless.
IMMUTABLE_TYPES = (type(None), bool, int, float, complex, str, bytes, frozenset)

#: Types seen with no handler, so the "silently aliased" warning fires once each.
_warned_unhandled = set()


def register_clone_handler(name, matches, clone, to_device=None, is_buffer=True):
    """Teach the state machinery how to copy a new kind of field value.

    Registered last wins, so this overrides the built-ins for an overlapping type.

        register_clone_handler(
            'mylib.Buffer',
            matches=lambda v: isinstance(v, mylib.Buffer),
            clone=lambda v, detach: v.copy(),
            to_device=lambda v, device: v.to(device),
        )
    """
    handler = CloneHandler(name, matches, clone, to_device, is_buffer)
    _CLONE_HANDLERS.append(handler)
    return handler


def _find_handler(value) -> Optional[CloneHandler]:
    for handler in reversed(_CLONE_HANDLERS):
        if handler.matches(value):
            return handler
    return None


def clone_value(value, detach=False):
    """An independent copy of one field value, dispatched on its type.

    Unknown types are passed through by reference and warned about once, because that
    is precisely the case that produces a silently wrong answer.
    """
    if isinstance(value, IMMUTABLE_TYPES):
        return value
    handler = _find_handler(value)
    if handler is not None:
        return handler.clone(value, detach)
    _warn_unhandled(value)
    return value


def empty_value(value):
    """The 'no value here' placeholder for a stage-local or not-yet-computed field.

    Buffers become None. Plain values are left alone: nulling a float parameter that
    happens to sit on an `ephemeral` field would break systems that read it back.
    """
    if value is None or isinstance(value, IMMUTABLE_TYPES):
        return None if value is None else value
    handler = _find_handler(value)
    if handler is not None:
        return None if handler.is_buffer else value
    _warn_unhandled(value)
    return value


def move_value(value, device):
    """`value` on `device`, for types that have a notion of one."""
    if isinstance(value, IMMUTABLE_TYPES):
        return value
    handler = _find_handler(value)
    if handler is not None and handler.to_device is not None:
        return handler.to_device(value, device)
    return value


def _warn_unhandled(value):
    kind = type(value)
    if kind in _warned_unhandled:
        return
    _warned_unhandled.add(kind)
    import warnings
    warnings.warn(
        f"No clone handler for field values of type {kind.__module__}.{kind.__qualname__}; "
        f"it will be shared by reference between the initial state and every stage buffer. "
        f"If it holds mutable state, every stage will write into the same object and the "
        f"result will be wrong with no error raised. Register one with "
        f"integrators.register_clone_handler(...).",
        RuntimeWarning,
        stacklevel=4,
    )


# ---- built-in handlers ----------------------------------------------------- #

register_clone_handler(
    'torch.Tensor',
    matches=lambda v: isinstance(v, torch.Tensor),
    clone=lambda v, detach: v.detach().clone() if detach else v.clone(),
    to_device=lambda v, device: v.to(device),
)


def _is_warp_array(value) -> bool:
    """Duck-typed so importing this package never imports warp."""
    module = type(value).__module__ or ''
    return module.split('.')[0] == 'warp' and type(value).__name__ == 'array'


def _clone_warp(value, detach):
    import warp as wp
    return wp.clone(value, requires_grad=False if detach else None)


def _warp_to_device(value, device):
    import warp as wp
    return wp.clone(value, device=device)


register_clone_handler('warp.array', _is_warp_array, _clone_warp, _warp_to_device)


def _clone_sequence(value, detach):
    cloned = [clone_value(v, detach) for v in value]
    if isinstance(value, tuple):
        # NamedTuples take their fields positionally, plain tuples take an iterable.
        return type(value)(*cloned) if hasattr(value, '_fields') else type(value)(cloned)
    return type(value)(cloned)


def _sequence_to_device(value, device):
    moved = [move_value(v, device) for v in value]
    if isinstance(value, tuple):
        return type(value)(*moved) if hasattr(value, '_fields') else type(value)(moved)
    return type(value)(moved)


register_clone_handler(
    'list/tuple',
    matches=lambda v: isinstance(v, (list, tuple)),
    clone=_clone_sequence,
    to_device=_sequence_to_device,
)

register_clone_handler(
    'dict',
    matches=lambda v: isinstance(v, dict),
    clone=lambda v, detach: type(v)((k, clone_value(x, detach)) for k, x in v.items()),
    to_device=lambda v, device: type(v)((k, move_value(x, device)) for k, x in v.items()),
)


# Kept as the internal spelling used by the state helpers below.
_op = clone_value
_empty_like = empty_value


def _clone_state(state, *, include_ephemeral=False, detach=False):
    """Generic clone driven by behavior metadata. No per-field manual code."""
    kwargs = {}
    for f in dataclasses.fields(state):
        value = getattr(state, f.name)
        if field_behavior(f) == 'ephemeral' and not include_ephemeral:
            kwargs[f.name] = _empty_like(value)
        else:
            kwargs[f.name] = _op(value, detach)
    return type(state)(**kwargs)


def _state_initialize(state, detach=False):
    kwargs = {}
    for f in dataclasses.fields(state):
        value = getattr(state, f.name)
        if field_behavior(f) in ('constant', 'integrated'):
            kwargs[f.name] = _op(value, detach)
        else:
            kwargs[f.name] = _empty_like(value)
    return type(state)(**kwargs)


def _state_finalize(live_state, last_substep_state):
    """Copy 'copied' fields from last substep. Leave everything else alone."""
    for f in dataclasses.fields(live_state):
        if field_behavior(f) == 'copied':
            setattr(live_state, f.name, getattr(last_substep_state, f.name))
    return live_state


def _maybe_reference_state(obj):
    """The tagged reference state of a system, or the object itself if it has none.

    Schemes work with systems, but `copied` is a property of state fields. This lets
    the finalize path accept either without the caller having to know which it has.
    """
    try:
        return get_tagged_attr(obj, role='reference_state')
    except (LookupError, TypeError):
        return obj


def clear_ephemeral_fields(system):
    """Null out `ephemeral` fields in place.

    Most schemes get this for free, because they assemble the final state with a fresh
    `initializeNewState()`. PEFRL does not -- its final state *is* the buffer the last
    evaluation ran on -- so it has to ask, or stage-local scratch leaks out of the step.
    """
    state = _maybe_reference_state(system)
    for f in dataclasses.fields(state):
        if field_behavior(f) == 'ephemeral':
            setattr(state, f.name, _empty_like(getattr(state, f.name)))
    return system


def copy_finalized_fields(final_system, last_stage_system):
    """Carry `copied` fields from the last evaluated stage into the final state.

    A `copied` field is one the system recomputes per stage rather than integrating --
    summation density, pressure, a smoothing length. It is nulled in
    `initializeNewState` and so would otherwise arrive at the caller as `None`, which
    is what made the behaviour indistinguishable from `ephemeral` (NOTES.md 2.7).

    Called by `finalizeSystem` before the user's `finalize` hook runs, so a hook that
    wants to override the copied value still can.
    """
    if last_stage_system is None:
        return final_system
    _state_finalize(_maybe_reference_state(final_system), _maybe_reference_state(last_stage_system))
    return final_system


def _diff_dataclass(a, b):
    """``a - b`` over ``integrated`` fields of one dataclass level. No nesting."""
    kwargs = {}
    for f in dataclasses.fields(a):
        value_a = getattr(a, f.name)
        if field_behavior(f) == 'integrated':
            kwargs[f.name] = value_a - getattr(b, f.name)
        else:
            kwargs[f.name] = _op(value_a, False)
    return type(a)(**kwargs)


def state_difference(state_a, state_b):
    """``a - b`` over ``integrated`` fields, as a new object of ``state_a``'s type.

    Non-integrated fields are cloned from ``state_a`` unchanged, matching what
    ``initializeNewState`` already does for them. This is the general primitive
    behind ``butcher._error_estimate`` (the difference between an embedded pair's two
    solutions) and behind a Newton/Picard stage residual (the difference between two
    iterates) -- both are "subtract two same-shaped states", so it lives here once
    rather than being hand-rolled per call site.

    ``state_a``/``state_b`` may be raw states or systems that tag one with
    ``reference_state``; either way the result has the *same shape as the input*
    (a differenced system stays a system) so callers can keep using
    ``get_reference_state`` on it, unlike a version that unwrapped down to the inner
    state and lost the wrapper.
    """
    try:
        ref_field = find_tagged_field(state_a, role='reference_state')
    except (LookupError, TypeError):
        return _diff_dataclass(state_a, state_b)

    inner = _diff_dataclass(getattr(state_a, ref_field.name), getattr(state_b, ref_field.name))
    kwargs = {}
    for f in dataclasses.fields(state_a):
        if f.name == ref_field.name:
            kwargs[f.name] = inner
        else:
            kwargs[f.name] = _op(getattr(state_a, f.name), False)
    return type(state_a)(**kwargs)


def state_norm(state, rtol: float = 1e-3, atol: float = 1e-6, *, reference=None) -> float:
    """Hairer-Wanner weighted RMS norm over ``state``'s ``integrated`` fields.

    ``sqrt(mean((state_i / (atol + rtol*|reference_i|))**2))``, pooled over every
    element of every integrated field. ``reference`` supplies the scale (``atol +
    rtol*|reference|``) and defaults to ``state`` itself, which is the right choice
    when ``state`` already *is* the quantity whose own magnitude should set the
    scale (e.g. a Newton iterate); pass the base solution explicitly when ``state``
    is a difference or residual instead (e.g. an embedded-pair error estimate, scaled
    against the propagated solution rather than against its own, physically
    meaningless magnitude).

    Returns 0.0 for a state with no integrated tensor fields.
    """
    s = _maybe_reference_state(state)
    ref = _maybe_reference_state(reference if reference is not None else state)
    total_sq = 0.0
    total_n = 0
    for f in dataclasses.fields(s):
        if field_behavior(f) != 'integrated':
            continue
        value = getattr(s, f.name)
        if not isinstance(value, torch.Tensor):
            continue
        ref_value = getattr(ref, f.name)
        scale = atol + rtol * ref_value.abs()
        weighted = value / scale
        total_sq += float((weighted ** 2).sum())
        total_n += weighted.numel()
    if total_n == 0:
        return 0.0
    return math.sqrt(total_sq / total_n)


@dataclass
class BaseState:
    def initializeNewState(self, **kwargs):
        return _state_initialize(self)

    def clone(self):
        return _clone_state(self, include_ephemeral=True)

    def nograd(self):
        return _clone_state(self, detach=True)

    def to(self, device):
        # Dispatched, so a warp array or a list of tensors moves too rather than
        # silently staying where it was.
        return type(self)(**{f.name: move_value(getattr(self, f.name), device)
                             for f in dataclasses.fields(self)})
    

from .specs import ComponentUpdateSpec, PositionUpdateSpec


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
    if hasattr(spec, 'current_velocity_dt') and spec.current_velocity_dt is not None:
        value = value + spec.current_velocity_dt * get_tagged_attr(state, tag=velocity_tag)
    if hasattr(spec, 'update_velocity_dt') and spec.update_velocity_dt is not None and has_derivative and velocity_derivative_tag is not None:
        value = value + spec.update_velocity_dt * _resolve_delta(update, velocity_derivative_tag)
    set_tagged_attr(state, value, tag=position_tag)
    return system

