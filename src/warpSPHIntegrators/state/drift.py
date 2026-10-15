"""Drift fields: configuration quantities advanced with the drifts.

A partitioned (Verlet-family) scheme moves the positions in *drifts* (``x += tau
* v``) and the velocities in *kicks* (``v += tau * a``); every other integrated
field used to ride along with the kicks, updated from the stage rate like a
velocity. That is wrong for a quantity whose rate is a function of the
configuration times the velocity -- the weakly-compressible continuity density
``drho/dt = -rho div v`` above all. Such a quantity is a configuration
variable (with summation density it is literally ``rho(x)``), so it has to be
advanced like the positions: with the velocity the positions drift with.

Getting this wrong costs stability, not just accuracy. On the linear acoustic
pair (``rho' = A v``, ``v' = -B rho``, ``omega^2 = AB``, ``h = omega dt``) the
kick-attached update gives, per step,

    symplecticEuler (DualSPHysics' predictor-corrector)  det = 1 + h^4/4
    velocityVerlet                                       det = 1 + h^2/2
    leapFrog, semiImplicitEuler                          det = 1 + h^2
    PEFRL, VEFRL                                         ~ 1 + h^2

i.e. growth at every step size, held down only by whatever dissipation the
system carries. With the density in the drifts each becomes a symplectic map of
the pair (det = 1, stable for h below the scheme's own limit).
``tests/test_drift_fields.py`` measures both on ``testing.acoustic_problem``.

The contract (all optional; a system without it runs exactly as before):

* declare the field ``integrated(update_key, drift_key=...)`` (``fields.py``);
* the right-hand side's update object carries the velocity-linear part of that
  field's rate under ``drift_key`` (the total stays under ``update_key``);
* the system implements ``drift_rates(velocities, aux=None, **kwargs)``: the
  ``drift_key`` rates at *its own* configuration for the given velocity field,
  without mutating itself; ``aux`` is the stage's auxiliary return (e.g. a
  neighbour list), or None where no stage ran on that configuration;
* optionally ``drift_fields_enabled(**kwargs) -> bool`` to switch it per run.

Runge-Kutta, multistep and implicit schemes treat every field alike, so there is
nothing to fix there and they ignore the declaration.
"""

import copy
import dataclasses
from typing import Dict, List, NamedTuple

from warpSPHIntegrators.state.fields import field_behavior, get_reference_state, get_tagged_attr

__all__ = [
    'DriftField', 'drift_fields', 'drift_fields_active', 'drift_rates',
    'kinematic_part', 'nonkinematic_part', 'without_kinematic', 'set_drift_fields', 'add_to_drift_fields',
    'velocity_of', 'acceleration_of',
]


class DriftField(NamedTuple):
    name: str          #: the state field
    update_key: str    #: its total rate on the update object
    drift_key: str     #: the velocity-linear part of that rate on the update object


def drift_fields(system) -> List[DriftField]:
    """The drift fields declared on ``system``'s reference state (empty if none)."""
    state = get_reference_state(system)
    out = []
    for f in dataclasses.fields(state):
        if field_behavior(f) == 'integrated' and f.metadata.get('drift_key') is not None:
            out.append(DriftField(f.name, f.metadata['update_key'], f.metadata['drift_key']))
    return out


def drift_fields_active(system, **kwargs) -> bool:
    """True when the drift-field path applies: fields declared, ``drift_rates``
    implemented, and ``drift_fields_enabled`` (if the system has it) says so."""
    if not hasattr(system, 'drift_rates'):
        return False
    try:
        if not drift_fields(system):
            return False
    except (LookupError, TypeError):
        return False
    enabled = getattr(system, 'drift_fields_enabled', None)
    return True if enabled is None else bool(enabled(**kwargs))


def drift_rates(system, velocities, aux=None, **kwargs) -> Dict[str, object]:
    """``{field: rate}`` -- the velocity-linear rates at ``system``'s configuration
    evaluated with ``velocities``."""
    result = system.drift_rates(velocities, aux=aux, **kwargs)
    return {df.name: getattr(result, df.drift_key) for df in drift_fields(system)}


def kinematic_part(system, update) -> Dict[str, object]:
    """``{field: update.<drift_key>}``: the velocity-linear rate the stage computed."""
    return {df.name: getattr(update, df.drift_key) for df in drift_fields(system)}


def without_kinematic(system, update):
    """A shallow copy of ``update`` whose drift fields' total rates have their
    velocity-linear part removed -- what the kicks still apply."""
    out = copy.copy(update)
    for df in drift_fields(system):
        setattr(out, df.update_key, getattr(update, df.update_key) - getattr(update, df.drift_key))
    return out


def nonkinematic_part(system, update) -> Dict[str, object]:
    return {df.name: getattr(update, df.update_key) - getattr(update, df.drift_key)
            for df in drift_fields(system)}


def set_drift_fields(target, base, *terms):
    """``target.q = base.q + sum(scale * rates[q])`` for every drift field ``q``;
    ``terms`` are ``(scale, {field: rate})`` pairs."""
    tstate, bstate = get_reference_state(target), get_reference_state(base)
    for df in drift_fields(target):
        value = getattr(bstate, df.name)
        for scale, rates in terms:
            value = value + scale * rates[df.name]
        setattr(tstate, df.name, value)
    return target


def add_to_drift_fields(target, *terms):
    """``target.q += sum(scale * rates[q])`` for every drift field ``q``."""
    return set_drift_fields(target, target, *terms)


def velocity_of(system):
    return get_tagged_attr(get_reference_state(system), tag='velocity')


def acceleration_of(update):
    return get_tagged_attr(update, tag='velocity_derivative')
