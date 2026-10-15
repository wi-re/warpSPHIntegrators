"""Adapt floating-point tensor ODEs f(t, y) to the existing state protocol."""

from dataclasses import dataclass
import math
from typing import ClassVar

import torch

from warpSPHIntegrators.state.fields import BaseState, integrated, reference_state, tagged, update_component
from warpSPHIntegrators.state.protocol import BaseIntegrationSystem
from warpSPHIntegrators.adapters.rhs import RHS


@dataclass
class _TensorFields(BaseState):
    y: torch.Tensor = integrated('dy', tags=('quantity',))


@dataclass
class _TensorUpdate:
    dy: torch.Tensor = tagged(tags=('quantity_derivative',))


@dataclass
class TensorState(BaseIntegrationSystem):
    """A protocol-ready state: TensorState(y0, t=0); read the tensor as .y.

    Scalar, vector and batched tensors keep their shape, device, dtype and graph.
    The RHS must return a tensor of exactly the same shape/device/dtype.
    Use catalog records with state_kind='tensor'; geometric/Newmark methods need
    an explicit position/velocity system. Use solve() or a caller-owned stepping loop.
    """

    state: _TensorFields | torch.Tensor = reference_state()
    t: float = 0.0
    _tensor_adapter: ClassVar[bool] = True

    def __post_init__(self):
        if isinstance(self.state, torch.Tensor):
            self.state = _TensorFields(self.state)
        if not isinstance(self.state, _TensorFields):
            raise TypeError('TensorState needs a floating-point torch.Tensor')
        if not isinstance(self.y, torch.Tensor) or not self.y.is_floating_point() or not self.y.numel():
            raise TypeError('TensorState needs a nonempty floating-point torch.Tensor')
        if not math.isfinite(float(self.t)):
            raise ValueError('TensorState time must be finite')

    @property
    def y(self):
        return self.state.y

    def initializeNewState(self, *args, **kwargs):
        return TensorState(self.state.initializeNewState(), self.t)

    def apply_quantity_update(self, update, spec, **kwargs):
        return update_component(self, update, spec, 'quantity', 'quantity_derivative')

    def apply_state_update(self, update, spec, **kwargs):
        return self.apply_quantity_update(update, spec, **kwargs)

    def apply_position_update(self, update, spec, **kwargs):
        return self

    def apply_velocity_update(self, update, spec, **kwargs):
        return self


def tensor_rhs(function, /, **parameters):
    """Adapt f(t, y, *args, **parameters), including structured RHS parts.

    Bind named RHS parameters here (or in a closure). Step options such as
    solver_opts/verbose are not passed to f. Positional step arguments are
    forwarded. Inputs must be treated as read-only, as with the full protocol.
    A structured RHS retains its declared additive/semilinear capabilities.
    """
    def adapt(fn, *, combined=False):
        if fn is None:
            return None
        if isinstance(fn,RHS) and not combined:
            return tensor_rhs(fn,**parameters)
        if not callable(fn):
            raise TypeError('tensor_rhs needs a callable or structured RHS')

        def evaluate(system, dt, *args, **kwargs):
            y = system.y
            dy = fn(system.t, y, *args, **parameters)
            if not isinstance(dy, torch.Tensor):
                raise TypeError('Tensor RHS must return a torch.Tensor')
            if dy.shape != y.shape or dy.dtype != y.dtype or dy.device != y.device:
                raise ValueError('Tensor RHS must preserve state shape, dtype and device')
            return _TensorUpdate(dy), None
        return evaluate

    if isinstance(function, RHS):
        return RHS(combined=adapt(function,combined=True), **{
            name: adapt(getattr(function, '_' + name)) for name in
            ('explicit', 'implicit', 'linear', 'nonlinear','conservative','dissipative')
        },autonomous=function.autonomous)
    return adapt(function)
