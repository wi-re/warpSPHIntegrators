"""Field values other than `torch.Tensor` must actually be copied, not aliased.

`_op` and `_empty_like` used to special-case `torch.Tensor` and pass every other type
through **by reference**. A state holding `wp.array` fields -- or a plain list or dict
of tensors -- was therefore not cloned at all: every RK stage wrote into the same
buffer as the initial state, stage 2 read stage-1-corrupted data, and nothing raised
anywhere (NOTES.md 4.1).

That is the worst failure mode this library can have, because it is silent. These tests
cover the dispatch itself and then the thing that actually matters: an integration step
that would corrupt its own input under the old behaviour.
"""

import warnings
from dataclasses import dataclass

import pytest
import torch

from integrators import (
    BaseIntegrationSystem,
    BaseState,
    ComponentUpdateSpec,
    PositionUpdateSpec,
    clone_value,
    constant,
    empty_value,
    ephemeral,
    get_reference_state,
    integrated,
    move_value,
    reference_state,
    register_clone_handler,
    tagged,
    testing,
)
from integrators import fields as F

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:                                                # pragma: no cover
    HAVE_WARP = False

needs_warp = pytest.mark.skipif(not HAVE_WARP, reason='warp not available')


@pytest.fixture(autouse=True)
def _reset_unhandled_warnings():
    """The 'no handler' warning fires once per type per process."""
    F._warned_unhandled.clear()
    yield
    F._warned_unhandled.clear()


# --------------------------------------------------------------------------- #
# clone_value                                                                  #
# --------------------------------------------------------------------------- #

def test_tensor_is_cloned():
    t = torch.zeros(3)
    assert clone_value(t) is not t
    assert torch.equal(clone_value(t), t)


def test_list_is_cloned_elementwise():
    original = [torch.zeros(3), torch.ones(3)]
    copy = clone_value(original)
    assert copy is not original
    assert all(a is not b for a, b in zip(copy, original))
    copy[0] += 5
    assert original[0].sum() == 0, 'writing to the copy reached the original'


def test_dict_is_cloned_valuewise():
    original = {'a': torch.zeros(3)}
    copy = clone_value(original)
    assert copy is not original and copy['a'] is not original['a']
    copy['a'] += 5
    assert original['a'].sum() == 0


def test_nested_containers_recurse():
    original = [{'b': (torch.zeros(2),)}]
    copy = clone_value(original)
    assert copy[0]['b'][0] is not original[0]['b'][0]
    copy[0]['b'][0].add_(1)          # in place: the tuple itself is immutable
    assert original[0]['b'][0].sum() == 0


def test_namedtuple_type_is_preserved():
    from collections import namedtuple
    Pair = namedtuple('Pair', 'x y')
    original = Pair(torch.zeros(2), torch.ones(2))
    copy = clone_value(original)
    assert isinstance(copy, Pair) and copy.x is not original.x


def test_immutables_pass_through():
    for value in (None, True, 3, 2.5, 'fluid', b'x', 1 + 2j, frozenset({1})):
        assert clone_value(value) is value


def test_detach_propagates_into_containers():
    original = [torch.zeros(3, requires_grad=True)]
    assert clone_value(original, detach=False)[0].requires_grad
    assert not clone_value(original, detach=True)[0].requires_grad


@needs_warp
def test_warp_array_is_cloned():
    a = wp.zeros(3, dtype=float)
    b = clone_value(a)
    assert b is not a
    wp.copy(b, wp.full(3, 7.0, dtype=float))
    assert a.numpy().tolist() == [0.0, 0.0, 0.0], 'writing to the clone reached the original'
    assert b.numpy().tolist() == [7.0, 7.0, 7.0]


# --------------------------------------------------------------------------- #
# empty_value                                                                  #
# --------------------------------------------------------------------------- #

def test_buffers_are_emptied_but_plain_values_are_not():
    assert empty_value(torch.zeros(3)) is None
    assert empty_value([torch.zeros(3)]) is None
    assert empty_value({'a': torch.zeros(3)}) is None
    # A scalar parameter sitting on an ephemeral field must survive; nulling it would
    # break systems that read it back.
    assert empty_value(2.5) == 2.5
    assert empty_value('fluid') == 'fluid'


@needs_warp
def test_warp_array_is_emptied():
    assert empty_value(wp.zeros(3, dtype=float)) is None


# --------------------------------------------------------------------------- #
# Unknown types                                                                #
# --------------------------------------------------------------------------- #

class Opaque:
    """Stands in for a user type the library knows nothing about."""


def test_unknown_type_warns_once_and_passes_through():
    value = Opaque()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert clone_value(value) is value
        clone_value(Opaque())
        clone_value(Opaque())
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1, f'expected one warning per type, got {len(runtime)}'
    assert 'register_clone_handler' in str(runtime[0].message)


def test_registering_a_handler_silences_the_warning_and_copies():
    register_clone_handler(
        'test.Opaque',
        matches=lambda v: isinstance(v, Opaque),
        clone=lambda v, detach: Opaque(),
    )
    try:
        value = Opaque()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            copy = clone_value(value)
        assert copy is not value
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    finally:
        F._CLONE_HANDLERS.pop()


def test_later_registration_wins():
    """So a user can override a built-in for an overlapping type."""
    sentinel = torch.full((1,), 99.0)
    register_clone_handler('test.override', lambda v: isinstance(v, torch.Tensor),
                           lambda v, detach: sentinel)
    try:
        assert clone_value(torch.zeros(3)) is sentinel
    finally:
        F._CLONE_HANDLERS.pop()
    assert clone_value(torch.zeros(3)) is not sentinel


# --------------------------------------------------------------------------- #
# move_value / BaseState.to                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class ContainerState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    bag: list = constant(default=None)
    table: dict = constant(default=None)
    scratch: list = ephemeral(default=None)


def _container_state():
    return ContainerState(x=torch.zeros(3), bag=[torch.zeros(3)],
                          table={'a': torch.zeros(3)}, scratch=[torch.ones(3)])


def test_to_device_reaches_inside_containers():
    moved = _container_state().to('cpu')
    assert moved.bag[0].device.type == 'cpu'
    assert moved.table['a'].device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_to_cuda_moves_containers():
    moved = _container_state().to('cuda')
    assert moved.x.device.type == 'cuda'
    assert moved.bag[0].device.type == 'cuda', 'a list of tensors stayed on the CPU'
    assert moved.table['a'].device.type == 'cuda'


def test_move_value_leaves_unknown_types_alone():
    value = Opaque()
    assert move_value(value, 'cpu') is value


# --------------------------------------------------------------------------- #
# State-level behaviour                                                        #
# --------------------------------------------------------------------------- #

def test_container_fields_are_cloned_by_both_paths():
    state = _container_state()
    for copy in (state.clone(), state.initializeNewState()):
        assert copy.bag is not state.bag
        assert copy.bag[0] is not state.bag[0]
        assert copy.table['a'] is not state.table['a']


def test_ephemeral_container_is_nulled_not_shared():
    state = _container_state()
    assert state.initializeNewState().scratch is None


# --------------------------------------------------------------------------- #
# The thing that actually matters: a step must not corrupt its own input       #
# --------------------------------------------------------------------------- #

@dataclass
class BufferState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    u: torch.Tensor = integrated('dudt', tags=('velocity',))
    e: torch.Tensor = integrated('dedt', tags=('quantity',))
    m: torch.Tensor = constant(tags=('mass',))
    #: Stands in for anything a real system keeps in a container: per-stage scratch,
    #: a neighbour list, a set of per-species buffers.
    buffers: list = constant(default=None)


@dataclass
class BufferUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dudt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dedt: torch.Tensor = tagged(tags=('quantity_derivative',))


@dataclass
class BufferSystem(BaseIntegrationSystem):
    state: BufferState = reference_state(tags=('particles',))
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        return BufferSystem(state=get_reference_state(self).initializeNewState(), t=self.t)

    def preprocess(self, initialState, dt, *args, **kwargs):
        """A stage writing into its own buffer, as an SPH preprocess would."""
        get_reference_state(self).buffers[0] += 1.0
        return self

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return F.update_position(self, update, spec, 'position', 'position_derivative',
                                 'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return F.update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return F.update_component(self, update, spec, 'quantity', 'quantity_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        self.apply_position_update(
            update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend), **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self


def _buffer_rhs(system, dt, **kwargs):
    s = get_reference_state(system)
    return BufferUpdate(dxdt=s.u.clone(), dudt=-4.0 * s.x, dedt=torch.zeros_like(s.e)), None


def _buffer_system():
    one = torch.ones(1, dtype=torch.float64)
    return BufferSystem(state=BufferState(
        x=one.clone(), u=torch.zeros(1, dtype=torch.float64),
        e=torch.zeros(1, dtype=torch.float64), m=one.clone(),
        buffers=[torch.zeros(1, dtype=torch.float64)]))


def test_stage_writes_do_not_reach_the_callers_buffer(scheme):
    """Under the old aliasing, every stage bumped the caller's tensor."""
    system = _buffer_system()
    before = system.state.buffers[0].item()
    scheme(system, dt=0.05, f=_buffer_rhs)
    assert system.state.buffers[0].item() == before, (
        f'{scheme.name}: the caller\'s buffer was written {system.state.buffers[0].item() - before:.0f} '
        f'times by stage preprocessing -- the list was shared, not cloned'
    )


def test_final_state_owns_its_buffer(scheme):
    """The returned state must hold its own container, not the caller's object.

    Identity, not accumulated value: PEFRL and VEFRL are compositions of maps, so each
    of their stage buffers is legitimately built from the previous stage rather than
    from the initial state, and a container written by every stage genuinely
    accumulates. What must never happen -- for any scheme -- is that the object handed
    back is the same one the caller passed in.
    """
    system = _buffer_system()
    result = scheme(system, dt=0.05, f=_buffer_rhs)
    final = get_reference_state(result.state)
    assert final.buffers is not system.state.buffers, f'{scheme.name}: returned the caller\'s list'
    assert final.buffers[0] is not system.state.buffers[0], (
        f'{scheme.name}: returned the caller\'s tensor inside a fresh list'
    )


@pytest.mark.parametrize('name', ['RK4', 'Midpoint', 'SSP RK3', 'Dormand-Prince 5(4)'])
def test_runge_kutta_stages_each_start_from_the_initial_state(name):
    """For a Butcher tableau every stage buffer is built from y^n, so writes never stack."""
    from integrators import getIntegrator

    system = _buffer_system()
    result = getIntegrator(name)(system, dt=0.05, f=_buffer_rhs)
    final = get_reference_state(result.state).buffers[0].item()
    assert final <= 1.0, (
        f'{name}: the final buffer accumulated {final:.0f} stage writes; each stage should '
        f'start from a fresh copy of the initial value'
    )


def test_container_state_still_integrates_correctly():
    """The dispatch must not disturb the numerics."""
    from integrators import getIntegrator

    s = getIntegrator('RK4')
    order, _ = testing.convergence(s, testing.PROBLEMS['oscillator'](),
                                   testing.default_step_sizes(), 2.0)
    assert order >= 4 - 0.15
