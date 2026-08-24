"""Reference systems and convergence measurement, for tests and for the scripts.

This is the smallest complete implementation of the integration protocol: a state
with tagged position / velocity / quantity fields, an update dataclass with the
matching derivative tags, and a system that routes the four ``apply_*_update`` calls
through the generic ``update_component`` / ``update_position`` kernels. It doubles as
worked documentation of what a user's system has to provide.

Three problems are provided, chosen so that between them they catch every class of
integration defect this library has had:

``oscillator``
    Harmonic oscillator, autonomous and Hamiltonian. Measures nominal convergence
    order, and its energy drift separates symplectic schemes from dissipative ones.
``forced``
    ``x'' = cos(t)``. Non-autonomous, so a scheme that evaluates the right-hand side
    at the wrong stage *time* collapses to first order here while still looking
    correct on ``oscillator``.
``kepler``
    Circular two-body orbit. Nonlinear and Hamiltonian, with an analytic solution,
    so it catches errors that a linear problem cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence

import torch

from .fields import BaseState, constant, integrated, get_reference_state, reference_state, tagged, update_component, update_position
from .history import StepHistory
from .protocol import BaseIntegrationSystem
from .specs import ComponentUpdateSpec, PositionUpdateSpec


# --------------------------------------------------------------------------- #
# Reference system                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class ParticleState(BaseState):
    """Position, velocity, and one passively integrated quantity."""

    x: torch.Tensor = integrated('dxdt', tags=('position',))
    u: torch.Tensor = integrated('dudt', tags=('velocity',))
    e: torch.Tensor = integrated('dedt', tags=('quantity',))
    m: torch.Tensor = constant(tags=('mass',))


@dataclass
class ParticleUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dudt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dedt: torch.Tensor = tagged(tags=('quantity_derivative',))


@dataclass
class ParticleSystem(BaseIntegrationSystem):
    state: ParticleState = reference_state(tags=('particles',))
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        # `t` is forwarded on purpose. The integrator sets stage times explicitly, so
        # dropping it here is no longer fatal (see NOTES.md 2.3), but forwarding it is
        # what a real system should do.
        return ParticleSystem(state=get_reference_state(self).initializeNewState(), t=self.t)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'quantity', 'quantity_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        self.apply_position_update(
            update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend), **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self


# --------------------------------------------------------------------------- #
# Problems                                                                     #
# --------------------------------------------------------------------------- #

class Problem(NamedTuple):
    name: str
    description: str
    #: f(system, dt, **kwargs) -> (ParticleUpdate, aux)
    rhs: Callable
    #: initial system at t=0
    initial: Callable[[], ParticleSystem]
    #: analytic (x, u) at time t, as flat lists
    exact: Callable[[float], tuple]
    #: total energy of a state, for the Hamiltonian drift tests; None if not Hamiltonian
    energy: Optional[Callable[[ParticleState], float]] = None
    #: True for a right-hand side that depends explicitly on t
    autonomous: bool = True


def _vec(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def oscillator_problem(k: float = 4.0) -> Problem:
    """x'' = -(k/m) x, with x(0) = 1, u(0) = 0. Autonomous and Hamiltonian."""
    omega = math.sqrt(k)

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        return ParticleUpdate(dxdt=s.u.clone(),
                              dudt=-k / s.m * s.x,
                              dedt=torch.zeros_like(s.e)), None

    def initial():
        return ParticleSystem(state=ParticleState(x=_vec(1.0), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
                              t=0.0)

    def energy(state):
        return float(0.5 * (state.u ** 2).sum() + 0.5 * k * (state.x ** 2).sum())

    return Problem('oscillator',
                   f'harmonic oscillator, k={k}, m=1 (autonomous, Hamiltonian), x(0)=1, u(0)=0',
                   rhs, initial,
                   lambda t: ([math.cos(omega * t)], [-omega * math.sin(omega * t)]),
                   energy, autonomous=True)


def forced_problem() -> Problem:
    """x'' = cos(t), with x(0) = 1, u(0) = 0, so x = 2 - cos t.

    The right-hand side reads ``system.t``, so every stage-time error shows up as an
    order collapse. This is the problem that exposes NOTES.md 2.2 and 2.3.
    """

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        return ParticleUpdate(dxdt=s.u.clone(),
                              dudt=torch.full_like(s.u, math.cos(float(system.t))),
                              dedt=torch.zeros_like(s.e)), None

    def initial():
        return ParticleSystem(state=ParticleState(x=_vec(1.0), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
                              t=0.0)

    return Problem('forced',
                   "x'' = cos(t) (non-autonomous), x(0)=1, u(0)=0",
                   rhs, initial,
                   lambda t: ([2 - math.cos(t)], [math.sin(t)]),
                   None, autonomous=False)


def damped_problem(k: float = 4.0, c: float = 0.5) -> Problem:
    """x'' = -k x - c u, with x(0) = 1, u(0) = 0. Autonomous, velocity-dependent.

    The velocity dependence matters for reuse analysis: velocity Verlet consumes only
    the *acceleration* of the reused stage, which is evaluated at the exact x^{n+1},
    so reuse is lossless whenever the force depends on position alone. As soon as the
    force sees velocity -- artificial viscosity, drag, any real SPH momentum equation
    -- the reused stage carries the half-step velocity v^{n+1/2} and reuse drops to
    first order. This problem is what makes that distinction measurable.
    """
    gamma = c / 2
    omega_d = math.sqrt(k - gamma ** 2)

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        return ParticleUpdate(dxdt=s.u.clone(),
                              dudt=-k / s.m * s.x - c * s.u,
                              dedt=torch.zeros_like(s.e)), None

    def initial():
        return ParticleSystem(state=ParticleState(x=_vec(1.0), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
                              t=0.0)

    def exact(t):
        decay = math.exp(-gamma * t)
        x = decay * (math.cos(omega_d * t) + gamma / omega_d * math.sin(omega_d * t))
        u = -decay * (k / omega_d) * math.sin(omega_d * t)
        return ([x], [u])

    return Problem('damped',
                   f'damped oscillator, x\'\' = -{k} x - {c} u (autonomous, velocity-dependent)',
                   rhs, initial, exact, None, autonomous=True)


def kepler_problem() -> Problem:
    """Circular two-body orbit: a = -x/|x|^3, x(0) = (1,0), u(0) = (0,1).

    Nonlinear and Hamiltonian, with the exact solution x(t) = (cos t, sin t).
    """

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        r = torch.linalg.norm(s.x)
        return ParticleUpdate(dxdt=s.u.clone(),
                              dudt=-s.x / r ** 3,
                              dedt=torch.zeros_like(s.e)), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=_vec(1.0, 0.0), u=_vec(0.0, 1.0), e=_vec(0.0, 0.0), m=_vec(1.0, 1.0)),
            t=0.0)

    def energy(state):
        return float(0.5 * (state.u ** 2).sum() - 1.0 / torch.linalg.norm(state.x))

    return Problem('kepler',
                   'circular Kepler orbit, a = -x/|x|^3 (autonomous, Hamiltonian, nonlinear)',
                   rhs, initial,
                   lambda t: ([math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]),
                   energy, autonomous=True)


PROBLEMS = {
    'oscillator': oscillator_problem,
    'forced': forced_problem,
    'damped': damped_problem,
    'kepler': kepler_problem,
}


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #

def run(scheme, problem: Problem, dt: float, T: float, *,
        reuse: bool = False, history: bool = False, history_length: int = 4) -> ParticleSystem:
    """Integrate ``problem`` from t=0 to t=T in fixed steps of ``dt``.

    ``history=True`` threads a ``StepHistory`` through the run instead of (or, if
    ``reuse`` is also set, alongside) the single-entry ``priorStep``. A one-step
    scheme ignores a ``history`` kwarg it doesn't recognise -- ``RungeKuttaB`` pops it
    like ``priorStep`` and only acts on it if given one -- so this is safe to pass for
    every registered scheme today; it exists so a multistep scheme (NOTES.md S3
    Phase 1) or a step-reuse test that wants more than one entry of lookback can
    exercise this without every other test's ``run`` call changing shape first
    (NOTES.md S5).
    """
    system = problem.initial()
    prior = None
    step_history = StepHistory(maxlen=history_length) if history else None
    for _ in range(int(round(T / dt))):
        kwargs = {}
        if reuse:
            kwargs['priorStep'] = prior
        if history:
            kwargs['history'] = step_history
        result = scheme(system, dt=dt, f=problem.rhs, **kwargs)
        system = result.state
        if reuse:
            prior = result.stages[-1]
        if history and result.history is not None:
            step_history = result.history
    return system


def final_error(scheme, problem: Problem, dt: float, T: float, *,
                reuse: bool = False, history: bool = False) -> float:
    """L1 error in (x, u) at time T against the analytic solution."""
    system = run(scheme, problem, dt, T, reuse=reuse, history=history)
    s = get_reference_state(system)
    ex, eu = problem.exact(T)
    return (sum(abs(a - b) for a, b in zip(s.x.tolist(), ex))
            + sum(abs(a - b) for a, b in zip(s.u.tolist(), eu)))


def default_step_sizes(dt: float = 0.1, levels: int = 4) -> List[float]:
    return [dt / 2 ** i for i in range(levels)]


def measured_order(errors: Sequence[Optional[float]], dts: Sequence[float],
                   floor: float = 1e-12) -> Optional[float]:
    """Convergence order from the finest usable pair, skipping the roundoff floor.

    Returns None if fewer than two points sit above ``floor`` -- which happens when a
    scheme is so accurate on a problem that double precision runs out before the
    asymptotic regime does.
    """
    pairs = [(e, d) for e, d in zip(errors, dts) if e is not None and e > floor]
    if len(pairs) < 2:
        return None
    (e0, d0), (e1, d1) = pairs[-2], pairs[-1]
    return math.log(e0 / e1) / math.log(d0 / d1)


def convergence(scheme, problem: Problem, dts: Sequence[float], T: float = 2.0,
                *, reuse: bool = False):
    """(measured order, errors) for ``scheme`` on ``problem`` over ``dts``."""
    errors = [final_error(scheme, problem, dt, T, reuse=reuse) for dt in dts]
    return measured_order(errors, dts), errors


def energy_drift(scheme, problem: Problem, dt: float, T: float) -> float:
    """Relative change in total energy over [0, T]. Requires a Hamiltonian problem."""
    if problem.energy is None:
        raise ValueError(f'{problem.name} is not a Hamiltonian problem')
    start = problem.initial()
    e0 = problem.energy(get_reference_state(start))
    final = run(scheme, problem, dt, T)
    e1 = problem.energy(get_reference_state(final))
    return abs(e1 - e0) / abs(e0)


def max_energy_drift(scheme, problem: Problem, dt: float, T: float) -> float:
    """Largest relative energy error seen at any step in [0, T].

    This, not the endpoint value, is what distinguishes a symplectic scheme from a
    dissipative one. A symplectic integrator conserves a shadow Hamiltonian, so its
    energy error *oscillates* within an O(dt^p) band no matter how long the run; a
    non-symplectic one accumulates error secularly and the maximum grows with T.
    Sampling only the endpoint measures the phase of the oscillation, not its bound.
    """
    if problem.energy is None:
        raise ValueError(f'{problem.name} is not a Hamiltonian problem')
    system = problem.initial()
    e0 = problem.energy(get_reference_state(system))
    worst = 0.0
    for _ in range(int(round(T / dt))):
        system = scheme(system, dt=dt, f=problem.rhs, priorStep=None).state
        worst = max(worst, abs(problem.energy(get_reference_state(system)) - e0) / abs(e0))
    return worst
