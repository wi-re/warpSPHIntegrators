"""Reference systems and convergence measurement, for tests and for the scripts.

This is the smallest complete implementation of the integration protocol: a state
with tagged position / velocity / quantity fields, an update dataclass with the
matching derivative tags, and a system that routes the four ``apply_*_update`` calls
through the generic ``update_component`` / ``update_position`` kernels. It doubles as
worked documentation of what a user's system has to provide.

The registry (``PROBLEMS``) provides:

``oscillator``
    Harmonic oscillator, autonomous and Hamiltonian. Measures nominal convergence
    order, and its energy drift separates symplectic schemes from dissipative ones.
``forced``
    ``x'' = cos(t)``. Non-autonomous, so a scheme that evaluates the right-hand side
    at the wrong stage *time* collapses to first order here while still looking
    correct on ``oscillator``.
``damped``
    ``x'' = -k x - c u``. Velocity-dependent force; separates lossless reuse
    (position-only forces) from lossy reuse.
``kepler``
    Circular two-body orbit. Nonlinear and Hamiltonian, with an analytic solution,
    so it catches errors that a linear problem cannot.
``stiffRelaxation``
    Prothero-Robinson ``x' = -sign*rate*(x - s(t)) + s'(t)`` with configurable
    stiffness ``rate``, sign (stable ``+1`` / unstable ``-1`` variant), and smooth
    target ``s`` (``tanh``/``sin``/``cos``); exact solution ``x = s`` in both
    variants. The nonlinear stiff reference problem.
``stiffDampedOscillator``
    ``x'' = -omega^2 x - c u`` with independently scalable frequency and damping,
    separating high-frequency stiffness from true dissipative stiffness; closed-form
    solution in all three damping regimes.
``vanDerPol``
    ``x'' = mu*(1 - x^2)*u - x``. Nonlinear, no closed form; the attractor is a
    limit cycle of amplitude ~2, so tests check the cycle instead of an exact value.
``robertson``
    The classic three-species chemical kinetics (stiffness ratio ~1e9). Invariants
    (mass conservation, positivity, monotone ``y3``, quasi-steady ``y2``) stand in
    for the unknown exact solution.
``diffusion``
    Semi-discrete 1D diffusion on ``n`` interior points with Dirichlet boundaries,
    initialized on two eigenvectors of the discrete Laplacian so the exact
    semi-discrete solution is known and the spectral stiffness scale is
    ``|mu_n| ~= 4*D/h^2``.
``advection``
    Semi-discrete 1D periodic advection ``u_t + c u_x = 0`` with first-order
    upwind differences -- the model hyperbolic problem for the TVD/SSP
    classifier (NOTES S3.11). ``ic='mode'`` initialises on two Fourier modes,
    which are exact eigenmodes of the periodic upwind operator, so the exact
    semi-discrete solution is known; ``ic='step'`` is a periodic square wave
    (no closed form -- use the total-variation checks).
``burgers``
    Semi-discrete inviscid Burgers ``u_t + (u^2/2)_x = 0`` with Lax-Friedrichs
    fluxes on a periodic grid, initialised on a smooth entropy wave. The
    nonlinear hyperbolic problem: it is where the SSP (convex-combination)
    representation of a scheme separates from schemes with the same stability
    function (tests/test_tvd.py).
``viscousBurgers``
    Semi-discrete viscous Burgers ``u_t + (u^2/2)_x = nu u_xx`` with central
    fluxes, shipped as a ``SemilinearRHS`` (``L(u) = nu u_xx`` +
    ``N(u) = -(u^2/2)_x``). The canonical semilinear benchmark for the
    structured RHS interface (NOTES S3.12, Phase 14) and the Phase 7
    accuracy/cost problem; the discrete mass is exactly conserved.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence

import torch

from .fields import BaseState, constant, integrated, get_reference_state, reference_state, tagged, update_component, update_position
from .history import StepHistory
from .protocol import BaseIntegrationSystem
from .rhs import SemilinearRHS
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


def stiff_relaxation_problem(rate: float = 10.0, forcing: str = 'tanh', sign: int = +1) -> Problem:
    """Prothero-Robinson relaxation toward a smooth target.

    ``x' = -sign*rate*(x - s(t)) + s'(t)`` with the invariant (exact) solution
    ``x(t) = s(t)`` whenever ``x(0) = s(0)``. ``sign = +1`` is the stable variant
    (stiff eigenvalue ``-rate``): explicit methods are restricted to
    ``dt < ~2/rate``, implicit ones are not. ``sign = -1`` is the classic
    *unstable* Prothero-Robinson test (eigenvalue ``+rate``): the exact solution
    is still ``s(t)``, but the explicit stability boundary is hit on the positive
    real axis instead. ``rate`` is the stiff eigenvalue magnitude and
    ``forcing`` selects the smooth target:

    - ``'tanh'``: ``s = tanh(t)``   (``s(0) = 0``)
    - ``'sin'``:  ``s = sin(t)``    (``s(0) = 0``)
    - ``'cos'``:  ``s = cos(t)``    (``s(0) = 1``)
    """
    targets = {
        'tanh': (lambda t: math.tanh(t), lambda t: 1.0 / math.cosh(t) ** 2),
        'sin': (lambda t: math.sin(t), lambda t: math.cos(t)),
        'cos': (lambda t: math.cos(t), lambda t: -math.sin(t)),
    }
    if forcing not in targets:
        raise ValueError(f"Unknown stiff_relaxation forcing {forcing!r}; choose from {sorted(targets)}")
    if sign not in (+1, -1):
        raise ValueError(f'stiff_relaxation sign must be +1 (stable) or -1 (unstable PR), got {sign!r}')
    s, s_prime = targets[forcing]
    eigenvalue = -sign * rate

    def rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        time = float(system.t)
        return ParticleUpdate(
            dxdt=eigenvalue * (state.x - s(time)) + s_prime(time),
            dudt=torch.zeros_like(state.u),
            dedt=torch.zeros_like(state.e),
        ), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=_vec(s(0.0)), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
            t=0.0)

    return Problem('stiffRelaxation',
                   f'Prothero-Robinson relaxation, x\' = {eigenvalue:g}*(x - {forcing}(t)) + d/dt {forcing}(t), '
                   f'exact x = {forcing}(t), stiff scale {rate}',
                   rhs, initial,
                   lambda t: ([s(t)], [0.0]),
                   None, autonomous=False)


def stiff_damped_oscillator_problem(omega: float = 1.0, c: float = 50.0) -> Problem:
    """Damped oscillator ``x'' = -omega^2 x - c*u`` with separable stiffness regimes.

    Two stiffness scales that explicit methods must resolve but that are
    fundamentally different:

    - *high-frequency* stiffness (large ``omega``, small ``c``): the step
      restriction is set by the oscillation, ``dt < 2/omega``;
    - *dissipative* stiffness (small ``omega``, large ``c``): the restriction is
      set by the relaxation rate, ``dt < 2/c``.

    The default parameters sit in the dissipative regime (overdamped). Closed-form
    solution for all three damping regimes, ``x(0) = 1``, ``u(0) = 0``.
    """
    gamma = c / 2.0

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        return ParticleUpdate(
            dxdt=s.u.clone(),
            dudt=-omega ** 2 * s.x - c * s.u,
            dedt=torch.zeros_like(s.e),
        ), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=_vec(1.0), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
            t=0.0)

    def exact(t):
        if c < 2.0 * omega:  # underdamped
            omega_d = math.sqrt(omega ** 2 - gamma ** 2)
            decay = math.exp(-gamma * t)
            x = decay * (math.cos(omega_d * t) + gamma / omega_d * math.sin(omega_d * t))
            u = -decay * (omega ** 2 / omega_d) * math.sin(omega_d * t)
        elif c > 2.0 * omega:  # overdamped
            r1 = (-c + math.sqrt(c ** 2 - 4.0 * omega ** 2)) / 2.0
            r2 = (-c - math.sqrt(c ** 2 - 4.0 * omega ** 2)) / 2.0
            x = (r1 * math.exp(r2 * t) - r2 * math.exp(r1 * t)) / (r1 - r2)
            u = r1 * r2 * (math.exp(r2 * t) - math.exp(r1 * t)) / (r1 - r2)
        else:  # critically damped
            x = math.exp(-gamma * t) * (1.0 + gamma * t)
            u = -gamma ** 2 * t * math.exp(-gamma * t)
        return ([x], [u])

    regime = 'underdamped' if c < 2.0 * omega else ('overdamped' if c > 2.0 * omega else 'critically damped')
    return Problem('stiffDampedOscillator',
                   f'damped oscillator, x\'\' = -{omega ** 2:g} x - {c:g} u ({regime}), x(0)=1, u(0)=0',
                   rhs, initial, exact, None, autonomous=True)


def van_der_pol_problem(mu: float = 2.0) -> Problem:
    """Van der Pol oscillator ``x'' = mu*(1 - x^2)*u - x`` with ``u = x'``.

    Autonomous and nonlinear, with no closed-form solution: the long-time attractor
    is a limit cycle of amplitude ~2 for every ``mu > 0``, and the cycle becomes
    increasingly stiff in ``u`` as ``mu`` grows (``|dudt_u| <= mu*(1 + amplitude^2)``
    sets the explicit step restriction). ``exact`` raises on purpose; use the
    limit-cycle amplitude checks (see ``tests/test_benchmarks.py``).
    """

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        return ParticleUpdate(
            dxdt=s.u.clone(),
            dudt=mu * (1.0 - s.x ** 2) * s.u - s.x,
            dedt=torch.zeros_like(s.e),
        ), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=_vec(1.0), u=_vec(0.0), e=_vec(0.0), m=_vec(1.0)),
            t=0.0)

    def exact(t):
        raise NotImplementedError(
            'van der Pol has no closed-form solution; check the limit-cycle attractor instead')

    return Problem('vanDerPol',
                   f'van der Pol, x\'\' = {mu:g}*(1 - x^2)u - x (autonomous, nonlinear, limit cycle)',
                   rhs, initial, exact, None, autonomous=True)


def robertson_problem() -> Problem:
    """Robertson chemical kinetics, the classic nonlinear stiff chemistry benchmark.

    Three species (carried in the ``x`` components), the classic Robertson
    kinetics with rate coefficients spanning ~1e9:

    - ``y1' = -0.04*y1 + 1e4*y2*y3``
    - ``y2' =  0.04*y1 - 3e7*y2^2 - 1e4*y2*y3``
    - ``y3' =  3e7*y2^2``,   ``y(0) = (1, 0, 0)``

    No closed form, but the system has exact invariants that a correct stiff
    integrator must respect: ``sum(y) = 1`` is conserved, ``y3' >= 0`` so ``y3``
    increases monotonically, all species stay non-negative, and the intermediate
    ``y2`` stays on its quasi-steady plateau (``~ sqrt(y1/3e7)``, i.e. ``O(1e-4)``).
    ``exact`` raises on purpose; use the invariant checks.
    """
    k1, k2, k3 = 0.04, 3.0e7, 1.0e4

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        y1, y2, y3 = s.x
        return ParticleUpdate(
            dxdt=torch.stack([-k1 * y1 + k3 * y2 * y3,
                              k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3,
                              k2 * y2 ** 2]),
            dudt=torch.zeros_like(s.u),
            dedt=torch.zeros_like(s.e),
        ), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=_vec(1.0, 0.0, 0.0), u=_vec(0.0, 0.0, 0.0),
                                e=_vec(0.0, 0.0, 0.0), m=_vec(1.0, 1.0, 1.0)),
            t=0.0)

    def exact(t):
        raise NotImplementedError(
            'Robertson kinetics has no closed-form solution; check the linear invariants instead')

    return Problem('robertson',
                   'Robertson kinetics, 3 species, stiffness ratio ~1e9 (autonomous, nonlinear, stiff)',
                   rhs, initial, exact, None, autonomous=True)


def diffusion_problem(n: int = 32, D: float = 1.0, L: float = 1.0) -> Problem:
    """Semi-discrete 1D diffusion ``u_t = D*u_xx`` with Dirichlet boundaries.

    ``n`` interior grid points, ``h = L/(n+1)``, second-order central differences
    with ``u(0) = u(L) = 0``. The initial condition is the sum of two eigenvectors
    of the discrete Laplacian, ``sin(pi*x) + 0.5*sin(5*pi*x)``, so the exact
    semi-discrete solution is known analytically:

    ``u_i(t) = e^{mu_1 t} sin(pi x_i) + 0.5 e^{mu_5 t} sin(5 pi x_i)``,
    ``mu_j = -(4D/h^2) sin^2(j pi h / 2)``.

    The spectral stiffness scale is ``|mu_n| ~= 4*D/h^2``: explicit methods need
    ``dt < ~h^2/(2D)`` while implicit methods do not. The state carries the field
    in ``x`` (``n`` components); ``u`` and ``e`` stay zero.
    """
    h = L / (n + 1)
    xs = [i * h for i in range(1, n + 1)]
    u0 = [math.sin(math.pi * x / L) + 0.5 * math.sin(5.0 * math.pi * x / L) for x in xs]
    eig = {j: -(4.0 * D / h ** 2) * math.sin(j * math.pi * h / (2.0 * L)) ** 2 for j in (1, 5)}

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        u = s.x
        lap = torch.empty_like(u)
        lap[0] = -2.0 * u[0] + u[1]
        lap[1:-1] = u[:-2] - 2.0 * u[1:-1] + u[2:]
        lap[-1] = u[-2] - 2.0 * u[-1]
        return ParticleUpdate(
            dxdt=D * lap / h ** 2,
            dudt=torch.zeros_like(s.u),
            dedt=torch.zeros_like(s.e),
        ), None

    def initial():
        return ParticleSystem(
            state=ParticleState(x=torch.tensor(u0, dtype=torch.float64),
                                u=torch.zeros(n, dtype=torch.float64),
                                e=torch.zeros(n, dtype=torch.float64),
                                m=torch.ones(n, dtype=torch.float64)),
            t=0.0)

    def exact(t):
        vals = [math.exp(eig[1] * t) * math.sin(math.pi * x / L)
                + 0.5 * math.exp(eig[5] * t) * math.sin(5.0 * math.pi * x / L) for x in xs]
        return (vals, [0.0] * n)

    return Problem('diffusion',
                   f'semi-discrete diffusion, n={n}, D={D}, L={L}, spectral stiffness ~{4.0 * D / h ** 2:.3g}',
                   rhs, initial, exact, None, autonomous=True)


def advection_problem(n: int = 64, c: float = 1.0, L: float = 1.0,
                      ic: str = 'mode') -> Problem:
    """Semi-discrete 1D periodic advection ``u_t + c u_x = 0``, first-order upwind.

    ``u`` is carried in ``system.state.x`` (``u`` and ``e`` are zero, ``m``
    ones) on the periodic grid ``x_i = i * L / n``. The semi-discretisation is
    ``(Au)_i = c (u_{i-1} - u_i) / h`` -- the model hyperbolic problem for the
    TVD/SSP classifier (NOTES S3.11): every Fourier mode ``e^{2 pi i m x / L}``
    is an exact eigenmode with eigenvalue ``(c/h) (e^{-2 pi i m / n} - 1)``, so
    the semi-discrete CFL scale is ``mu = c dt / h`` and the mode amplitudes
    decay like ``exp((c/h)(e^{-2 pi i m/n} - 1) t)``.

    ``ic='mode'`` starts on ``cos(2 pi x / L) + 0.5 cos(4 pi x / L)`` and the
    exact semi-discrete solution is known (the two eigenmodes above);
    ``ic='step'`` starts on a periodic square wave (+1 for ``x < L/2``, -1
    otherwise) which has no closed-form semi-discrete solution -- its exact
    solution is the initial step translated by ``c t`` with two sharp jumps, so
    the checks that matter are the total-variation ones (tests/test_tvd.py).
    """
    if ic not in ('mode', 'step'):
        raise ValueError(f"Unknown advection initial condition {ic!r}")
    h = L / n
    xs = [i * h for i in range(n)]
    if ic == 'mode':
        # (mode number, amplitude, eigenvalue) -- the real part of
        # a_m exp(2 pi i m x / L) exp(lam_m t) is a solution.
        modes = [
            (1, 1.0, (c / h) * (complex(math.cos(2 * math.pi / n),
                                        -math.sin(2 * math.pi / n)) - 1.0)),
            (2, 0.5, (c / h) * (complex(math.cos(4 * math.pi / n),
                                        -math.sin(4 * math.pi / n)) - 1.0)),
        ]
        u0 = [1.0 * math.cos(2 * math.pi * x / L) + 0.5 * math.cos(4 * math.pi * x / L)
              for x in xs]

        def exact(t):
            vals = []
            for x in xs:
                v = 0.0
                for m, a, lam in modes:
                    v += a * (cmath.exp(lam * t) * cmath.exp(2j * math.pi * m * x / L)).real
                vals.append(v)
            return (vals, [0.0] * n)
    else:
        u0 = [1.0 if x < L / 2 else -1.0 for x in xs]

        def exact(t):
            raise NotImplementedError(
                'the exact semi-discrete solution of a step initialisation '
                'has no closed form; use the total-variation checks '
                '(tests/test_tvd.py)')

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        u = s.x
        return ParticleUpdate(
            dxdt=c * (torch.roll(u, 1) - u) / h,
            dudt=torch.zeros_like(u),
            dedt=torch.zeros_like(u)), None

    def initial():
        return ParticleSystem(
            state=ParticleState(
                x=torch.tensor(u0, dtype=torch.float64),
                u=torch.zeros(n, dtype=torch.float64),
                e=torch.zeros(n, dtype=torch.float64),
                m=torch.ones(n, dtype=torch.float64)),
            t=0.0)

    return Problem(
        name=f'advection ({ic})',
        description=(f'semi-discrete 1D periodic advection, first-order upwind, '
                     f'n={n}, c={c}, CFL scale mu = c dt / h'),
        rhs=rhs,
        initial=initial,
        exact=exact,
        energy=None,
        autonomous=True)


def burgers_problem(n: int = 64, L: float = 1.0) -> Problem:
    """Semi-discrete inviscid Burgers ``u_t + (u^2/2)_x = 0``, Lax-Friedrichs.

    ``u`` is carried in ``system.state.x`` (``u`` and ``e`` are zero, ``m``
    ones) on the periodic grid ``x_i = i * L / n``. The numerical flux is the
    local Lax-Friedrichs flux
    ``F_{i+1/2} = (f(u_i) + f(u_{i+1}))/2 - (alpha/2)(u_{i+1} - u_i)`` with
    ``f(u) = u^2 / 2`` and ``alpha = max_j |u_j|`` (recomputed at every
    evaluation), giving the semi-discretisation
    ``(Au)_i = (F_{i-1/2} - F_{i+1/2}) / h``. The CFL scale is
    ``mu = alpha dt / h``.

    Initialised on the smooth entropy wave ``u(x) = -sin(2 pi x / L)``
    (range [-1, 1]): smooth, but it steepens as it evolves, so a scheme that
    is not a convex combination at the working CFL produces overshoots and
    negative dips here even when its linear stability function is identical to
    an SSP scheme's of the same order (tests/test_tvd.py).
    """
    h = L / n
    u0 = [-math.sin(2 * math.pi * i / n) for i in range(n)]

    def rhs(system, dt, **kwargs):
        s = get_reference_state(system)
        u = s.x
        f = 0.5 * u ** 2
        alpha = u.abs().max()
        f_plus = 0.5 * (f + torch.roll(f, -1)) - 0.5 * alpha * (torch.roll(u, -1) - u)
        f_minus = 0.5 * (f + torch.roll(f, 1)) - 0.5 * alpha * (u - torch.roll(u, 1))
        return ParticleUpdate(
            dxdt=(f_minus - f_plus) / h,
            dudt=torch.zeros_like(u),
            dedt=torch.zeros_like(u)), None

    def initial():
        return ParticleSystem(
            state=ParticleState(
                x=torch.tensor(u0, dtype=torch.float64),
                u=torch.zeros(n, dtype=torch.float64),
                e=torch.zeros(n, dtype=torch.float64),
                m=torch.ones(n, dtype=torch.float64)),
            t=0.0)

    def exact(t):
        raise NotImplementedError(
            'inviscid Burgers has no closed-form semi-discrete solution for '
            'this initialisation; use the total-variation checks '
            '(tests/test_tvd.py)')

    return Problem(
        name='burgers',
        description=(f'semi-discrete inviscid Burgers, Lax-Friedrichs fluxes, '
                     f'n={n}, CFL scale mu = alpha dt / h with alpha = max|u|'),
        rhs=rhs,
        initial=initial,
        exact=exact,
        energy=None,
        autonomous=True)


def viscous_burgers_problem(n: int = 64, L: float = 1.0, nu: float = 0.01) -> Problem:
    """Semi-discrete viscous Burgers ``u_t + (u^2/2)_x = nu u_xx``, central differences.

    The canonical **semilinear** benchmark (NOTES S3.12, Phase 14): the dynamics
    split cleanly into a constant-coefficient linear operator
    ``L(u) = nu u_xx`` (a cheap matrix-free action, the stiff part) and the
    nonlinear convective remainder ``N(u) = -(u^2/2)_x``. It is shipped as a
    ``SemilinearRHS`` so the split is exercised end to end, and it doubles as the
    Phase 7 accuracy/cost problem -- Rosenbrock-W and exponential integrators are
    conventionally demonstrated on exactly this equation.

    Distinct from the inviscid ``burgers_problem`` (Lax-Friedrichs, TVD/SSP
    classifier): here the flux is *central*, so the linear/semilinear split is
    clean, and the viscosity ``nu`` is what makes the linear part stiff. ``u`` is
    carried in ``system.state.x`` (``u``, ``e`` zero, ``m`` ones) on the periodic
    grid ``x_i = i * L / n``; both the convective flux and the Laplacian have zero
    periodic mean, so the discrete mass ``sum(u)`` is exactly conserved.

    No closed-form semi-discrete solution: the reference is a fine-``dt``
    integration of the combined (unsplit) dynamics, e.g.
    ``testing.run(getIntegrator('RK4'), problem, dt, T)`` with a small ``dt``
    (the stiffest decay rate is ``~4 nu / h^2``).
    """
    h = L / n
    xc = L / 2
    sigma = 0.08
    u0 = [math.exp(-((i * h - xc) ** 2) / (2 * sigma ** 2)) for i in range(n)]

    def linear(state, dt, *args, **kwargs):
        # L(u) = nu u_xx, second-order central Laplacian (zero periodic mean).
        s = get_reference_state(state)
        u = s.x
        u_xx = (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / (h * h)
        return ParticleUpdate(dxdt=nu * u_xx,
                              dudt=torch.zeros_like(u),
                              dedt=torch.zeros_like(u)), None

    def nonlinear(state, dt, *args, **kwargs):
        # N(u) = -(u^2/2)_x, second-order central flux (zero periodic mean).
        s = get_reference_state(state)
        u = s.x
        f = 0.5 * u ** 2
        return ParticleUpdate(dxdt=-(torch.roll(f, -1) - torch.roll(f, 1)) / (2 * h),
                              dudt=torch.zeros_like(u),
                              dedt=torch.zeros_like(u)), None

    rhs = SemilinearRHS(linear=linear, nonlinear=nonlinear)

    def initial():
        return ParticleSystem(
            state=ParticleState(
                x=torch.tensor(u0, dtype=torch.float64),
                u=torch.zeros(n, dtype=torch.float64),
                e=torch.zeros(n, dtype=torch.float64),
                m=torch.ones(n, dtype=torch.float64)),
            t=0.0)

    def exact(t):
        raise NotImplementedError(
            'viscous Burgers has no closed-form semi-discrete solution for this '
            'initialisation; use a fine-dt integration of the combined dynamics '
            '(see the docstring)')

    return Problem(
        name='viscousBurgers',
        description=(f'semi-discrete viscous Burgers, central fluxes, n={n}, nu={nu}; '
                     f'semilinear split L(u)=nu u_xx + N(u)=-(u^2/2)_x, stiff rate ~4 nu/h^2'),
        rhs=rhs,
        initial=initial,
        exact=exact,
        energy=None,
        autonomous=True)


PROBLEMS = {
    'oscillator': oscillator_problem,
    'forced': forced_problem,
    'damped': damped_problem,
    'kepler': kepler_problem,
    'stiffRelaxation': stiff_relaxation_problem,
    'stiffDampedOscillator': stiff_damped_oscillator_problem,
    'vanDerPol': van_der_pol_problem,
    'robertson': robertson_problem,
    'diffusion': diffusion_problem,
    'advection': advection_problem,
    'burgers': burgers_problem,
    'viscousBurgers': viscous_burgers_problem,
}


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #

def run(scheme, problem: Problem, dt: float, T: float, *,
        reuse: bool = False, history: bool = False, history_length: int = 4,
        **scheme_kwargs) -> ParticleSystem:
    """Integrate ``problem`` from t=0 to t=T in fixed steps of ``dt``.

    ``history=True`` threads a ``StepHistory`` through the run instead of (or, if
    ``reuse`` is also set, alongside) the single-entry ``priorStep``. A one-step
    scheme ignores a ``history`` kwarg it doesn't recognise -- ``RungeKuttaB`` pops it
    like ``priorStep`` and only acts on it if given one -- so this is safe to pass for
    every registered scheme today; it exists so a multistep scheme (NOTES.md S3
    Phase 1) or a step-reuse test that wants more than one entry of lookback can
    exercise this without every other test's ``run`` call changing shape first
    (NOTES.md S5).

    ``**scheme_kwargs`` are forwarded to the scheme on every step (and are therefore
    ignored by schemes that do not read them). They exist for schemes that take a
    per-step parameter the run loop does not own -- the RKC/RKL family's stage count
    ``s=`` (or ``lambda_max=``) is the first consumer (NOTES.md S3.13, Phase 12).
    """
    system = problem.initial()
    prior = None
    step_history = StepHistory(maxlen=history_length) if history else None
    for _ in range(int(round(T / dt))):
        kwargs = dict(scheme_kwargs)
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
                reuse: bool = False, history: bool = False,
                **scheme_kwargs) -> float:
    """L1 error in (x, u) at time T against the analytic solution.

    ``**scheme_kwargs`` are forwarded to :func:`run` (and hence to the scheme); see
    that function for the RKC/RKL stage-count use.
    """
    system = run(scheme, problem, dt, T, reuse=reuse, history=history, **scheme_kwargs)
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
                *, reuse: bool = False, **scheme_kwargs):
    """(measured order, errors) for ``scheme`` on ``problem`` over ``dts``.

    ``**scheme_kwargs`` are forwarded to :func:`final_error` (and hence the scheme).
    """
    errors = [final_error(scheme, problem, dt, T, reuse=reuse, **scheme_kwargs) for dt in dts]
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
