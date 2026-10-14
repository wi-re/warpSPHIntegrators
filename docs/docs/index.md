---
sidebar_label: Overview
---

# warpSPHIntegrators — scheme wiki

Differentiable ODE integration for PyTorch (and, eventually, NVIDIA Warp):
a typed state/update system and matrix-free implicit machinery (JFNK + GMRES).

[catalog:summary:start]: #

**140 registered schemes across 11 families**, with orders through 14 and 47 estimate emitters.

[catalog:summary:end]: #

The three documents, in order of depth:

| Document | What it is |
|---|---|
| `README.md` (repo root) | The short reference: quick start, the scheme table, API, benchmarks. |
| This wiki | One page per driver family: the equations, the Butcher tableaus, the stability, the citations, and the test files that pin each claim. |
| [Numerical records](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/research/integrators/notes-index.md) | Dated derivations, stability sweeps, benchmark numbers and design decisions. |

Interactive walkthroughs: `examples/notebooks/integrators.ipynb` (explicit methods, convergence
order, stability regions, Hamiltonian energy drift), `examples/notebooks/viscous_burgers_demo.ipynb`
(semilinear benchmarks, the nonlinear-solver cost model),
`examples/integrations/jfnk_wave_equation.ipynb` (JFNK on the wave equation). The open-items-only
designs are indexed in [developer documentation](/docs/development);
`REPOSITORY_PLAN.md` is the current task list.

## Quick start

Copy this complete CPU script or run `python examples/harmonic_oscillator.py`
in a checkout. The RHS returns `(update, aux)` and the step returns a frozen
`IntegrationResult` dataclass.

[example:include examples/harmonic_oscillator.py]: #
```python
"""Complete CPU example: python examples/harmonic_oscillator.py."""

from dataclasses import dataclass

import torch

from warpSPHIntegrators import (
    BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec,
    constant, getIntegrator, get_reference_state, integrated, reference_state,
    tagged, update_component, update_position,
)


@dataclass
class OscillatorState(BaseState):
    position: torch.Tensor = integrated('dx_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    mass: torch.Tensor = constant()
    spring_constant: torch.Tensor = constant()


@dataclass
class OscillatorUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))


@dataclass
class OscillatorSystem(BaseIntegrationSystem):
    state: OscillatorState = reference_state()
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        return OscillatorSystem(get_reference_state(self).initializeNewState(), self.t)

    def apply_position_update(self, update, spec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_state_update(self, update, spec, **kwargs):
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        return self


def rhs(system, dt, **kwargs):
    state = get_reference_state(system)
    return OscillatorUpdate(
        dx_dt=state.velocity,
        dv_dt=-(state.spring_constant / state.mass) * state.position,
    ), None


def run_example():
    system = OscillatorSystem(OscillatorState(
        position=torch.tensor([1.0], dtype=torch.float64),
        velocity=torch.tensor([0.0], dtype=torch.float64),
        mass=torch.tensor([1.0], dtype=torch.float64),
        spring_constant=torch.tensor([4.0], dtype=torch.float64),
    ))
    scheme = getIntegrator('RK4')
    for _ in range(100):
        result = scheme(system, dt=0.01, f=rhs)
        system = result.state
    return result


if __name__ == '__main__':
    result = run_example()
    state = get_reference_state(result.state)
    print(f't={result.state.t:.2f}, x={state.position.item():.6f}, '
          f'v={state.velocity.item():.6f}, stages={len(result.stages)}')
```

The state/update protocol (field behaviors `integrated` / `constant` /
`copied` / `ephemeral`, update specs) is in the README's *Quick Start* and
*API Reference* sections.

## The 11 families

Every scheme in the registry belongs to exactly one family enum (see the
`Family enums` section of `README.md` for the lookup API):

[catalog:families:start]: #

| Family | Enum | Schemes | Page |
|---|---|---:|---|
| Explicit RK | `ExplicitRK` | 63 | [explicit-rk](/docs/families/explicit-rk) |
| Symplectic | `Symplectic` | 6 | [symplectic](/docs/families/symplectic) |
| Relaxed Chebyshev | `RelaxedChebyshev` | 3 | [relaxed-chebyshev](/docs/families/relaxed-chebyshev) |
| Multistep explicit | `MultistepExplicit` | 10 | [multistep-explicit](/docs/families/multistep-explicit) |
| DIRK | `DIRK` | 36 | [dirk](/docs/families/dirk) |
| Coupled RK | `CoupledRK` | 4 | [coupled-rk](/docs/families/coupled-rk) |
| Multistep implicit | `MultistepImplicit` | 8 | [multistep-implicit](/docs/families/multistep-implicit) |
| IMEX | `IMEX` | 6 | [imex](/docs/families/imex) |
| Linearly implicit | `LinearlyImplicit` | 1 | [rosenbrock](/docs/families/rosenbrock) |
| Exponential | `Exponential` | 2 | [exponential](/docs/families/exponential) |
| Newmark | `Newmark` | 1 | [newmark](/docs/families/newmark) |

[catalog:families:end]: #

The machinery underneath the implicit families:

- [Nonlinear solver, preconditioning, adaptive step control](/docs/solver)

## Reading a scheme page

Each family page follows the same shape: what the class is and when to use it,
then per scheme the defining equation or Butcher tableau, the stability, the
cost per step, the citation, and the test file(s) that pin the claims. Numbers
quoted as *measured* come from those tests or the benchmark scripts under
`scripts/`; derivations live in NOTES.md (linked per scheme).
