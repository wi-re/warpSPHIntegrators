# warpSPHIntegrators — scheme wiki

Differentiable ODE integration for PyTorch (and, eventually, NVIDIA Warp):
a typed state/update system, **64 registered time-stepping schemes across 11
families**, and a matrix-free implicit machinery (JFNK + GMRES) that keeps the
implicit schemes differentiable through the solves.

The three documents, in order of depth:

| Document | What it is |
|---|---|
| `README.md` (repo root) | The short reference: quick start, the scheme table, API, benchmarks. |
| This wiki | One page per driver family: the equations, the Butcher tableaus, the stability, the citations, and the test files that pin each claim. |
| `NOTES.md` (repo root) | The detailed measurement record: derivations, stability-function sweeps, benchmark numbers, design decisions. |

Interactive walkthroughs: `integrators.ipynb` (explicit methods, convergence
order, stability regions, Hamiltonian energy drift), `viscous_burgers_demo.ipynb`
(semilinear benchmarks, the nonlinear-solver cost model),
`jfnk_wave_equation.ipynb` (JFNK on the wave equation). The open-items-only
roadmap is `IMPLICIT_ROADMAP.md` (repo root).

## Quick start

```python
from warpSPHIntegrators import BaseState, integrated, constant, explicit_step, getIntegrator

class SimpleSystem(BaseState):
    position: float = integrated('velocity', tags=('position',))
    velocity: float = integrated('dudt', tags=('velocity',))

system = SimpleSystem(position=0.0, velocity=1.0)

def rhs(t, x):
    return x.velocity, -x.position

integrator = getIntegrator('RK4')

for _ in range(10):
    result = integrator(system, 0.1, rhs)
    system = result.state
    print(system.position, system.velocity, result.rhs_evaluations)
```

The state/update protocol (field behaviors `integrated` / `constant` /
`copied` / `ephemeral`, update specs) is in the README's *Quick Start* and
*API Reference* sections.

## The 11 families

Every scheme in the registry belongs to exactly one family enum (see the
`Family enums` section of `README.md` for the lookup API):

| Family | Enum | Schemes | One line | Page |
|---|---|---|---|---|
| Explicit RK | `ExplicitRK` | 21 | Butcher-tableau one-step explicit RK, classical/SSP/TVD/embedded | [explicit-rk](families/explicit-rk.md) |
| Symplectic | `Symplectic` | 6 | Geometric position/velocity splitting for second-order ODEs | [symplectic](families/symplectic.md) |
| Relaxed Chebyshev | `RelaxedChebyshev` | 3 | Matrix-free super-timestepping, real-axis stability $O(s^2)$ in stages | [relaxed-chebyshev](families/relaxed-chebyshev.md) |
| Multistep explicit | `MultistepExplicit` | 7 | Adams-Bashforth and PECE Adams-Bashforth-Moulton | [multistep-explicit](families/multistep-explicit.md) |
| DIRK | `DIRK` | 7 | Diagonally implicit RK, one JFNK solve per implicit stage | [dirk](families/dirk.md) |
| Coupled RK | `CoupledRK` | 2 | Gauss-Legendre 2, Radau IIA s=2 — one $s\times s$ JFNK block solve | [coupled-rk](families/coupled-rk.md) |
| Multistep implicit | `MultistepImplicit` | 8 | BDF1-5 and JFNK-corrected Adams-Moulton | [multistep-implicit](families/multistep-implicit.md) |
| IMEX | `IMEX` | 6 | Additive explicit/implicit splits (IMEX Euler, ARK, SBDF/CNAB) | [imex](families/imex.md) |
| Linearly implicit | `LinearlyImplicit` | 1 | Rosenbrock-W (ROS3P): frozen-operator GMRES per stage | [rosenbrock](families/rosenbrock.md) |
| Exponential | `Exponential` | 2 | ETD2RK, EXPRB32 — matrix-free `phi_k` Krylov | [exponential](families/exponential.md) |
| Newmark | `Newmark` | 1 | Newmark beta/gamma for second-order dynamics | [newmark](families/newmark.md) |

The machinery underneath the implicit families:

- [Nonlinear solver, preconditioning, adaptive step control](solver.md)

## Reading a scheme page

Each family page follows the same shape: what the class is and when to use it,
then per scheme the defining equation or Butcher tableau, the stability, the
cost per step, the citation, and the test file(s) that pin the claims. Numbers
quoted as *measured* come from those tests or the benchmark scripts under
`scripts/`; derivations live in NOTES.md (linked per scheme).

```{toctree}
:maxdepth: 1
:caption: Families

families/explicit-rk
families/symplectic
families/relaxed-chebyshev
families/multistep-explicit
families/dirk
families/coupled-rk
families/multistep-implicit
families/imex
families/rosenbrock
families/exponential
families/newmark
solver
```
