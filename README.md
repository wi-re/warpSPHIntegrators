# warpSPHIntegrators — Differentiable ODE Integration with PyTorch

A flexible, fully differentiable numerical ODE integration library for PyTorch. Implements multiple integration schemes with support for complex state management, custom field behavior, and both typed and legacy APIs.

> **Status.** Today this is a pure-PyTorch library; there is no NVIDIA Warp code in it
> yet. The name describes where it is going, not what it does. State cloning is now
> backend-dispatched, so `wp.array` fields are copied correctly and a Warp state is safe
> to hold — but a Warp *backend* still needs a preallocated-buffer path instead of the
> allocate-per-stage design, and a decision between torch autograd and `wp.Tape` for
> gradients. See `NOTES.md`.

## Overview

This library provides a set of time integration schemes for solving systems of ordinary differential equations (ODEs) where states may be complex objects with multiple field types and integration behaviors. All computations are differentiable through PyTorch, making the library suitable for physics-informed machine learning, neural ODEs, and scientific computing.

The integrators never refer to `position` / `velocity` / `density` by name. They only say
"advance the component tagged X by `c·dt` times the derivative tagged Y, optionally
blended with a reference state", and your system object decides what that means.

### Key Features

- **Multiple Integration Schemes**: Runge-Kutta up to 5th order, embedded FSAL pairs (Bogacki–Shampine, Dormand–Prince, Cash–Karp), TVD-RK2/3, symplectic Verlet, Forest–Ruth high-order, and Euler methods; stabilized explicit super-timestepping for parabolic stiffness (RKC1/RKC2/RKL2, no nonlinear solver); diagonally implicit (Backward Euler, Implicit Midpoint, Trapezoidal, SDIRK2, TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA, Newmark) via a pluggable `NonlinearSolver`; explicit multistep (Adams-Bashforth 2–5, Adams-Bashforth-Moulton 2–4); implicit multistep (BDF1–BDF5, fully implicit Adams-Moulton 2–4); additive (IMEX) RK (ARK3(2)4L[2]SA, ARK4(3)6L[2]SA) and IMEX Euler, both through an explicit/implicit RHS split
- **Flexible State Management**: Custom state objects with metadata-driven field behavior (integrated, constant, copied, ephemeral, custom)
- **Type-Safe Protocol**: Structural typing for integration systems with clear separation of concerns
- **Fully Differentiable**: All operations preserve gradient flow for end-to-end learning
- **Extensible**: Easy to add new schemes or state types

## Installation

```bash
pip install warpSPHIntegrators
```

The distribution and the import name are both `warpSPHIntegrators`. The import name
used to be `integrators`; `import integrators` no longer works.

For a checkout, with the test dependencies:

```bash
pip install -e ".[test]"
pytest
```

## Quick Start

### 1. Define Your State

Create a state class that inherits from `BaseState` and annotate fields with their integration behavior:

```python
from dataclasses import dataclass
import torch
from warpSPHIntegrators import BaseState, integrated, constant

@dataclass
class MyState(BaseState):
    # Fields that evolve over time
    position: torch.Tensor = integrated('dx_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    
    # Constant parameters (don't change during integration)
    mass: torch.Tensor = constant(tags=('mass',))
    spring_constant: torch.Tensor = constant(tags=('spring_constant',))
```

**Field Behavior Options:**

| Behavior | `initializeNewState()` | End of step | For |
|---|---|---|---|
| `integrated('dfield', ...)` | cloned | advanced by the scheme | position, velocity, energy |
| `constant(...)` | cloned | unchanged | mass, material parameters |
| `copied(...)` | nulled | **taken from the last stage** | quantities recomputed per stage: summation density, pressure, smoothing length |
| `ephemeral(...)` | nulled | stays null | stage-local scratch, neighbour lists |
| `custom(...)` | nulled | your responsibility | anything the generic paths must not touch |

A field with no behavior declared is treated as `constant`. That is the conservative
default: it survives both `clone()` and `initializeNewState()`, so a forgotten decorator
cannot silently turn a tensor into `None` halfway through a step.

**Field value types.** Cloning is dispatched on the value's type, not hard-coded to
`torch.Tensor`. Out of the box: torch tensors, `wp.array` (detected without importing
warp), and lists / tuples / dicts recursed into. Immutable scalars are shared, which is
safe. Anything else is shared **by reference** and warns once — that would mean every
stage writing into the same object — so register a handler for it:

```python
from warpSPHIntegrators import register_clone_handler

register_clone_handler(
    'mylib.Buffer',
    matches=lambda v: isinstance(v, mylib.Buffer),
    clone=lambda v, detach: v.copy(),
    to_device=lambda v, device: v.to(device),   # optional, drives BaseState.to()
)
```

### 2. Define Your Update Structure

Create a dataclass that holds all derivatives and auxiliary values:

```python
from dataclasses import dataclass
from warpSPHIntegrators import tagged

@dataclass
class MyUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))
```

### 3. Implement the Integration System Protocol

Create an `IntegrationSystem` that knows how to apply updates to your state:

```python
from warpSPHIntegrators import BaseIntegrationSystem, IntegrationSystem, PositionUpdateSpec, ComponentUpdateSpec
from dataclasses import dataclass

@dataclass
class MySystem(BaseIntegrationSystem):
    state: MyState = reference_state(tags=('my_state',))
    t: float = 0.0  # Current time
    
    def initializeNewState(self, *args, **kwargs):
        """Create a fresh state copy for the next integration step."""
        current_state = get_reference_state(self)
        return MySystem(
            state=current_state.initializeNewState(),
            t=self.t
        )
    
    def apply_position_update(self, update: MyUpdate, spec: PositionUpdateSpec, **kwargs):
        """Apply a position update using velocity (semi-implicit or explicit)."""
        # Implementation handles velocity drift and derivative steps
        return update_position(self, update, spec, 
                             'position', 'position_derivative',
                             'velocity', 'velocity_derivative')
    
    def apply_velocity_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        """Apply a velocity update."""
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    
    def apply_state_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        """Apply all state updates in sequence."""
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        return self
```

**Protocol Methods:**
- `initializeNewState()` — Create a fresh state for the next step
- `apply_position_update()` — Position update (often semi-implicit with velocity)
- `apply_velocity_update()` — Velocity update
- `apply_quantity_update()` — Other field updates
- `apply_state_update()` — General state update (calls the above in sequence)

### 4. Define Your Right-Hand Side (RHS) Function

The RHS function receives the current system and returns an update structure plus optional auxiliary values:

```python
def my_rhs(system: MySystem, dt: float, verbose: bool = False) -> tuple:
    """
    Compute derivatives and auxiliary values.
    
    Returns:
        (update: MyUpdate, aux: tuple)
    """
    state = get_reference_state(system)
    
    # Compute derivatives
    dx_dt = state.velocity
    dv_dt = -(state.spring_constant / state.mass) * state.position
    
    # Auxiliary values (optional, for logging/analysis)
    kinetic_energy = 0.5 * state.mass * state.velocity**2
    potential_energy = 0.5 * state.spring_constant * state.position**2
    
    return (
        MyUpdate(dx_dt=dx_dt, dv_dt=dv_dt),
        (kinetic_energy, potential_energy)
    )
```

This plain callable is all most schemes need. Split methods take a
**structured RHS** instead: `IMEXRHS(explicit=, implicit=)` for the additive
split `f = f_E + f_I` (IMEX Euler, ARK3/ARK4), or `SemilinearRHS(linear=,
nonlinear=)` for the semilinear split `f = L·y + N`. Both remain callable as
the combined `f`, so a scheme that only needs the full right-hand side treats
the split as fully implicit; a scheme that needs a split the RHS does not
declare fails before the solve with an error naming the missing capability.
See [NOTES.md §3.12](NOTES.md#312-phase-14-the-structured-rhs-interface--done-2026-09-10).

### 5. Choose an Integration Scheme and Integrate

```python
from warpSPHIntegrators import getIntegrator, IntegrationSchemeType

# Create initial system
initial_state = MyState(
    position=torch.tensor([1.0]),
    velocity=torch.tensor([0.0]),
    mass=torch.tensor([1.0]),
    spring_constant=torch.tensor([5.0])
)
system = MySystem(state=initial_state, t=0.0)

# Get an integrator
integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

# Integrate one step
dt = 0.01
result = integrator.function(
    system,
    dt=dt,
    f=my_rhs,
    verbose=False
)

# Access results
next_system = result.state  # IntegrationResult.state
last_stage = result.stages[-1]  # Last StageResult
aux_values = last_stage.aux  # Auxiliary output from RHS
```

## Return Type: `IntegrationResult`

All integrator functions return an `IntegrationResult` named tuple:

```python
from warpSPHIntegrators import IntegrationResult, StageResult

result: IntegrationResult = integrator.function(...)

# Fields:
result.state        # The new system state after integration
result.stages       # List[StageResult] — all intermediate stages

# Each stage:
stage: StageResult = result.stages[-1]
stage.aux           # Auxiliary values returned from RHS
stage.update        # The k-value (derivative) used at this stage
```

This replaces the older tuple return format and provides a clear, self-documenting API.

## Available Integration Schemes

`Order` throughout is the **global** convergence order, verified by `tests/test_convergence.py`
on an autonomous, a time-dependent, and a nonlinear problem. `Reuse` is the order retained
when the previous step's last stage is fed back in as `k0` — see
[First-stage reuse](#first-stage-reuse-priorstep).

### Explicit Runge-Kutta Methods

| Scheme | Order | Stages | Reuse | Use Case |
|--------|-------|--------|-------|----------|
| **Forward Euler** / **Explicit Euler** | 1 | 1 | refused | Quick, low-accuracy tests; baseline |
| **Midpoint** / **RK2** / **EPEC** | 2 | 2 | 2 ✓ | Moderate accuracy in 2 evaluations; reuse is free |
| **Heun 2nd** / **EPEC Modified** | 2 | 2 | 2 ✓ | As above |
| **Ralston 2nd** | 2 | 2 | 1 | Minimises the local error constant |
| **RK3**, **Heun 3rd**, **Ralston 3rd**, **Wray 3rd** | 3 | 3 | 1–2 | Third order in 3 evaluations |
| **SSP-RK3** | 3 | 3 | 1 | Conservation laws; strong-stability-preserving |
| **RK4 (Classic)** | 4 | 4 | 3 | High accuracy; the usual default |
| **RK4 (alternative)** | 4 | 4 | 2 | 3/8 rule |
| **Nyström 5th** | 5 | 6 | 1 | Fifth order without step control |

### Embedded pairs (error estimate + adaptive `dt`)

These return a step-size-control estimate in `IntegrationResult.error`. The first two are
**FSAL**, so first-stage reuse is exact and costs one evaluation less per step.

| Scheme | Order | Stages | Reuse | Use Case |
|--------|-------|--------|-------|----------|
| **Bogacki–Shampine 3(2)** | 3 | 4 (3 under reuse) | 3 ✓ FSAL | Cheapest useful pair; `scipy`'s `RK23` |
| **Dormand–Prince 5(4)** | 5 | 7 (6 under reuse) | 5 ✓ FSAL | The workhorse; `scipy`'s `RK45`, MATLAB's `ode45` |
| **Cash–Karp 5(4)** | 5 | 6 | 1 | Well-conditioned pair when DP5 is more than needed |

### Symplectic Methods

Energy error stays inside an `O(dt^p)` band however long the run, rather than drifting
secularly — this is what `dissipation=False` on the scheme records, and
`tests/test_hamiltonian.py` holds them to it.

**These schemes assume a separable Hamiltonian**, i.e. a force depending on position
alone. With a velocity-dependent force — artificial viscosity, drag, any real SPH
momentum equation — Leap Frog, Velocity Verlet, PEFRL and VEFRL all drop to **first
order**. Symplectic Euler does not; it keeps second order either way.

| Scheme | Order | Reuse | Use Case |
|--------|-------|-------|----------|
| **Semi-Implicit Euler** | 1 | refused | Cheapest symplectic scheme |
| **Symplectic Euler** | 2 | 1 | Kick-drift-kick; second order even for velocity-dependent forces |
| **Velocity Verlet** | 2 (1 if `f` sees velocity) | 2 ✓ | Standard Verlet; reuse is free (FSAL property) |
| **Leap-Frog** | 2 (1 if `f` sees velocity) | 1 | Synchronised form |
| **PEFRL** | 4 (1 if `f` sees velocity) | refused | High-order Forest–Ruth, position-first |
| **VEFRL** | 4 (1 if `f` sees velocity) | refused | Velocity-first variant |

### TVD and Conservative Schemes

| Scheme | Order | Reuse | Use Case |
|--------|-------|-------|----------|
| **TVD-RK2** | 2 | refused | Conservation laws with the TVD property |
| **TVD-RK3** | 3 | refused | Shu–Osher form of SSP-RK3 |

### Stabilized Explicit (RKC / RKL)

Chebyshev- and Legendre-based super-timestepping for **parabolic stiffness**
(diffusion, SPH viscosity). No tableau and no nonlinear solver: a fixed coefficient
recursion, and a stage count `s` chosen *per step* so the real-axis stability range
`K(s)` covers `dt·|λ_max|`. The driver therefore requires `s=` *or* `lambda_max=`
(omitting both is a `ValueError`); `lambda_max=` runs `s = stage_count(dt·|λ_max|,
family)` internally. First-stage reuse is refused — `s` is a per-step quantity. The
family is built for real (negative) eigenvalues, so the hyperbolic TVD/SSP question
does not apply to it. On the semi-discrete diffusion benchmark it reaches a fixed
error with fewer total RHS evaluations than BE / BDF2 / TR-BDF2 run with the default
JFNK — and BDF2's practical stability there turns out to be solver-limited rather
than method-limited ([`scripts/rkc_benchmark.py`](scripts/rkc_benchmark.py),
![RKC / RKL vs the implicit baselines](images/rkc_benchmark.png),
[NOTES §3.13](NOTES.md#313-phase-12-rkc--rkl--stabilized-explicit-super-timestepping--done-2026-09-11)):

| Scheme | Order | Stability range (real axis) | Reuse | Use Case |
|--------|-------|-----------------------------|-------|----------|
| **RKC1** | 1 | K = 2s² | refused | Fewest stages per unit stiffness — the cheapest option |
| **RKC2** | 2 | K = 2(s²−1)/3 | refused | Order 2; the wider of the two order-2 ranges |
| **RKL2** | 2 | K = (s²+s−2)/2 | refused | Legendre (relaxed Lobatto) variant; Meyer, Balsara & Aslam 2014 |

```python
from warpSPHIntegrators import RKC2

system = RKC2(system, dt=dt, f=rhs, s=s)            # explicit stage count
system = RKC2(system, dt=dt, f=rhs, lambda_max=L)   # or let stage_count pick s
```

### Diagonally Implicit (DIRK) Methods

Each stage's implicit diagonal term is closed with a `NonlinearSolver`. The default is
`JFNKSolver`, which uses an inexact Newton solve with matrix-free GMRES and finite-difference
Jacobian-vector products, so it can converge where Picard iteration diverges. For a fixed-depth,
CUDA-graph-capturable non-stiff solve, explicitly pass `FixedPointSolver(iterations=...)` or the
damped `RelaxedFixedPointSolver(relaxation=..., iterations=...)`. The stiffly accurate schemes
with an explicit first stage — Trapezoidal, TR-BDF2, and the two ESDIRKs — implement lossless
`priorStep` reuse (the previous step's last stage *is* `f(t^{n+1}, y^{n+1})`); the rest reject it
with the standard warning. See [NOTES.md §3.6](NOTES.md#36-valid-schemes-and-what-each-costs) for
the derivation and the remaining scheme work:

| Scheme | Order | Stability | Use Case |
|--------|-------|-----------|----------|
| **Backward Euler (implicit)** | 1 | L | Reference/fallback; heavily damping |
| **Implicit Midpoint** | 2 | A, symplectic | JFNK converges the stage equation by default |
| **Trapezoidal (Crank-Nicolson)** | 2 | A, not L | Classic pair with BDF2; symmetric |
| **SDIRK2** | 2 | L | L-stability for real stiffness |
| **TR-BDF2** | 2 | L, stiffly accurate | Three-stage L-stable DIRK; trapezoidal substep plus a BDF2 endpoint solve |
| **ESDIRK3(2)4L[2]SA** | 3 | L, stiffly accurate | Kennedy–Carpenter 4-stage ESDIRK with explicit first stage; embedded (3, 2) pair |
| **ESDIRK4(3)6L[2]SA** | 4 | L, stiffly accurate | Kennedy–Carpenter 6-stage ESDIRK with explicit first stage; embedded (4, 3) pair |
| **Newmark** | 2 | A | Second-order structural dynamics update; JFNK closure through the repo's implicit solver API |

```python
from warpSPHIntegrators import getIntegrator

scheme = getIntegrator('Newmark')
result = scheme(system, dt=dt, f=rhs, beta=0.25, gamma=0.5)
```

**Implicit Midpoint is registered as non-dissipative because JFNK converges its stage equation by
default.** Its textbook symplectic energy bound (drift near machine precision and flat with `T`)
depends on that nonlinear convergence. An explicit low-cost Picard override is still available:

```python
from warpSPHIntegrators import FixedPointSolver, getIntegrator

scheme = getIntegrator('Implicit Midpoint')
result = scheme(system, dt=dt, f=rhs, solver=FixedPointSolver(iterations=16))
```

Use this override only when its fixed cost matters more than the converged method's qualitative
energy behaviour.

**A fixed-count Picard solve is not a stiff solver at any iteration count**: L-stability is a
property of the exact method, not of a truncated iterate, and a stiff problem (`dt·omega ≳ 1`) makes
a low-iteration Picard solve wrong, and more iterations of it explosively wrong, regardless of which
tableau you picked. The default JFNK path resolves this with a matrix-free Newton correction; relaxed
Picard can only improve the non-stiff regime, not replace Newton for stiff systems.

**JFNK accepts an optional operator-based preconditioner.** Pass
`JFNKSolver(preconditioner=..., preconditioning='right'|'left', preconditioner_context=...)`
(or the same keys through a driver's `solver_opts=`). The callable has the 3-arg contract
`preconditioner(v, state, context) -> vector`: it should approximate the inverse of the
stage Jacobian at the current Newton iterate `state`, with `context` carrying
`{'state': state, **preconditioner_context}` so it can read anything fixed for the step
(`dt`, an operator, ...). Only operator applications — never a dense matrix. Right
preconditioning (the default) solves `A·M·z = b` and returns `x = M·z`; left solves
`M·A·x = M·b`. Without a preconditioner the solve is bit-for-bit what it was before the
hook existed, and `identity_preconditioner` reproduces that in both modes.
`diagonal_preconditioner(inv_diag)` — a fixed tensor or a state-dependent callable, the
`1/(1 + dt*damping)` relaxation shape — is the ready-made example. On a variable-coefficient
stiff relaxation the per-particle diagonal holds GMRES to a flat 9–13 iterations through
`n = 1024` where the unpreconditioned solve grows to ~170
([`scripts/jfnk_preconditioner_benchmark.py`](scripts/jfnk_preconditioner_benchmark.py),
![JFNK preconditioner size sweep](images/jfnk_preconditioner_benchmark.png)), and on the
warpSPH wave-equation stage the block-lower-triangular Laplacian factor cuts the stage solve
41 → 17 (`matvec='fd'`) / 33 → 7 (`matvec='jvp'`) iterations. The full
walkthrough — solver-interface experiments 1–4 on the 1-D standing wave plus
experiments 5–6 showing this preconditioner setup at the `gmres` level,
end-to-end through `JFNKSolver`, and on the registered 2-D
`waveEquationCase` — is [jfnk_wave_equation.ipynb](jfnk_wave_equation.ipynb).

### Explicit Multistep (Adams-Bashforth / Adams-Bashforth-Moulton)

**These require `history=` to be threaded across calls to get their claimed cost.** Unlike every
other scheme here, where `history=`/`priorStep=` are opt-in bookkeeping, a multistep scheme's past
derivatives only exist if you carry `IntegrationResult.history` forward yourself:

```python
from warpSPHIntegrators import getIntegrator, StepHistory

scheme = getIntegrator('Adams-Bashforth 4')
history = StepHistory(maxlen=3)   # order - 1
for _ in range(n_steps):
    result = scheme(system, dt=dt, f=rhs, history=history)
    system, history = result.state, result.history
```

or, in the test harness, `testing.run(scheme, problem, dt, T, history=True)`. Forgetting `history=`
does not silently corrupt the trajectory: with fewer than `order - 1` past derivatives available
these schemes bootstrap by running Dormand–Prince 5(4) instead, which is *more* accurate than any of
them, not less — so a caller who never threads `history=` transparently keeps paying Dormand–Prince's
cost every step rather than getting a wrong answer.

| Scheme | Order | Evaluations/step | History needed | Use Case |
|--------|-------|-------------------|-----------------|----------|
| **Adams-Bashforth 2–5** | 2–5 | **1** | order − 1 | One force evaluation per step regardless of order — the real prize of multistep |

### Implicit Multistep (BDF)

`BDF1`–`BDF5` use the same matrix-free JFNK closure as the DIRK methods. BDF2 and
above need previous-state snapshots, so thread `IntegrationResult.history` between
steps (`maxlen=order - 1`); without enough history they safely use the repository's
Dormand-Prince 5(4) starter rather than silently dropping to first order. The
standard BDF formula evaluates the right-hand side only at the new grid time (the
history carries state values), so it keeps its full order on non-autonomous
problems.

```python
from warpSPHIntegrators import StepHistory, getIntegrator

scheme = getIntegrator('BDF4')
history = StepHistory(maxlen=3)
for _ in range(n_steps):
    result = scheme(system, dt=dt, f=rhs, history=history)
    system, history = result.state, result.history
```

| Scheme | Order | Stability | Use Case |
|--------|-------|-----------|----------|
| **BDF1** | 1 | L | Backward-Euler form for strongly damped stiff modes |
| **BDF2** | 2 | A | General stiff integration when a one-step history is acceptable |
| **BDF3** | 3 | A(α) 86.03° | Third-order stiff integration; sectorial (not full A-) stability |
| **BDF4** | 4 | A(α) 73.35° | Fourth-order stiff integration; the whole negative real axis stays in the region |
| **BDF5** | 5 | A(α) 51.84° | Highest-order stiff integration here; stiff *oscillatory* modes outside its 51.84° cone are not damped |

### Fully Implicit Adams-Moulton

`Adams-Moulton 2 (implicit)`–`Adams-Moulton 4 (implicit)` are the Adams-Moulton
corrector formulas solved *to convergence* for the unknown endpoint derivative with
the same JFNK interface — as distinct from the Adams-Bashforth-Moulton (PECE)
schemes above, which apply one uniterated correction. The history carries the past
*derivatives* (the `update` of each entry) instead of BDF's state snapshots;
`maxlen=order - 1`. A matching Adams-Bashforth prediction is the default initial
guess (`predictor=False` starts the solve from the known part instead). On a stiff
nonlinear relaxation the iterated corrector beats the same-order PECE scheme by an
order of magnitude (`tests/test_am.py`). AM2 is the trapezoidal rule and inherits
its A-stability; AM3/AM4 have only a bounded stability region, so a stiff problem
should be checked against it (or use a BDF scheme).

```python
from warpSPHIntegrators import StepHistory, getIntegrator

scheme = getIntegrator('Adams-Moulton 3 (implicit)')
history = StepHistory(maxlen=2)
for _ in range(n_steps):
    result = scheme(system, dt=dt, f=rhs, history=history)
    system, history = result.state, result.history
```

| Scheme | Order | Stability | Use Case |
|--------|-------|-----------|----------|
| **Adams-Moulton 2 (implicit)** | 2 | A | Implicit trapezoidal rule; A-stable (bounded for any stiff scale, but weakly damped) |
| **Adams-Moulton 3 (implicit)** | 3 | bounded | Third-order implicit multistep at one nonlinear solve per step |
| **Adams-Moulton 4 (implicit)** | 4 | bounded | Fourth-order implicit multistep at one nonlinear solve per step |

### IMEX Euler

`IMEX Euler` is the first split explicit-implicit scheme. It accepts an ordinary
RHS exactly like every other implicit method and treats that complete update as
implicit. To split terms, pass an `IMEXRHS` — a thin constructor over the
structured `RHS` interface (NOTES §3.12) whose `__call__` is the combined
`f = f_E + f_I`; this avoids overloading the existing `(update, aux)` RHS
return convention.

```python
from warpSPHIntegrators import IMEXRHS, getIntegrator

scheme = getIntegrator('IMEX Euler')
rhs = IMEXRHS(explicit=transport_rhs, implicit=diffusion_rhs)
result = scheme(system, dt=dt, f=rhs)
```

The explicit callback is evaluated from the known state once per step. Only the
implicit callback is evaluated inside the JFNK nonlinear solve.

### Additive (IMEX) Runge-Kutta — ARK3(2)4L[2]SA and ARK4(3)6L[2]SA

The two Kennedy–Carpenter additive pairs are the higher-order split methods: each
stage is an explicit (ERK) half plus a diagonally-implicit (ESDIRK) half, and the
propagated update is the additive `b`-weighted combination of both. They take the
same `IMEXRHS` split as IMEX Euler — an ordinary RHS stays fully implicit (the step
reduces to the implicit ESDIRK half, a standalone method of the same order). The
combined method is **not** FSAL (the explicit half is not stiffly accurate), so a
reused `priorStep` is rejected and the schemes register `stiffly_accurate=False`.

```python
from warpSPHIntegrators import IMEXRHS, getIntegrator

scheme = getIntegrator('ARK3(2)4L[2]SA')   # or 'ARK4(3)6L[2]SA'
rhs = IMEXRHS(explicit=transport_rhs, implicit=diffusion_rhs)
result = scheme(system, dt=dt, f=rhs)
```

When both callbacks are active, the implicit callback owns the stage buffer (the JFNK
solve and its post-convergence re-evaluation run on it) and the explicit callback is
evaluated on a throwaway clone of the converged stage; the final state's copied fields
come from the implicit buffer. Each carries an embedded (3, 2) / (4, 3) error
estimate, and their two-parameter stability region is in
[NOTES.md §3.6](NOTES.md#36-valid-schemes-and-what-each-costs) and
`images/imex_stability_slices.png`.
| **Adams-Bashforth-Moulton 2–4 (PECE)** | 2–4 | 2 | order − 1 | Predict-Evaluate-Correct-Evaluate; a fixed (uniterated) correction |

No linear multistep method is symplectic for a general Hamiltonian (Tang, 1993); all seven measure
`dissipation=True`. None implement `priorStep` reuse — that is a different, single-entry-lookback
mechanism `history=`'s multi-entry `StepHistory` generalizes past, not an alternative spelling of it.

## Supported Integrators

This registry-aligned table is the short reference for every available scheme. Stability is the
scalar Dahlquist classification when it applies. `yes` means the direct nonlinear Kepler
symplectic-form test passes; `linear only` means the linear oscillator map preserves phase area,
but the scheme is not generally symplectic. A `conditional` result assumes a separable,
position-only Hamiltonian.

| Scheme | Order | Kind | General Property | Symplectic |
|---|---:|---|---|---|
| Forward Euler | 1 | Explicit Euler | Baseline; bounded on $[-2,0]$ | no |
| Explicit Euler | 1 | Explicit Euler | Baseline alias | no |
| Midpoint | 2 | Explicit RK | General second-order RK | no |
| Heun's Method (2nd order) | 2 | Explicit RK | General second-order RK | no |
| Ralston's Method (2nd order) | 2 | Explicit RK | Low local-error constant | no |
| RK3 | 3 | Explicit RK | General third-order RK | no |
| Heun's Method (3rd order) | 3 | Explicit RK | General third-order RK | no |
| Ralston's Method (3rd order) | 3 | Explicit RK | General third-order RK | no |
| Wray's Method (3rd order) | 3 | Explicit RK | General third-order RK | no |
| SSP RK3 | 3 | Explicit SSP RK | Strong-stability-preserving | no |
| RK4 | 4 | Explicit RK | Classical fourth-order RK | no |
| RK4 (alternative) | 4 | Explicit RK | 3/8-rule fourth-order RK | no |
| Nystrom 5th order | 5 | Explicit RK | Fifth-order reference method | no |
| Bogacki-Shampine 3(2) | 3 | Embedded explicit RK | Error estimate; FSAL | no |
| Dormand-Prince 5(4) | 5 | Embedded explicit RK | Error estimate; FSAL | no |
| Cash-Karp 5(4) | 5 | Embedded explicit RK | Error estimate | no |
| Semi-Implicit Euler | 1 | Symplectic Euler | Lowest-cost geometric method | yes |
| Symplectic Euler | 2 | Symplectic splitting | Second-order kick-drift-kick | yes |
| Leap Frog | 2 | Symplectic splitting | $dt\omega\le2$; conditional | yes, conditional |
| Velocity Verlet | 2 | Symplectic splitting | $dt\omega\le2$; conditional | yes, conditional |
| PEFRL | 4 | Symplectic splitting | Forest-Ruth variant; conditional | yes, conditional |
| VEFRL | 4 | Symplectic splitting | Forest-Ruth variant; conditional | yes, conditional |
| EPEC | 2 | Explicit RK | Midpoint-style PECE | no |
| EPEC Modified | 2 | Explicit RK | Heun-style PECE | no |
| TVD RK2 | 2 | Explicit TVD RK | Conservative/TVD form | no |
| TVD RK3 | 3 | Explicit TVD RK | Shu-Osher SSP form | no |
| Backward Euler (implicit) | 1 | DIRK | L-stable | no |
| Implicit Midpoint | 2 | DIRK | A-stable; strict solve for geometry | yes |
| Trapezoidal (Crank-Nicolson) | 2 | DIRK | A-stable and symmetric | linear only |
| SDIRK2 | 2 | DIRK | L-stable | no |
| TR-BDF2 | 2 | DIRK | L-stable and stiffly accurate | no |
| ESDIRK3(2)4L[2]SA | 3 | DIRK | L-stable, stiffly accurate, explicit first stage | no |
| ESDIRK4(3)6L[2]SA | 4 | DIRK | L-stable, stiffly accurate, explicit first stage | no |
| Newmark | 2 | Implicit second-order | Average acceleration is oscillator-unconditionally stable | linear only |
| BDF1 | 1 | Implicit multistep | L-stable | no |
| BDF2 | 2 | Implicit multistep | A-stable | no |
| BDF3 | 3 | Implicit multistep | Sectorial A(alpha) stability | no |
| BDF4 | 4 | Implicit multistep | A(alpha), 73.35 deg cone | no |
| BDF5 | 5 | Implicit multistep | A(alpha), 51.84 deg cone | no |
| IMEX Euler | 1 | IMEX | Explicit/implicit split, JFNK implicit side | no |
| ARK3(2)4L[2]SA | 3 | Additive IMEX RK | L[2]-stable split (ERK + ESDIRK half); embedded (3, 2); not FSAL | no |
| ARK4(3)6L[2]SA | 4 | Additive IMEX RK | L[2]-stable split (ERK + ESDIRK half); embedded (4, 3); not FSAL | no |
| Adams-Bashforth 2 | 2 | Explicit multistep | One RHS evaluation after startup | no |
| Adams-Bashforth 3 | 3 | Explicit multistep | One RHS evaluation after startup | no |
| Adams-Bashforth 4 | 4 | Explicit multistep | One RHS evaluation after startup | no |
| Adams-Bashforth 5 | 5 | Explicit multistep | One RHS evaluation after startup | no |
| Adams-Bashforth-Moulton 2 (PECE) | 2 | Predictor-corrector | Two RHS evaluations after startup | no |
| Adams-Bashforth-Moulton 3 (PECE) | 3 | Predictor-corrector | Two RHS evaluations after startup | no |
| Adams-Bashforth-Moulton 4 (PECE) | 4 | Predictor-corrector | Two RHS evaluations after startup | no |
| Adams-Moulton 2 (implicit) | 2 | Implicit multistep | JFNK-corrected trapezoidal rule; A-stable | no |
| Adams-Moulton 3 (implicit) | 3 | Implicit multistep | JFNK-corrected AM3; bounded stability region | no |
| Adams-Moulton 4 (implicit) | 4 | Implicit multistep | JFNK-corrected AM4; bounded stability region | no |

## First-stage reuse (`priorStep`)

Passing `priorStep=result.stages[-1]` feeds the last stage of step *n* in as the first
stage of step *n+1*, saving one right-hand-side evaluation. It is standard practice in
the SPH literature, but **whether it is valid is a property of the tableau, not of the
caller** — so ask before opting in:

```python
from warpSPHIntegrators import getIntegrator, step_reuse_order, supports_step_reuse

scheme = getIntegrator('Dormand-Prince 5(4)')
supports_step_reuse(scheme)   # True  -- FSAL, reuse is exact
step_reuse_order(scheme)      # 5

scheme = getIntegrator('SSP RK3')
supports_step_reuse(scheme)   # False
step_reuse_order(scheme)      # 1  -- third order becomes first
```

The same lossless reuse applies to the stiffly accurate DIRK schemes with an explicit
first stage (Trapezoidal, TR-BDF2, `ESDIRK3(2)4L[2]SA`, `ESDIRK4(3)6L[2]SA`): their last
stage is `f(t^{n+1}, y^{n+1})` exactly, so `supports_step_reuse` is `True` for them too.
The DIRK reuse is exact only up to the JFNK solve's last-bit sensitivity (~1e-11) rather
than the ~1e-14 the all-explicit FSAL pairs reach.

Supplying `priorStep` to a scheme that loses order still works — trading order for half
the evaluations is a legitimate choice — but warns once, naming the order you will
actually get. `step_reuse_order` returns `None` for schemes that do not implement reuse
at all; they warn and ignore it. Single-stage schemes refuse outright, because reuse
turns them into `y^{n+1} = y^n + dt·f(y^{n-1})`, which has no stability region on the
imaginary axis.

```python
result = scheme(system, dt=dt, f=rhs)
result.error        # embedded pairs only: y_high - y_low, as a state
```

## Example: Damped Harmonic Oscillator with Feedback

See [integrators.ipynb](integrators.ipynb) for the complete interactive example. This example demonstrates:

- Custom state with multiple field types
- Energy-dependent feedback (spring constant varies with stored energy)
- Comparison of three integration schemes (RK4, RK2, Forward Euler)
- Visualization of phase plot, energy evolution, and trajectories

**Example Output:**

![Integration schemes comparison: Phase plot and energy evolution](images/example.png)

The visualization shows:
- **Phase Plot (top-left)**: How position and velocity evolve together
- **Energy (top-right & bottom-right)**: Kinetic, potential, and total energy over time
- **Position/Velocity Trajectories**: Individual component evolution for each scheme

## Integrator Image Gallery

| Order | Modified Harmonic Oscillator | Integrator Comparison |
|------:|-------------------------------|-----------------------|
| 1 | ![Modified harmonic oscillator: order 1 integrators](images/modified_harmonic_oscillator_order_1_integrators.png) | ![Integrator comparison for order 1](images/integrator_comparison_order_1.png) |
| 2 | ![Modified harmonic oscillator: order 2 integrators](images/modified_harmonic_oscillator_order_2_integrators.png) | ![Integrator comparison for order 2](images/integrator_comparison_order_2.png) |
| 3 | ![Modified harmonic oscillator: order 3 integrators](images/modified_harmonic_oscillator_order_3_integrators.png) | ![Integrator comparison for order 3](images/integrator_comparison_order_3.png) |
| 4 | ![Modified harmonic oscillator: order 4 integrators](images/modified_harmonic_oscillator_order_4_integrators.png) | ![Integrator comparison for order 4](images/integrator_comparison_order_4.png) |
| 5 | ![Modified harmonic oscillator: order 5 integrators](images/modified_harmonic_oscillator_order_5_integrators.png) | ![Integrator comparison for order 5](images/integrator_comparison_order_5.png) |

### Newmark demo figures for the default oscillator case

These match the gallery style used for the drift / phase plots of the other schemes, but specifically highlight the second-order Newmark family on the repo’s default oscillator demo.

| Plot | Figure |
|------|--------|
| Modified oscillator with the second-order family | ![Newmark family on the default oscillator demo](images/newmark_modified_harmonic_oscillator_order_2_integrators.png) |
| Difference from the reference 5th-order solver | ![Newmark difference plot against the reference Nystrom 5th-order scheme](images/newmark_integrator_comparison_order_2.png) |

### All implicit schemes on the default oscillator case

![Phase, position, hidden-energy, and reference-energy comparison for every implicit scheme](images/all_implicit_schemes_default_demo.png)

## Stability Regions and Stiff Problems

For the Dahlquist test equation $y' = \lambda y$, a stability region is the set of
$z = dt\lambda$ for which the numerical solution stays bounded. The green region in
the tableau plot and the orange region in the BDF plot are the computed regions for
the methods' propagated solutions; they serve as a regression check on the registered
tableaus and BDF coefficients.

![Dahlquist stability regions for implemented explicit and diagonally implicit tableau methods](images/dahlquist_stability_tableau_methods.png)

![Dahlquist stability regions for BDF1 through BDF5](images/dahlquist_stability_bdf_methods.png)

Run `conda run -n warp python scripts/stability_gallery.py` to regenerate these
figures. Newmark and the Verlet family are second-order oscillator methods, so their
relevant stability domain is parameterized by $dt^2\omega^2$, not scalar Dahlquist
$z$. IMEX methods likewise have a two-parameter region $z_{explicit}, z_{implicit}$
that depends on the caller's chosen split; plotting either as a one-parameter region
would be misleading.

![Two-parameter IMEX stability slices: IMEX Euler, ARK3(2)4L[2]SA, ARK4(3)6L[2]SA over $z_{implicit}$ slices in the $z_{explicit}$ plane](images/imex_stability_slices.png)

![Undamped oscillator amplification stability for Newmark and Verlet-family methods](images/oscillator_stability_newmark_verlet.png)

The plot uses the spectral radius of the linear oscillator amplification matrix in
the dimensionless variable $dt\omega$. Leap Frog, Velocity Verlet, and Symplectic
Euler are bounded through $dt\omega=2$; average-acceleration Newmark is unconditionally
bounded for this undamped linear problem, while linear-acceleration Newmark is bounded
through $dt\omega=\sqrt{12}$.

With damping, the amplification matrix gains the dimensionless damping
$c = 2\zeta\,dt\omega$ and the stability domain is two-parameter. The gallery below
shows $\log_{10}\rho(h\omega, \zeta)$ for $h\omega \in [0, 8]$ and $\zeta \in [0, 1]$
with the $\rho = 1$ boundary contoured (the same closed-form matrices as the test
suite's one-step cross-checks, in `warpSPHIntegrators.stability`). Damping extends
velocity Verlet's region — at $h\omega = 2.5$ the stable window is
$0.5055 < \zeta < 0.8$ — but it cannot rescue Leap Frog: its damping term is
evaluated at the *old* velocity, so it is itself an explicit update unstable for
$c > 2$. Newmark (average acceleration) is neutrally stable undamped for every
$h\omega$ and asymptotically stable for any $\zeta > 0$.

![Damped oscillator stability: log10 spectral radius over (dt*omega, zeta) for Newmark and the Verlet family, rho = 1 boundary contoured](images/oscillator_stability_damped_newmark_verlet.png)

Run `conda run -n warp python scripts/oscillator_stability_gallery.py` to
regenerate both figures.

### Nonlinear and stiff benchmarks

`warpSPHIntegrators.testing` ships eleven problem factories, five of them stiff or
nonlinear stiff and two hyperbolic (`advection_problem`, `burgers_problem`), used
by `tests/test_benchmarks.py` (52 tests, all small and deterministic),
`tests/test_tvd.py`, and by the benchmark script below:

- `stiff_relaxation_problem(rate, forcing, sign)` — Prothero–Robinson with a
  configurable smooth target; `sign=-1` is the *unstable* variant (positive
  eigenvalue) whose exact solution is still the target.
- `stiff_damped_oscillator_problem(omega, c)` — separates high-frequency stiffness
  from true dissipative stiffness; closed-form exact solution in all three damping
  regimes.
- `van_der_pol_problem(mu)`, `robertson_problem()`, `diffusion_problem(n)` — limit
  cycle, stiff chemistry (invariants: mass conservation, positivity, monotone
  $y_3$, quasi-steady $y_2$), and spectral-stiffness diffusion on Laplacian
  eigenvectors, respectively.

`conda run -n warp python scripts/stiff_benchmark_suite.py` regenerates the
accuracy/cost figure below (all parameters and the JFNK settings are stated in the
script's docstring): error vs $dt$ on the four stiff benchmarks, with the explicit
methods' walls marked where the step restriction is actually active, per-step
RHS-evaluation and GMRES-iteration panels, and the van der Pol energy evolution
(settling orbits fill a bounded band; the diverging ones leave the axis).

![Phase 8 stiff benchmark suite: error vs dt on Prothero-Robinson, diffusion, van der Pol, and Robertson kinetics, per-step RHS and GMRES cost, and van der Pol energy vs time](images/stiff_benchmark_suite.png)

Two behaviours worth knowing before reading those curves:

- **Robertson needs a bootstrap.** The initial quasi-steady layer (width $\sim10^{-3}$)
  is not crossable from $y(0) = (1,0,0)$ at $dt \ge 0.02$; ten $dt = 0.001$ steps
  put the solution on the quasi-steady manifold first. Backward Euler at
  $dt \ge 2$ then converges each nonlinear solve to an unstable discrete fixed
  point of the stiff quadratic and diverges — BE inaccuracy above the fast scale
  (fast eigenvalue $\sim 2.2\times10^{3}$), not a solver failure.
- **ESDIRK6's stability function has a pole at $z = 1$** (the stiffly-accurate
  $\gamma = 1$ stage), so it diverges on the *unstable* Prothero–Robinson variant
  at $z = +5$ even though it is A-stable. On that problem only BE/BDF-type methods
  track the solution, and there the explicit wall is the point: DP5 amplifies the
  unstable mode by $R(+5) = 117$ per step.

### TVD / total-variation classification

The TVD/SSP property of the registered schemes is measured, not asserted by name.
`warpSPHIntegrators.tvd_analysis` classifies every registered scheme on the model
hyperbolic problem (1D periodic first-order upwind advection, step initial
condition, `testing.advection_problem`) with two independent measurements:

- the **SSP coefficient** `convex_combination_cfl(tableau)` — the largest CFL up
  to which every stage-value map *and* the final map is a convex combination of
  periodic shifts, from the Fourier stage maps of the upwind model. A
  convex-combination step is TVD by construction, so this is a *rigorous* TVD CFL
  for every scheme that exposes its tableau;
- the **measured per-step TVD CFL** `measure_tvd_cfl(scheme, problem, cfls,
  dt_scale)` — the registered driver run on the step initial condition, requiring
  the per-step total-variation increase to stay within $10^{-8} \cdot TV_0$ in a
  post-burn-in window. This one sees the actual driver, so it also classifies the
  schemes that expose no tableau (the BDF/Adams family).

Results (n = 64, tested to CFL 5): the TVD-named schemes — TVD RK2, TVD RK3, and
SSP RK3, together with every second- and third-order explicit RK scheme — have
the published $r = 1$ and measure TVD to CFL 1. TVD is strictly broader than SSP:
classical RK3 has $r = 1/2$ yet a per-step TVD CFL of 1 (RK4: $r = 2/3$, TVD to
1.25); Nystrom 5th order and Dormand–Prince 5(4) have $r = 0$ — their stage maps
leave the convex hull at *any* CFL, verified by exact rational arithmetic — and
are still per-step TVD to CFL 1.5. The L-stable implicits (Backward Euler,
SDIRK2, TR-BDF2, ESDIRK3/4, the ARK pairs) are TVD to the tested CFL 5, and the
multistep family (BDF2–5, Adams-Bashforth 2–5, ABM 2–4, implicit AM 2–4) to a
measured CFL 1.5. The Verlet family and Newmark integrate second-order systems,
so the TVD question does not apply to them.

The per-scheme verdict table is the maintained fact printed by
`python scripts/tvd_classifier.py` (re-run it to refresh) and pinned by
`tests/test_tvd.py`; NOTES.md §3.11 has the full landscape, including why
per-step TV cannot separate classical RK3 from TVD RK3 at CFL 1 and how the
classifier detects the difference at the stage level instead.

## Symplecticity Validation

The test suite checks symplecticity directly, not only through energy drift. On the
one-degree-of-freedom oscillator it finite-differences every registered one-step map
and asserts $\det(D\Phi_h)=1$ for the expected area-preserving methods. It additionally
checks $D\Phi_h^T\Omega D\Phi_h=\Omega$ on nonlinear Kepler dynamics for Leap Frog,
Velocity Verlet, Symplectic Euler, PEFRL, VEFRL, Semi-Implicit Euler, and implicit
midpoint. Crank-Nicolson and average-acceleration Newmark preserve oscillator phase
area but are deliberately excluded from the nonlinear symplectic set: that linear
property does not establish general Hamiltonian symplecticity. The implicit checks
use a strict JFNK tolerance; an intentionally truncated Picard solve or loose Newton
termination approximates the map and cannot be expected to preserve its exact
geometric invariants.

## API Reference

### State Definition

```python
from warpSPHIntegrators import BaseState, integrated, constant, copied, ephemeral, custom

@dataclass
class MyState(BaseState):
    # Integrated: evolves using derivative field
    x: torch.Tensor = integrated('dx_dt', tags=('position',))
    
    # Constant: fixed throughout integration
    m: torch.Tensor = constant(tags=('mass',))
    
    # Copied: recomputed each stage; the final state inherits the last stage's value
    density: torch.Tensor = copied(default=None)
    
    # Ephemeral: stage-local scratch, never carried out of the step
    neighbours: Any = ephemeral(default=None)
    
    # Custom: the library will not touch it; your system owns it entirely
    custom_field: Any = custom(default=None)
```

### Update Specifications

```python
from warpSPHIntegrators import (
    PositionUpdateSpec, ComponentUpdateSpec, StateBlend,
    blend_state, explicit_step, semi_implicit_position_step, verlet_position_step,
)

# `StateBlend` is a three-field dataclass describing how the existing value is folded
# in:  value <- self_scale * value + reference_weight * reference_state_value + ...
# `reference_state` and `reference_weight` must be given together.
blend = blend_state(self_scale=3/4, reference_state=y_1, reference_weight=1/4)

# Component (velocity, quantity, ...) update: value += derivative_dt * k
comp_spec = ComponentUpdateSpec(derivative_dt=0.5 * dt, blend=blend)

# Position update, plus optionally one of the two special position modes:
#   current_velocity_dt=s  ->  x += s * v_current       (semi-implicit drift)
#   update_velocity_dt=s   ->  x += s * update.velocity (Verlet correction)
pos_spec = PositionUpdateSpec(derivative_dt=0.5 * dt)

# The constructors below are the preferred spelling:
explicit_step(dt)                                    # x += dt * k
semi_implicit_position_step(dt)                      # x += dt * v_current
verlet_position_step(dt, update_velocity_dt=dt**2/2) # x += dt*k.x + (dt^2/2)*k.v
```

### Integration Functions

```python
from warpSPHIntegrators import (
    IntegrationSchemeType,
    getIntegrator,
    IntegrationResult,
    StageResult,
)

# Get an integrator
integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

# Call it
result: IntegrationResult = integrator.function(
    state,           # Current system state
    dt=0.01,         # Time step
    f=rhs_function,  # RHS function: (system, dt, **kwargs) -> (update, aux)
    verbose=False,   # Enable verbose output
    **kwargs         # Additional args passed to RHS
)

# Access results
next_state = result.state
stages = result.stages  # List of StageResult namedtuples
```

### Helper Functions

```python
from warpSPHIntegrators import (
    get_reference_state,        # Extract state from system
    get_tagged_attr,            # Access fields by tag
    update_position,            # Apply position update
    update_component,           # Apply component update
)

# Extract state from system
state = get_reference_state(system)

# Get field by tag
position = get_tagged_attr(state, tag='position')

# Apply typed updates
system = update_position(system, update, spec, 
                        'position', 'position_derivative',
                        'velocity', 'velocity_derivative')
```

## Advanced Usage

### Multi-Stage Systems

For systems with multiple coupled components, use multiple field types:

```python
@dataclass
class ComplexState(BaseState):
    # Position-velocity pair
    position: torch.Tensor = integrated('dp_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    
    # Additional quantities
    energy: torch.Tensor = integrated('de_dt', tags=('energy',))
    temperature: torch.Tensor = constant(tags=('temperature',))
    
    # Metadata
    particle_id: int = copied()
```

### Custom Update Logic

`custom()` marks a field as one the library will not touch: it is neither integrated,
cloned, nor nulled by the generic paths, and your system is responsible for it entirely.
There is no `apply_fn` hook — the field is handled by your own `apply_*_update` methods
and lifecycle hooks.

```python
@dataclass
class MyState(BaseState):
    custom_field: Any = custom(default=None)

@dataclass
class MySystem(BaseIntegrationSystem):
    state: MyState = reference_state()
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        fresh = get_reference_state(self).initializeNewState()
        fresh.custom_field = my_own_rule(get_reference_state(self).custom_field)
        return MySystem(state=fresh, t=self.t)
```

If what you need is "recomputed every stage, and the final state should keep the last
stage's value" — summation density, pressure, a smoothing length — use `copied()`
instead. The integrator carries those over for you when the step is finalized.

### Passing Extra Arguments to RHS

```python
def advanced_rhs(system, dt, config: dict, verbose: bool = False):
    """RHS with extra parameters."""
    state = get_reference_state(system)
    # Use config['param'] as needed
    return update, aux

# Call with extra kwargs
result = integrator.function(
    system,
    dt=0.01,
    f=advanced_rhs,
    config={'param': 42},
    verbose=True
)
```

## Extending the Library

### Adding a New Integration Scheme

If the scheme has a Butcher tableau, that is the whole job — add it to
`getButcherTableau` and register `butcherScheme('yourTableau')`. The stage loop, the
embedded-pair handling, and the reuse analysis all fall out of the tableau:

```python
# in butcher.py
elif scheme == 'myTableau':
    return butcherTableau(a=..., b=..., c=...)   # b may be a (main, embedded) tuple

myScheme = butcherScheme('myTableau')

# in enums.py, then integration.py
IntegrationSchemes.append(IntegrationScheme(myScheme, 'My Scheme',
                                            IntegrationSchemeType.myScheme, order,
                                            dissipation, nonLagrangian))
```

`reuse_order` and `fsal` are derived automatically from the tableau; do not set them by
hand. Running `pytest` then holds the new scheme to its declared order on three problems,
checks its reuse order against the prediction, and checks its `dissipation` flag against
its actual energy behaviour.

For a hand-rolled scheme with no tableau:

1. Implement the scheme function in a new module or existing one
2. `kwargs.pop('priorStep')` — either reuse it or pass it to `reject_prior_step`, but
   never let it flow through into `f`
3. Call `initializeSystem` once, and set `.t` explicitly on every stage buffer
4. Evaluate `f` on a fresh `initializeNewState()` buffer, never on the caller's state
5. Return `IntegrationResult(state=..., stages=[StageResult(...), ...])`
6. Register in the `IntegrationSchemeType` enum and record its reuse behaviour in
   `HANDROLLED_REUSE` in [reuse.py](src/warpSPHIntegrators/reuse.py)

Example template:

```python
def myCustomScheme(state, dt, f, *args, **kwargs):
    """Custom O(h^p) integrator."""
    # Compute stages
    k1, aux1 = f(state, dt, *args, **kwargs)
    # ... more stages ...
    
    # Apply updates
    new_state = apply_state_update(state, ...)
    
    # Return consistent result
    return IntegrationResult(
        state=new_state,
        stages=[
            StageResult(aux=aux1, update=k1),
            # ... more stages
        ]
    )
```

### Implementing Custom State Behaviors

There are exactly four dispatch points — `apply_position_update`, `apply_velocity_update`,
`apply_quantity_update`, `apply_state_update`. There is no dispatch on arbitrary tag
names, so an extra field is routed through one of those four (`apply_quantity_update`
is the usual home) rather than through a method named after its tag:

```python
@dataclass
class MyState(BaseState):
    energy:  torch.Tensor = integrated('dedt', tags=('quantity',))
    entropy: torch.Tensor = integrated('dsdt', tags=('entropy',))

class MySystem(BaseIntegrationSystem):
    def apply_quantity_update(self, update, spec, **kwargs):
        update_component(self, update, spec, 'quantity', 'quantity_derivative')
        update_component(self, update, spec, 'entropy', 'entropy_derivative')
        return self
```

Tags select *which field* `update_component` reads and writes; they do not select which
method runs.

## Performance Notes

- **Memory**: Named tuples have minimal overhead vs raw tuples
- **Gradients**: All operations preserve gradient flow; use `torch.no_grad()` if differentiation isn't needed
- **GPU**: Works transparently with GPU tensors (CUDA/MPS)
- **Benchmarking**: See [integrators.ipynb](integrators.ipynb) for timing comparisons

## Known Limitations

- **Adaptive step-size control not built-in** (use external error estimators; the embedded pairs'
  `IntegrationResult.error` gives you the estimate, the driving loop and step-rejection path are not
  written yet — NOTES.md §2.3).
- **Implicit schemes differentiate by the implicit function theorem, not by unrolling.** The
    Newton/GMRES iteration runs under `torch.no_grad()`; the gradient is re-attached at the converged
    fixed point (`JFNKSolver.solve`). Two consequences worth knowing: the gradient is *exact* and
    independent of `newton_tol` — it depends on where the fixed point is, not on how many iterations
    found it — and asking for a gradient costs two extra `step` evaluations in the forward pass plus
    one vector-Jacobian product per Krylov iteration of the adjoint solve in the backward pass. The
    trajectory itself is bit-for-bit identical whether or not gradients are requested
    (`tests/test_gradients.py`).
- **JFNK is matrix-free but not fixed-cost.** The default DIRK/Newmark Newton solve uses residual
    evaluations and GMRES iterations, so it is not CUDA-graph-capturable in the way explicit Picard is.
    `FixedPointSolver` and `RelaxedFixedPointSolver` remain opt-in alternatives for a fixed schedule.
    The unpreconditioned GMRES iteration count grows with the stage operator's eigenvalue spread, so
    at particle scale a problem-specific operator-based preconditioner (the
    `JFNKSolver(preconditioner=...)` hook, DIRK section above) is what keeps the Krylov count flat.
- **Fully implicit RK is not implemented.** Gauss, Radau, and Lobatto methods need a coupled
    `s·N`-unknown solve rather than the sequential DIRK solve. BDF1–BDF5 and the fully implicit
    Adams-Moulton correctors AM2–AM4 are available; the coupled-RK family remains planned — scoped in
    [NOTES.md §3.6](NOTES.md#36-valid-schemes-and-what-each-costs).
- **Only two additive (IMEX) pairs are shipped.** The Kennedy–Carpenter
    ARK3(2)4L[2]SA and ARK4(3)6L[2]SA are implemented (with embedded estimators and the
    `IMEXRHS` split); other additive families (higher-stage ARK, Radau-type, Rosenbrock)
    remain planned (NOTES.md §3.6).

Explicit multistep (Adams-Bashforth 2–5, Adams-Bashforth-Moulton 2–4), implicit multistep
(BDF1–BDF5, fully implicit Adams-Moulton 2–4), seven DIRK schemes (Backward Euler, Implicit
Midpoint, Trapezoidal, SDIRK2, TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA), and two
additive IMEX pairs (ARK3(2)4L[2]SA, ARK4(3)6L[2]SA) *are* implemented — see the sections
above. Everything still open
is scoped and costed in [NOTES.md §3](NOTES.md#3-multistep-and-implicit-methods), including which
schemes are worth adding next and what each one costs.

## Contributing

Contributions welcome! Areas of interest:

- Remaining implicit schemes (fully implicit RK: Gauss, Radau, Lobatto; BDF6+; Rosenbrock/W)
- Adaptive time stepping
- Better documentation and examples
- Performance optimizations
- Additional state field behaviors

## License

Apache 2.0 — See [license.md](license.md)

## Citation

If you use this library in research, please cite:

```bibtex
@software{warpSPHIntegrators2024,
  title={Differentiable ODE Integration with PyTorch},
  author={Winchenbach, Rene},
  url={https://github.com/wi-re/warpSPHIntegrators},
  year={2024}
}
```

## References

- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving ordinary differential equations I: Nonstiff problems*
- Ruth, R. D. (1983). A canonical integration technique. IEEE Trans. Nucl. Sci., 24(2), 2669–2671
- Verlet, L. (1967). Computer "experiments" on classical fluids. I. Thermodynamical properties by a molecular dynamics method. Phys. Rev., 159(1), 98
