# Physical operator splitting

`ConservativeDissipativeRHS` declares two autonomous physical parts. `LieTrotter`,
`Strang`, `Yoshida4` and `compose` build parameterized fixed-step schemes over existing
one-step integrators. They remain outside the fixed catalog; pass the returned
scheme to `solve` or call it directly. The 137 existing registrations are intact.

[example:include examples/operator_splitting.py]: #
```python
"""Physical operator splitting: python examples/operator_splitting.py."""

import math
import torch
from warpSPHIntegrators.wrappers import (
    ConservativeDissipativeRHS, Strang, LinearExponentialFlow,
)
from warpSPHIntegrators.adapters import SemilinearRHS
from warpSPHIntegrators import TensorState, tensor_rhs, solve


def run_example():
    # q' = v, v' = -4*q - .5*v: motion and damping are separate maps.
    initial = TensorState(torch.tensor([1., 0.], dtype=torch.float64))
    damping = torch.diag(torch.tensor([0., -.5], dtype=torch.float64))
    rhs = tensor_rhs(ConservativeDissipativeRHS(
        conservative=lambda t, y: torch.stack((y[1], -4*y[0])),
        dissipative=SemilinearRHS(linear=lambda t, y: damping@y)))
    method = Strang('RK4', LinearExponentialFlow(damping))
    result = solve(method, initial, 1., rhs, dt=.01)
    omega = math.sqrt(4 - .25**2)
    exact = torch.tensor([
        math.exp(-.25)*(math.cos(omega) + .25/omega*math.sin(omega)),
        -4/omega*math.exp(-.25)*math.sin(omega)], dtype=torch.float64)
    assert method.order == 2
    assert result.state.t == 1.
    assert torch.linalg.vector_norm(result.state.y-exact) < 1e-4
    assert torch.equal(initial.y, torch.tensor([1., 0.], dtype=torch.float64))
    return result


if __name__ == '__main__':
    result = run_example()
    print(f't={result.state.t}, [q, v]={result.state.y.tolist()}, steps={result.accepted_steps}')
```

## Accuracy and structural checks

Lie–Trotter advances each part once and is first order. Strang applies a
half dissipative step, a full conservative step, then another dissipative half.
Its order is the minimum of two and the sub-integrator orders. A backward-Euler
half therefore reports/measures order one. Second-order sub-integrators suffice
for second-order accuracy; symmetry is additionally needed for time reversibility
and the usual higher-order symmetric-composition arguments. Explicit midpoint
is a tested second-order example without that symmetry.

The current driver requires a declared autonomous split. Component pseudo-times
are reset to the outer start, and the returned state ends at `t+dt`; adding both
component clocks would double-advance physical time. Non-autonomous composition
needs an explicit clock extension and remains unsupported. Timestep-dependent
compatible-energy maps are not silently treated as autonomous vector fields.

Physical names alone do not certify a Hamiltonian. `check_contracts` checks
additivity and zero dissipative position rate; `require_separable=True` adds
mechanical probes. `classify_separable` measures conservative force dependence
on velocity at several perturbed velocities, checks position rate equals velocity
and rejects evolving extra integrated quantities. This catches the thermal-energy
shortcut even when force itself appears velocity-independent. These are local
structural diagnostics, not global mathematical or variational proofs. Unsupported
AD and missing mechanical tags produce a non-certified verdict. Geometric
conservative sub-integrators require these checks before stepping. Transported
PDE scalar values should use TensorState, not particle-position tags.

`LinearExponentialFlow(matrix_or_provider)` advances a declared homogeneous linear
part with Torch matrix_exp in flattened integrated-field order. It checks matrix
shape/device/dtype and the declared action at the current state. The dense matrix
path preserves state and matrix-parameter gradients and is intended for small
systems. Sparse/Krylov flow implementations for large SPH remain application work;
a declared operator depending on frozen configuration must remain fixed within
its subflow. Implicit midpoint supports nonlinear dissipative vector fields through
the existing JFNK/IFT path. Neither method removes a separate convective CFL bound.

## Lifecycle, ownership and options

Composition snapshots the caller and calls outer `initialize`/`finalize` once.
Inner steps retain preprocess/postprocess and copied-field propagation; they skip
outer finalization and call optional `initialize_substep` instead of outer initialize.
A custom finalize must declare `composition_compatible=True` and handle grouped
`composition_substeps`. Its return/update arguments are component results/stages,
and weights are their composition coefficients, not a combined Butcher quadrature.
Existing compatible-energy SPH finalizers require a dedicated substep work design.
No such production integration is claimed here.

Bind per-part options through `conservative_options` and `dissipative_options`.
Common step kwargs are passed to each part, with bound part options overriding
them. Full-protocol RHS/hooks must accept their consumer's kwargs, including the
internal substep flag and split-part label; Tensor callbacks receive only their
bound parameters. A failed sub-solve propagates its status without outer finalize;
`solve` rolls back the candidate and transaction. Precision-floor handling remains
the run API's explicit policy after the complete composition is assembled.

The runner supports accepted callbacks, fixed-grid output and in-memory restart
for the bound recipe. Supply the same RHS/parameters/operator/options on restart.
Adaptive estimates, dense interpolation, events, FSAL and multistep sub-integrators
are unsupported. Component stages are component derivatives, not a global FSAL
stage. Generic `compose` accepts orders one/two and refuses negative dissipative
coefficients; backward physical integration retains its ordinary signed-step behavior.

`Yoshida4(conservative, LinearExponentialFlow(matrix), negative_step_limit=.5)`
is a separate fourth-order recipe. It uses the symmetric triple jump from
[Yoshida (1990), Eq. 2.11](https://doi.org/10.1016/0375-9601(90)90092-3), with
adjacent exact D flows combined. It requires a supported symmetric conservative
integrator and a static Euclidean-dissipative matrix. Each negative D step must
satisfy `abs(tau)*norm(matrix, 2) <= negative_step_limit`. The default bound is an
engineering restriction on anti-diffusive growth, not a theorem of unconditional
stability. Strong damping is rejected before initialization. Generic nonlinear
or stiff negative dissipative flows and arbitrary higher-order recipes are unsupported.

## Toy problems and measurements

For the oscillator `q'=v, v'=-4*q-.5*v`, the conservative step advances
`q'=v, v'=-4*q`, and the dissipative step holds `q` fixed while multiplying
velocity by `exp(-.5*tau)`. These two simple maps separate mechanical motion
from damping. The complete oscillator has a known solution, so trajectory errors
can be measured without an SPH model or thermodynamic closure.

The split oscillator and periodic Burgers cases verify orders one/two/four, mass,
state/parameter gradients, failure propagation and lifecycle ownership. Implicit
and exponential diffusion stay bounded at 8–82 times the explicit diffusion step
bound; the explicit control is unstable. Accuracy still requires suitable steps.
Strang with Verlet/exact damping has phase-space determinant `exp(-.5*h)` to
roundoff; the restricted fourth-order recipe matches within 1e-9. Unsplit RK4
has a measurable defect at the same step. The oscillator's mechanical-energy
change plus trapezoidal integrated viscous loss converges at order two. These
are constant-damping toy diagnostics, not general SPH conservation claims.

A fixed-step static exact-flow benchmark merges two neighboring dissipative
half-flows: 20 steps use 21 flows instead of 40 and agree to roundoff. The
benchmark changes lifecycle boundaries and is not an automatic runner optimization;
callbacks, synchronized samples and non-exact/variable-step flows need a separate
representation and ownership design.

The known oscillator and PDE examples qualify the composition behavior without
requiring a new physics model. Entropy/compatible-energy SPH and large sparse
exponential flows still need application-specific checks. Nyström and the other
one-step configurations fit only where their state, force and quadrature
contracts match the physical subflow. The [validation record](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/phase7-foundation-validation.md)
keeps the measurements and application limits.
