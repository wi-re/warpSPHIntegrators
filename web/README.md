# Browser integrator laboratory

A standalone static browser port of warpSPHIntegrators, with the same dark panel
layout as the boundary-integral and remote tools. All computation stays in a
module worker on the visitor's device. No Python server or npm dependencies are
needed at runtime.

From this repository:

```bash
cd web
npm run dev
# Open http://127.0.0.1:4173
npm test
npm run check
npm run build
```

`dist/` is a portable static site. Serve it over HTTPS or localhost; browsers
require a secure context for WebGPU. If an adapter is unavailable or a GPU run fails, Auto retries with the
browser's float64 CPU engine and reports why. Explicit WebGPU selection preserves
the GPU failure for inspection. For WebGPU API details, see the
[MDN compute pipeline reference](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createComputePipelineAsync).

The documentation site's `npm run build` also copies the app into its
`laboratory/` route. The generated documentation assets are ignored; edit this
`web/` source directory.

## Experiments

- **Trajectory:** compare up to six methods, inspect time histories, phase/orbit
  curves, global error, signed energy defect, spatial profiles and space–time
  plots. Playback progressively reveals state, trajectory-error and energy curves
  with a synchronized cursor. Hover or focus a scheme in the left list, legend
  or results table to highlight its curves across the plots.
- **Convergence:** fixed-step refinement, actual endpoint RMS errors, observed
  orders, work and wall time. Auto selects CPU float64 here. Analytic references
  use the same semi-discrete spatial operator. Problems without a closed form
  use two refined Radau IIA runs and report their difference. Orders disappear
  below a conservative reference, rounding or solve-residual threshold; a plateau is not presented as order.
- **Symplectic:** long-run energy and phase plus the nonlinear canonical-form
  test `||DΦᵀ Ω DΦ − Ω||F`. The latter uses central differences of a separate
  CPU float64 step map and is labeled accordingly. Hidden-state methods are not
  incorrectly classified from a reduced position/velocity Jacobian.
- **SSP/TVD:** periodic upwind advection and local Lax–Friedrichs Burgers. Inspect
  maximum stage TV, stage overshoot, endpoint TV and the initial forward-Euler
  CFL scale. Known baseline SSP coefficients are shown; an unspecified
  coefficient is not claimed to be zero. A successful profile is not an SSP proof.
- **Stability:** scalar amplification from the source RK tableau or stabilized
  recurrence, compiled once to a polynomial/rational function for fast sampling.
  Maps default to 512 × 512; the visible resolution selector reaches 2048 × 2048.
  A bounded image is not a proof of A/L stability.
- **Analytic adjoints:** fixed-problem Jv/Jᵀv, their dot-product identity, and a
  continuous tangent integrated with RK4. This is explicitly independent of the
  selected integrator and does not claim AD through a numerical solve. The
  Newton–Krylov preset separately demonstrates matrix-free stage solves using Jv.

The presets cover smooth motion, stiff decay, SSP/TVD, nonlinear Kepler motion,
two-mass structural damping, history methods, and Newton–Krylov on viscous Burgers. Every preset
and problem selection resets to a tested combination of methods, physical
parameters and solver options. A description above the plots explains the
selected problem, initial state and reference. Controls expose
physical parameters, a categorized method browser with family/order/search
filters, step/end time, embedded
adaptivity, tolerances, recurrence stages, Newmark parameters and generalized
alpha's limiting spectral radius. Clicking a selected method opens its category
in replacement mode; Browse methods adds another comparison. Hover or keyboard
focus updates the right-hand inspector and linked path highlighting. Setup links preserve configuration; JSON exports
contain the executed setup, samples, refinement tables, engine, counters and failures.

## Numerical coverage

The checked-in source export contains **142 registered schemes and 138 additional
configurations**. All 280 configurations have CPU numeric drivers. **249** can
produce WebGPU kernels for a compatible small mechanics problem; eligibility
also depends on the particular problem, dimensions and selected options.

The additional configurations cover all 54 ExpandedRK maps; Gauss/Radau
collocation at 2–9 stages; all 14 Nyström and 14 ExponentialRK configurations;
three parameterized Rosenbrock maps; Euler/midpoint extrapolation; three bounded
stage configurations for each OrthogonalRK method; stabilized RKL/RKG; ParallelDIRK;
generalized alpha, NDF1–5 and corrected IRKN3/4. The 17 toy systems cover all 12 `testing.PROBLEMS`, the
acoustic and drift-field helpers, and the heated and structural examples.

The **Port coverage** view is the maintained user-facing record of limitations:

| Contract | Browser implementation / qualification |
| --- | --- |
| Explicit, DIRK, coupled RK, IMEX | Source coefficients, including available companions and stage times |
| Multistep | Accepted history, method-specific starters and uniform endpoint fitting; no repeated cold starts |
| Implicit closure | CPU analytic dense Newton or matrix-free Newton–GMRES; GPU analytic dense Newton; the linear two-mass Newmark/generalized-alpha system uses an equivalent direct modal solve to avoid float32 cancellation |
| Embedded adaptivity | CPU error-scaled acceptance/rejection for supported estimates; discrete samples |
| WebGPU | Float32; 64 state entries; 48 coupled stage unknowns; 100,000 steps per job |
| CPU fallback | Float64; cancellable worker; explicit failure messages and reference budgets |
| Exponential actions | CPU operator programs from source; scaled augmented Taylor actions with explicit budgets, distinct from Python's Arnoldi action policy |
| General AD | Not available; fixed analytic RHS derivatives only |
| VariableAdams / AdaptiveAdams | Variable-step and order controllers remain Python-only |
| NDF | Orders 1–5, source DIRK startup, owned backward differences and bounded variable-h transforms; CPU defect control with quasi-constant blocks and fixed-step WGSL |
| AdaptiveNDF | Separate variable-order controller remains Python-only |
| ImprovedNystrom | Corrected startup and actual IRKN3/4 warm recurrences on CPU and WebGPU; state/time/h changes restart CPU history; separable mechanics with passive extras |
| Orthogonal stage selection | Three bounded source stage choices per method; no automatic spectral sizing |
| Application mechanics | Conservative Lagrangian/projection wrappers and custom lifecycle contracts remain Python-only |
| Runtime | Implicit FSAL/cache policies, dense output, events, Python checkpoints, AlgorithmSwitch, arbitrary FunctionMap and user-defined physical splitting are not ported |
| Structured drift quantities | Generic ODE integration is available; drift-aware mechanics hooks for acoustic/drift-field states are not ported |

CPU counts record actual RHS and analytic Jv calls, Newton iterations, direct
linear solves or GMRES iterations. GPU counts record actual RHS calls, Newton
iterations and direct solves. These counts have different costs across backends.
GPU wall time includes batch submission/readback and excludes pipeline compilation;
convergence batch timings are not individual per-job timings. No speedup is inferred
from these tiny systems. Eleven source-declared explicit FSAL tableaus reuse accepted endpoint derivatives
on CPU and GPU. CPU trial memory is immutable, so rejected adaptive trials cannot
replace the accepted cache. Implicit FSAL and other cache policies remain open.

## Coefficient regeneration and validation

Use an environment containing this library's Python dependencies:

```bash
python web/tools/export_catalog.py
python web/tools/export_catalog.py --check
python web/tools/export_reference.py
```

The exporter reads the registered definitions and parameterized coefficient
factories directly. Orthogonal source recurrences are expanded using formal
independent RHS vectors. No method is replaced by a different scheme under its
original name. The JSON is deterministic and the check detects source drift.
The reference exporter executes Python drivers, including warm history, to create
560 independent fixtures plus 1,872 accepted NDF/IRKN endpoints covering fixed
steps, resampling, rapid caller changes and explicit restarts.

`npm test` checks all CPU configurations, fixture parity, cross-family and forced-time
orders, adaptive rejection, derivative/adjoint actions, Newton–GMRES, SSP stages,
nonlinear geometry, structural damping, NDF/IRKN history, explicit FSAL reuse
and stable exponential actions. Optional
Dawn tests compile all eligible shaders and execute over 340 cross-problem cases:

```bash
npm install --prefix /tmp/integrator-dawn --no-save webgpu
DAWN_NODE_MODULE=/tmp/integrator-dawn/node_modules/webgpu/index.js npm test
```

The default executing backend is Vulkan (`DAWN_BACKEND` can override it).
Without the Dawn binding, these optional compiler/execution tests are explicitly
skipped. A null adapter is used only for compiler tests, never numerical evidence.
No Python core code is changed by this browser port.
