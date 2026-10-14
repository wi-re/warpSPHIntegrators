---
sidebar_position: 3
title: API Reference
---

# API Reference

Everything below is importable from the package root (`from
warpSPHIntegrators import ...`); the complete list is `__all__` in
`__init__.py`. The package ships `py.typed`, so type checkers see the
annotations. Signatures show the load-bearing parameters; `...` marks the
rest. The [Quick Start](quickstart) shows a full minimal example.

## Tensor ODE adapter

`TensorState(y0, t=0)` and `tensor_rhs(f, **parameters)` adapt ordinary `(t, y)`
callbacks. Read `result.state.y`; splits, history, estimates and gradients use the
existing drivers. See [the runnable tensor guide](tensor-ode).

## Run orchestration

`solve(scheme, state, t_end, rhs, dt=..., adaptive=False, ...)` returns a
`RunResult` with counters, samples, events, status and restart checkpoint.
`IntegrationResult.step_status` exposes single-step success and convergence.
`Event`, `RunError`, `RunCheckpoint`, `snapshot_state`, `save_checkpoint` and
`load_checkpoint` support controlled runs; see [the contracts and example](runs).

## Physical operator splitting

`ConservativeDissipativeRHS` declares autonomous physical parts. `LieTrotter`,
`Strang`, restricted `Yoshida4` and `compose` bind sub-integrators and options into a `SplitScheme`
accepted by `solve`. `LinearExponentialFlow` supports small dense linear flows;
`classify_separable` returns a `SeparabilityResult` with measured local diagnostics.
See [the runnable guide and qualifications](operator-splitting).

## Variable-step Adams

`VariableAdams(order=3, kind='abm', max_step_ratio=4.)` supplies variable-step
AB/ABM orders 3–5 for fixed/adaptive trajectories. `adams_weights(nodes)` exposes
normalized cardinal integration. `IntegrationResult.estimate_order` can override
the steady estimator exponent during startup. `StepHistory.context` holds
method-owned trajectory metadata; variable methods explicitly opt into
`pushed(..., preserve_dt=True, context=...)`. See [the Adams backend](families/multistep-explicit).

## Stage-parameterized collocation

`Collocation(family='gauss', stages=4)` binds a stage-parameterized block driver
for Gauss/Radau, with stage counts 2–9 and estimate-controlled `solve` support.
`collocation_tableau` exposes its immutable float64 coefficients.
`stage_schur_preconditioner` binds a Schur decomposition provider and a factory
for shifted solves plus a matching frozen Jacobian action. See
[the coupled backend guide](families/coupled-rk).

## Stabilized recurrence factories

`StabilizedRK('rkg2', stages=5)` or `StabilizedRK('rkl2', lambda_max=...)`
binds RKL1/2 or RKG1/2 for fixed/adaptive runs. `stabilized_coefficients`
returns immutable recurrence weights and consistent stage times;
`stabilized_interval` exposes conservative negative-real sizing bounds.
Endpoint refresh costs one RHS beyond the recurrence's stage count. See
[the contracts and executable example](families/relaxed-chebyshev).

## Second-order numerical damping

`GeneralizedAlpha(rho_inf=.5)` binds a classical fixed-step mechanics driver.
Pass the factory to `solve` with a tagged position/velocity system and explicit
constant-mass acceleration RHS. The runner carries accepted algorithmic
acceleration through shortened steps and in-memory checkpoints.
`GeneralizedAlphaMemory` is stored in `HistoryEntry.state`; direct stepping loops
must feed accepted `result.history` back into the next call. This parameterized
factory is separate from the fixed catalog. See [the consumer and scope](structural-demo).

## State and fields

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `BaseState` | `@dataclass class BaseState` | Base for state classes; fields are annotated with a behavior decorator |
| `integrated` | `integrated(update_key, *, default=..., ...)` | Field advanced by the scheme; `update_key` names the derivative on the update structure |
| `constant` / `copied` / `ephemeral` / `custom` | `constant(default=..., ...)`, ... | The other field behaviors (see the Quick Start's behavior table) |
| `tagged` | `tagged(*, tags=None, role=None, ...)` | Field on the update dataclass (no integration behavior) |
| `reference_state` | `reference_state(*, tags=None, ...)` | System field that references the current state |
| `BlockState` | `BlockState(states: tuple)` | State for coupled multi-component schemes (Gauss-Legendre, Radau IIA) |
| `get_reference_state` | `get_reference_state(system)` | Extract the state from a system |
| `find_tagged_field` | `find_tagged_field(instance_or_type, *, role=None, tag=None)` | Locate a field by tag/role |
| `get_tagged_attr` / `set_tagged_attr` | `get_tagged_attr(obj, *, role=None, tag=None)` | Read/write a tagged attribute |
| `register_clone_handler` | `register_clone_handler(name, matches, clone, to_device=None, is_buffer=True)` | Teach cloning a custom value type (e.g. `wp.array` is built in) |
| `clone_value` / `empty_value` / `move_value` | `clone_value(value, detach=False)` | Low-level clone / null / device-move |
| `state_difference` / `state_norm` | `state_norm(state, rtol=1e-3, atol=1e-6, *, reference=None)` | State distance (norm-based, with tolerances) |
| `integrated_field_names` | `integrated_field_names(state) -> list[str]` | Names of the integrated fields |
| `flatten_integrated` / `unflatten_integrated` | `flatten_integrated(state) -> torch.Tensor` | Flat-vector view of the integrated fields (what the JFNK matvec sees) |
| `replace_integrated_fields` | `replace_integrated_fields(template, replacements)` | Build a state with selected fields swapped |

Particle selection and boundary masking are handled by the user's system/RHS.
`integrated` supplies field/update metadata and does not apply a particle mask.
The unused `fluid_only` argument and automatic metadata entry were removed;
defaults and other options are keyword-only.

## Update specs

The per-stage update descriptor passed to the system's apply methods.

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `ComponentUpdateSpec` | `ComponentUpdateSpec(derivative_dt, blend=...)` | One derivative component's step descriptor |
| `PositionUpdateSpec` | `PositionUpdateSpec(derivative_dt, current_velocity_dt=None, update_velocity_dt=None, blend=...)` | Position step (carries the velocity blend) |
| `StateBlend` | `StateBlend(self_scale=None, reference_state=None, reference_weight=None)` | Blend of the new value with a reference state |
| `blend_state` | `blend_state(*, self_scale=None, reference_state=None, reference_weight=None)` | Constructor helper for `StateBlend` |
| `explicit_step` | `explicit_step(dt, *, blend=None)` | Spec factory for an explicit derivative |
| `semi_implicit_position_step` | `semi_implicit_position_step(dt, *, blend=None)` | Spec factory for a velocity-first position update |
| `verlet_position_step` | `verlet_position_step(derivative_dt, *, update_velocity_dt, blend=None)` | Spec factory for the velocity-Verlet position update |

## Integration system protocol

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `IntegrationSystem` | Protocol | The interface every scheme's driver calls into |
| `BaseIntegrationSystem` | `@dataclass class BaseIntegrationSystem` | Convenience base implementing the boilerplate |
| `applyStateUpdate` | `applyStateUpdate(systemState, systemUpdate, spec, **kwargs)` | Apply all updates (default calls the component updates) |
| `applyPositionUpdate` / `applyVelocityUpdate` / `applyQuantityUpdate` | `applyPositionUpdate(systemState, systemUpdate, spec, **kwargs)` | Per-component apply methods |
| `update_position` / `update_component` | `update_position(system, update, spec, position_tag, position_derivative_tag, velocity_tag, ...)` | Tag-driven default implementations (used by `BaseIntegrationSystem` subclasses) |
| `updateStateEuler` / `updateStateSemiImplicitEuler` | `updateStateEuler(systemState_, systemUpdate, dt, copyState=True, **kwargs)` | Low-level explicit/semi-implicit state updates |

## Right-hand side

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `RHS` | `RHS(*, combined=None, explicit=None, implicit=None, linear=None, nonlinear=None)` | The structured right-hand side; a plain callable is also a valid `f` |
| `IMEXRHS` | `IMEXRHS(explicit, implicit) -> RHS` | Additive split `f = f_E + f_I` (IMEX Euler, ARK3/4, SBDF2/3, CNAB2) |
| `SemilinearRHS` | `SemilinearRHS(linear, nonlinear=None, combined=None) -> RHS` | Semilinear split `f = L·y + N` (ETD2RK, EXPRB32) |
| `resolve` | `resolve(f, *, scheme_name, need_linear=False) -> ResolvedRHS` | Resolve `f` against a scheme's needs; fails early, naming the missing capability |
| `CAPABILITIES` | frozenset | The capability names a scheme may require |
| `check_contracts` | `check_contracts(rhs, sample_state, dt, *, rtol=1e-8, atol=1e-10, seed=0)` | Verify a split RHS actually equals its combined `f` |
| `add_updates` / `sub_updates` | `add_updates(a, b)` | Linear combinations of update structures |

## Schemes and the registry

| Name | Purpose |
|---|---|
| `IntegrationSchemes` | The registry: every registered scheme (142), each a `IntegrationScheme` |
| `IntegrationScheme` | The registry record: `function`, `name`, `identifier`, `order`, `dissipation`, `reuse_order`, `fsal`, `implicit`, `steps`, `stiffly_accurate`, `stability`, `startup_order` |
| `SchemeMetadata` / `scheme.metadata` | Frozen metadata derived from the driver's tableau: stages, companion order, estimate exponent, required/optional RHS accessors, per-step options and dense output |
| `scheme.has_error_estimate` / `scheme.estimate_order` | Whether a step emits an estimate, and its exponent q for `propose_dt`; q is 3 for both TR-BDF2 and Gauss-Legendre 2 |
| `scheme.stage_count` / `scheme.embedded_order` | Stored/known driver stages and companion order; `None` means a variable or unmodeled driver stage count |
| `scheme.required_capabilities` | Mandatory structured RHS accessors. Only ETD2RK requires `linear`; a combined plain callable is accepted elsewhere |
| `IntegrationSchemeType` | Per-scheme identifier enum (the `getIntegrator` key) |
| `FAMILY_ENUMS` / `SCHEME_FAMILY` | The eleven per-family enums, and the scheme → family map |
| `getIntegrator` | `getIntegrator(integrator)` — look up a scheme by `IntegrationSchemeType` or family-enum member |
| `getIntegrationEnum` | `getIntegrationEnum(integrator)` — the reverse lookup |
| `filter_schemes` | AND filters for family/order/estimates/dense output/reuse/state kind/history/separability/drift/RHS capabilities; returns catalog records |
| `getPreferredIntegrator` | Conventional explicit order-1–5 tensor-ODE defaults as registered records |
| `getPreferredScheme` | Legacy order-1–5 selector returning a raw function; use `getIntegrator` for a scheme record and its metadata |
| `step_reuse_analysis` / `step_reuse_order` | `step_reuse_order(scheme) -> int | None` — order retained under first-stage reuse |
| `supports_step_reuse` / `is_fsal` | `is_fsal(scheme) -> bool` — whether reuse is exact (FSAL) |
| `unpack_prior_step` | `unpack_prior_step(priorStep, verbose=False)` — prepare a previous `IntegrationResult` for `priorStep=` |

Every scheme is also callable directly with the uniform signature
`scheme(state, dt, f, *args, **kwargs) -> IntegrationResult` (e.g.
`BDF3(state, dt, f)`); implicit schemes additionally accept `solver=` (a
`NonlinearSolver`, default `JFNKSolver()`), and the multistep schemes thread
their history through the `StepHistory` on the result.

Scheme metadata is derived without adding fields to the existing scheme tuple.
`metadata.required_options` lists alternatives: RKC/RKL need `s=` **or**
`lambda_max=` on each step. `metadata.estimate_extra_stages` records the
coupled pair's null companion stage. Optional split accessors are recorded
separately from mandatory accessors; plain-callable IMEX stepping uses the
pure implicit limit. Dense-output support currently means the DP5 helper.

[example:run metadata]: #
```python
from warpSPHIntegrators import getIntegrator

scheme = getIntegrator('Gauss-Legendre 2')
assert scheme.order == 4
assert scheme.embedded_order == 2
assert scheme.estimate_order == 3
assert scheme.has_error_estimate
assert scheme.metadata.estimate_extra_stages == 1
```

## Results and history

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `IntegrationResult` | `IntegrationResult(state, stages=..., error=None, history=None, solver_diagnostics=None)` | What every scheme returns |
| `StageResult` | `StageResult(aux=None, update=None)` | One stage: the RHS aux values and the k-value |
| `StepEvaluation` | `StepEvaluation(update, aux=None)` | A stage evaluation before it is folded into a result |
| `StepHistory` | `StepHistory(maxlen, entries=())` | Bounded history for the multistep schemes |
| `HistoryEntry` | `HistoryEntry(t, dt, update, aux=None, state=None)` | One recorded step |

## Nonlinear solvers

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `NonlinearSolver` | Protocol | The interface implicit schemes call into |
| `JFNKSolver` | `JFNKSolver(matvec='fd', tol=1e-8, max_iterations=20, gmres_..., newton_..., line_search=False, preconditioner=None, preconditioning='right', ...)` | Jacobian-free Newton–Krylov solver (the default for implicit schemes) |
| `FixedPointSolver` / `RelaxedFixedPointSolver` | `FixedPointSolver(iterations=2)` | Plain / relaxed fixed-point iteration (cheap, for mild stages) |
| `gmres` | `gmres(matvec, b, x0=None, tol=1e-8, maxiter=None, restart=30, preconditioner=None, preconditioning='right')` | Restarted GMRES with left/right preconditioning |
| `fd_matvec` / `jvp_matvec` | `fd_matvec(step, Y, y_flat, G_y, eps=None)` | Finite-difference / JVP Jacobian–vector products |
| `identity_preconditioner` | `identity_preconditioner(v, state, context)` | The 3-arg preconditioner hook, trivially |
| `diagonal_preconditioner` | `diagonal_preconditioner(inv_diag)` | Per-DOF diagonal preconditioner factory |
| `SolveDiagnostics` | `SolveDiagnostics(residual=None, gmres_iterations=0, rhs_evaluations=0, termination='fixed_iterations', line_search_backtracks=0)` | Per-solve observability, surfaced on `IntegrationResult.solver_diagnostics` |
| `SolveResult` | `SolveResult(y, converged, iterations, diagnostics=None)` | A solve's outcome |

## Adaptive step control

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `estimate_error_norm` | `estimate_error_norm(result, rtol=1e-3, atol=1e-6) -> float` | Embedded-pair error norm from an `IntegrationResult` |
| `propose_dt` | `propose_dt(error_norm, dt, order, *, target=1.0, safety=0.9, growth_min=0.2, growth_max=5.0)` | Classic PI-style step proposal |
| `dormand_prince_dense_output` | `dormand_prince_dense_output(state, stages, dt, theta, *args, **kwargs)` | DP5 interpolant at `theta ∈ [0, 1]` |

The helpers are scheme-agnostic; constant-coefficient multistep schemes refuse
adaptive stepping. VariableAdams uses its per-step estimator exponent and
variable-history contract (see the [solver](solver) page).
