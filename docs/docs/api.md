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

## State and fields

| Name | Signature (abridged) | Purpose |
|---|---|---|
| `BaseState` | `@dataclass class BaseState` | Base for state classes; fields are annotated with a behavior decorator |
| `integrated` | `integrated(update_key, fluid_only=True, ...)` | Field advanced by the scheme; `update_key` names the derivative on the update structure |
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
| `IntegrationSchemes` | The registry: every registered scheme (64), each a `IntegrationScheme` |
| `IntegrationScheme` | The registry record: `function`, `name`, `identifier`, `order`, `dissipation`, `reuse_order`, `fsal`, `implicit`, `steps`, `stiffly_accurate`, `stability`, `startup_order` |
| `IntegrationSchemeType` | Per-scheme identifier enum (the `getIntegrator` key) |
| `FAMILY_ENUMS` / `SCHEME_FAMILY` | The eleven per-family enums, and the scheme → family map |
| `getIntegrator` | `getIntegrator(integrator)` — look up a scheme by `IntegrationSchemeType` or family-enum member |
| `getIntegrationEnum` | `getIntegrationEnum(integrator)` — the reverse lookup |
| `getPreferredScheme` | `getPreferredScheme(order)` — a reasonable default for an order |
| `step_reuse_analysis` / `step_reuse_order` | `step_reuse_order(scheme) -> int | None` — order retained under first-stage reuse |
| `supports_step_reuse` / `is_fsal` | `is_fsal(scheme) -> bool` — whether reuse is exact (FSAL) |
| `unpack_prior_step` | `unpack_prior_step(priorStep, verbose=False)` — prepare a previous `IntegrationResult` for `priorStep=` |

Every scheme is also callable directly with the uniform signature
`scheme(state, dt, f, *args, **kwargs) -> IntegrationResult` (e.g.
`BDF3(state, dt, f)`); implicit schemes additionally accept `solver=` (a
`NonlinearSolver`, default `JFNKSolver()`), and the multistep schemes thread
their history through the `StepHistory` on the result.

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

The helpers are scheme-agnostic; multistep schemes refuse adaptive stepping
explicitly (see the [solver](solver) page for the rationale).
