# Scheme expansion plan — OrdinaryDiffEq gap (staged by family and complexity)

**Status: waves 1–8 committed; Phase 8 implements the first wave-9 batch
(three-stage Gauss/Radau) and a parameterized classical generalized-alpha driver.
Parameterized Gauss/Radau stages 2–9, variable Adams, MEBDF2/CNLF2, stabilized
Legendre/Gegenbauer, fourteen generalized Nyström, fixed-order NDF 1–5 and
adaptive NDF target-order, explicit extrapolation and owned algorithm switching
and direct state-map backends are implemented
with source/numerical/consumer gates; remaining waves 10–16 members retain theirs.**
The registry contains 142 schemes (140 before MEBDF2/CNLF2; 139 before fixed ABM5; 137 before the first
wave-9 batch; 64 at planning time).
The current execution gates are in [REPOSITORY_PLAN.md](../../REPOSITORY_PLAN.md).

Scope: the schemes present in local **OrdinaryDiffEq v7.8.1** but missing from this
library, staged into 16 waves ordered by complexity. Waves 1–5 cover the SSPRK gap;
waves 6–16 cover the rest of the ODE catalog. Every wave is independently shippable
(reviewed backend batches with full suite green). The current user direction is
to continue through the backend backlog between those batches.

## Reconciled dispositions

This is a source/design inventory. [The execution plan](../../REPOSITORY_PLAN.md)
owns scheduling and [the method ledger](method-dispositions.md) owns dispositions.
Waves 1–8 are implemented with these explicit exceptions:

- pRRK22/33/54 are standalone `kappa`/`dt`-dependent drivers, not fixed-tableau factories.
- Anas5 and FRK65 are deferred pending runtime parameter/step contracts.
- PFRK87 is deferred: the pinned `perform_step!` computes `nu=omega*dt` and
  runtime phase-fitted coefficients. Its fixed cache is insufficient for the method.
- The four ExplicitTableaus files are enumerated in the ledger; extra constructors
  remain awaiting equivalence/order/usefulness verification. Wave 7 completion
  refers to its eleven implemented methods, not every source-data candidate.
- SSPRK932, DP8 and Feagin methods ship fixed-step without their source estimates;
  PSRK members are plain fixed-step. Estimator exceptions remain in numerical records.
- SplitEuler is forward Euler on the combined `f1+f2` RHS (the old Symplectic Euler
  equivalence statement was incorrect). ARS222/232 share their
  stored implicit tableau; SDIRK22 is excluded because its
  algorithm constructor returns Trapezoid, despite a separately stored compact tableau.
- The historical wave 9 list mixed stage configurations with order-based Radau names; upstream order/stage
  conventions must be resolved before implementation and scaling gates.

## Source of truth (citation chain)

- Coefficients: `~/.julia/packages/OrdinaryDiffEq/OnlvZ/lib/<Pkg>/src/*_caches.jl`
  (`*ConstantCache` constructors) and the tableau data in
  `lib/OrdinaryDiffEq{Explicit,Implicit}Tableaus/src/tableaus_*.jl`.
- SSP coefficients: `lib/OrdinaryDiffEqSSPRK/src/alg_utils.jl` (`ssp_coefficient`).
- Paper citations: the `@doc ... references` blocks in each package's `algorithms.jl`.
- All member lists below were verified by struct grep of the local source on
  2026-10-12 (julia 1.13.1). Per-scheme structure (stage counts, embedded-pair
  shape, FSAL-ness, exact citations) is verified against the same source at
  implementation time — never transcribed from memory.

## Historical per-scheme checklist (use CONTRIBUTING for current gates)

1. `butcher.py`: `getButcherTableau` branch + `butcherScheme(...)` (citation comment:
   ODE file + paper). Embedded pairs: `b` as `(b_main, b_embedded)` tuple.
2. `enums.py`: `IntegrationSchemeType` member + family-enum member.
3. `integration.py`: import + `IntegrationSchemes.append(...)`.
4. `tests/test_tvd.py`: pin the SSP coefficient (SSP-family schemes) against the ODE
   `alg_utils.jl` value at the existing 2e-3 tolerance; `mu_max=` override when C
   exceeds the scan cap (SSPRK104/SSPRK932 precedent).
5. `benchmarks/tvd_classifier.py`: measure verdicts; expectations live in `tests/test_tvd.py`.
6. Docs: `docs/docs/families/explicit-rk.md` (or the matching family page) section +
   SSP/TVD landscape row; scheme count in `docs/docs/index.md`; `CHANGELOG.md`.

**Automatic, no per-scheme code:** the `scheme` fixture (`tests/conftest.py`) picks up
every registered scheme for `test_convergence.py` (order on oscillator/forced/kepler)
and the registry-wide TVD classifier; the reuse guard / `step_reuse_analysis` apply
automatically. Embedded pairs get `tests/test_embedded.py`-style rate tests and
`adaptive.py` works on them for free (`estimate_error_norm`/`propose_dt`).

## Complexity classes

- **C1** plain tableau on the existing `RungeKuttaB` driver — transcribe + register.
- **C2** embedded pair on the existing `b`-tuple mechanism (+ rate test, adaptive smoke).
- **C3** tableau on an *existing implicit* driver (DIRK-JFNK, FIRK-block, BDF-JFNK,
  RKC, Rosenbrock, exponential-phi-Krylov) — transcribe + driver-specific gate.
- **C4** new small driver (multistep-SSP, RKN, Nordsieck, IRKC, RKIP, extrapolation,
  PDIRK, FunctionMap, Composite) — design + driver + tableaus.
- **C5** new interface / large new driver (matrix-A protocol, split-dt state,
  JVP-Taylor coefficients) — protocol design first.
- **C6** screening wave (verify algebraic redundancy before registering).

## Waves 1–5 — the SSPRK gap (18 methods, `lib/OrdinaryDiffEqSSPRK`)

### Wave 1 — plain SSP tableaus (C1, 7 methods)

SSPRK53 (5/3, C=2.65), SSPRK63 (6/3, C=3.518), SSPRK73 (7/3, C=4.2879),
SSPRK83 (8/3, C=5.107), SSPRK54 (5/4, C=1.508), KYKSSPRK42 (4/2, C=2.459),
KYK2014DGSSPRK_3S2 (3/2, C=0.8417).

**Recorded gate:** measured order == registered order (automatic); retain measured
`convex_combination_cfl` and document differences from ODE C instead of assuming equality; per-step TVD sweep; real-axis stability interval recorded in the
`butcher.py` comment (SSPRK104 precedent, cross-checked against ODE).

### Wave 2 — embedded/adaptive SSP pairs (C2, 3 methods)

SSPRK43 (4-stage 3(2), C=2), SSPRK432 (low-storage 3(2), C=2), SSPRK932 (9-stage
3(2), C=6).

**Gate:** estimate-rate test (3(2) pair → estimate O(h³), standard `propose_dt`
exponent); adaptive-loop smoke test; SSPRK932 pinned with explicit `mu_max` (C=6
exceeds the scan cap); confirm the reuse guard rejects `priorStep` (non-FSAL).

### Wave 3 — multistep SSP (C4, 2 methods, new small driver)

SSPRKMSVS32 (3-step, order 2, C=0.5), SSPRKMSVS43 (4-step, order 3, C=0.33) — Shu's
multistep SSP (JCP 169 (2001)). Stages are convex combinations of past **states** +
stage derivatives → rides the existing `HistoryEntry.state` snapshot mechanism (BDF
already pushes state snapshots). Cold start: Dormand-Prince bootstrap pattern with
`startup_order`. Fixed-step only (no estimate → adaptive refused by construction).

**Gate:** order at constant dt; restart-on-dt-change (free from `StepHistory.pushed`);
TVD classifier verdict.

### Wave 4 — parametric-relaxation pRRK (C1 + factory, 3 methods)

pRRK22 (C=1), pRRK33 (C=1), pRRK54 (C=1.508) — Liu, Burchard, Kuzmin & Shang 2023.
Free `κ` parameter (ODE: `kappa` field on the alg struct); `κ=0` reduces exactly to
SSPRK22/33/54. Mechanism: the `generic2nd`/`generic3rd` parameterized-tableau
precedent — `pRRK33(kappa=...)` factory returning `butcherScheme('pRRK33', kappa=κ)`.

**Registry decision (flagged, not assumed):** at `κ=0` they duplicate existing
schemes, so their value is only at `κ≠0`, where there is no single fixed
order/identifier. Proposal: expose the factories alongside (not inside)
`IntegrationSchemes`; tests assert (a) exact `κ→0` reduction and (b) the published
stability-region enlargement at `κ>0` on the advection model. TVD measured ad hoc
via `convex_combination_cfl` on the κ-tableau (like the `TVD_RK*_TABLEAU` constants).

### Wave 5 — low-storage SSP screening (C6, 3 methods)

SSPRK53_2N1 (C=2.18), SSPRK53_2N2 (C=2.148), SSPRK53_H (C=2.65) — Higueras–Roldán.
First verify whether each is algebraically identical to SSPRK53 as a one-step map
(same `y^{n+1}` for same `y^n`). If identical: redundant in this driver (our
`RungeKuttaB` allocates a fresh stage state anyway, so low storage buys nothing)
→ document, skip registration. If distinct → register via the standard checklist.
Deliverable is a pinned verdict either way.

## Waves 6–7 — explicit RK tableaus (C1/C2, ~33 methods)

### Wave 6 — order ≤ 5 workhorses (~19 methods, `OrdinaryDiffEqLowOrderRK` + `OrdinaryDiffEqTsit5`)

Tsit5 (5(4), 6 stages — **ODE's default scheme, top priority in this wave**),
OwrenZen3/4/5 (embedded pairs), Stepanov5, BS5, Alshina2/3/6, Anas5, Ralston4, RKM,
MSRK5/6, SIR54, FRK65, RKO65, PSRK3p5q4/3p6q5/4p7q6 (embedded; name pattern =
propagated-order/stages/embedded-order — verify structure from source), SplitEuler.

**Notes (corrected 2026-10-14):** upstream `split_perform_step.jl` evaluates
`u = uprev + dt*(f1+f2)`. SplitEuler duplicates Forward Euler on a combined RHS,
not the registered `symplecticEuler`; no additional method is registered. The
remaining LowOrderRK members (Euler, Heun, Midpoint, Ralston, RK4, DP5, BS3) are
already ours. **Gate:** standard checklist + embedded-rate tests for the pairs.

**Done 2026-10-13 — 18 registered (registry 79 → 97), one bundled commit.**
Anas5(w) and FRK65(omega) deferred: parameter-dependent tableaus (a free
w / omega trades SSP against accuracy) — the pRRK situation, not fixed
tableaus. SplitEuler was not registered; Phase 3 corrected its historical
equivalence claim to Forward Euler on a combined RHS. The PSRK* names read as propagated-order/stages/
embedded-order, but none is actually embedded — plain fixed-step methods.
RKO65's first stage is free (a₁ = b₁ = 0: no order condition involves c₁)
and its reuse is lossless ((bᵀa)₁ = 0) — first consumer of the new gain-3
branch in `reuse.tableau_reuse_analysis`. RKM registers at its formal order
3 (pseudospectral stability design). NOTES.md §3.26.

### Wave 7 — order 6+ high-order tableaus (~12+ methods)

Vern6, Vern7, Vern8, Vern9, RKV76IIa (`OrdinaryDiffEqVerner`), DP8, PFRK87, TanYam7,
TsitPap8 (`OrdinaryDiffEqHighOrderRK`), Feagin10/12/14 (`OrdinaryDiffEqFeagin`), plus
whatever high-order tableaus the data package
`lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order{5,6,7,8_9}.jl` adds
(enumerate at implementation time).

**Gate:** standard checklist; **CI note:** high-stage methods (Feagin14, Verner9,
DP8) are the first real cost on the convergence shard — measure and rebalance after
this wave.

## Waves 8–9 — implicit tableaus on existing drivers (C3, ~31 methods)

### Wave 8 — DIRK (`OrdinaryDiffEqSDIRK`, ~25 tableaus on the existing JFNK DIRK driver)

Sub-batch A (L-stable stiffly-accurate — the stiff workhorses): KenCarp3, KenCarp4,
KenCarp47, KenCarp5, KenCarp58, Kvaerno3, Kvaerno4, Kvaerno5, ESDIRK325L2SA,
ESDIRK436L2SA2, ESDIRK437L2SA, ESDIRK547L2SA2, ESDIRK54I8L2SA, ESDIRK659L2SA,
SSPDIRK2 (SSP-SDIRK2).
Sub-batch B: ARS222, ARS232, ARS343, ARS443, BHR553, Cash4, CFNLIRK3, Hairer4,
Hairer42, SDIRK22, SFSDIRK4/5/6/7/8.

**Gate:** L-stability of the sub-A set on the stiff oscillator (existing
`stability` machinery); A(alpha) cone checks where applicable; JFNK convergence on
the semilinear test problems. Note: ODE's ESDIRK436L2SA2 is a *different* method
from our ESDIRK4(3)6L[2]SA — transcribe side by side, do not assume identity.

**Status 2026-10-14: implementation, numerical validation and documentation
committed with the Phase 1 gate; wave not yet released.**
29 of the 30 tableaus registered (registry 108 -> 137, DIRK 7 -> 36); SDIRK22 is
deliberately absent — its ODE constructor is a stub (`return Trapezoid(...)`,
algorithms.jl:314-330) with orphaned tableau fields, so there is nothing to
transcribe. The bundled changes include: `dirk.py` (29 tableaus),
`enums.py`, `integration.py`, `__init__.py`, and
`tests/{test_dirk,test_state,test_embedded,test_stiff,test_family_enums,conftest}.py`.

What worked (fix phase, after the first full run's 18 failures):

- ODE's stored b weights sum to 1 + up to 7.8e-12 (Cash4, SFSDIRK5/7/8): nudging
  only the LAST b weight makes sum(b) == 1 exactly, with no a/c or metadata
  ripple (pinned in the dirk.py block comments).
- The JFNK DIRK families sit on a higher effective roundoff floor than the
  explicit schemes: per-(scheme, problem) coarser grids in conftest
  `PROBLEM_COARSE_GRIDS` (ESDIRK6(5)9L[2]SA on osc/forced; Cash4, SFSDIRK5/7/8
  on two-to-three problems each) keep the measured order in band.
- ESDIRK4(3)7L[2]SA on `forced` is superconvergent (measures 5.0 on every grid
  above the floor) — no grid satisfies the two-sided order test, so it gets a
  one-sided exception cell (the BS5 precedent).
- Total f-call counting is unreliable for the reuse test (JFNK iteration noise
  shifts per-step counts by -7..+3): counting evaluations AT the step-start time
  is deterministic (no-reuse 20, reuse 1, saved exactly 19).
- `_stage_times` now collapses consecutive repeated stage times: the JFNK
  iteration count varies between steps; the stage-time pattern is what is
  invariant.
- ODE stores EXTRA stages vs the published forms for 12 methods (an FSAL-tail row
  at c = 1 and/or null stages: Kvaerno3/4/5, KenCarp3/4/5, ARS222/443, Cash4,
  CFNLIRK3, Hairer4/42): the registered stage counts are ODE's stored counts;
  R(z) is unchanged by the extras (identity verified).

Findings (pinned in NOTES.md 3.28 + tests): ODE stored-c execution bug (5
methods store c != row sums; we register row-sum c); stability 22 L-stable /
2 A-not-L (Kvaerno4, SSPSDIRK2) / 5 not A-stable (SFSDIRK4-8 have a bounded
real-axis region only); reuse 20 lossless / 9 refused; 17 embedded (p, p-1) pairs
(3 need the one-sided band); orders per ODE `alg_order` (ARS443 and BHR553 are
order 3; SFSDIRK5-8 are ALL order 4).

Validation and documentation gate completed during repository Phase 1:

- Corrected stage counts, Cash4 stage timing, the SSPSDIRK2 stiff-limit sign
  and geometric classification in NOTES §3.28; retained coefficient corrections,
  upstream differences, solver-floor and superconvergence evidence.
- Added the changelog entry and expanded the DIRK catalog from 7 to 36;
  refreshed overview/API/selection/README and site counts to 137.
- Recorded all 29 TVD verdicts and separated completeness from per-scheme sweeps.
- Added metadata-driven adaptive and implicit/explicit gradient coverage,
  36 DIRK RHS-parameter checks, semilinear Burgers and stiff-relaxation checks.
- Resolved the audited Kepler failures with strict-solve proofs and justified
  default grids. Exact FSAL and linear geometry checks use strict stage solves
  without widening their tolerances; default-solve property tests remain active.
- Full CPU Python gate: **6,459 passed, 605 skipped, zero failures**, 270.23s
  with eight work-stealing workers. Docusaurus production build passes.
  Final CI shard timings and partition evidence are retained in the
  [Phase 1 validation record](../development/phase1-validation.md).

Landing: one bundled wave 8 / Phase 1 commit. The numerical evidence remains
in NOTES §3.28 and the Phase 1 validation record.
Package metadata remains 0.6.1; these changes are unreleased.

<a id="wave-9--firk-block-tableaus-ordinarydiffeqfirk-seven-listed-stage-configurations"></a>

### Wave 9 — FIRK tableaus — first s=3 batch implemented

The original list mixed stage counts and upstream propagated-order names. Gauss
s=3/4/5 means orders 6/8/10. Upstream RadauIIA3 is order 3, s=2 (already ours);
RadauIIA5 is order 5, s=3; RadauIIA9 is order 9, s=5. Radau stages 4/9 would
mean orders 7/17 and cannot be inferred from those algorithm names. Re-enumerate
the pinned source before any future transcription. The current
[primary documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/implicit/FIRK/)
confirms order-based Radau names; this is not a new v7.8.1 coefficient oracle.

**Hard gate failed in repository Phase 6.** Three-stage Gauss/Radau prototypes
verify collocation, propagated orders 6/5, stability, Gauss symmetry/symplectic
identities and state/parameter gradients. They intentionally carry no companion
estimate. The 114-configuration CPU sweep shows a local stage-block preconditioner
helps reaction problems but exceeds the Krylov work bar at diffusion N=256 and
fails tolerance at N=1024. The registry stays at 137. See
[the measurement and gate decision](../development/phase6-validation.md).

The [spatial follow-up](../development/spatial-followup-validation.md) now clears
the same work, memory, failure and time bar on the recorded 1D spatial cases.
Generic SPH assembly remains application-specific. Registration
then still requires companions/estimate rates, coefficient/source validation and
complete numerical/gradient gates. No Lobatto addition is implied; its partitioned
interface remains a separate design.

Phase 8 clears these remaining gates for **Gauss-Legendre 3 (order 6)** and
**Radau IIA s=3 (order 5)**. The complete pinned monorepo tree is checked by the
Julia source oracle. Gauss carries the source same-node order-2 companion;
Radau uses the explicitly derived unsmoothed order-3 companion, not the source's
filtered adaptive controller. Actual registered-driver work includes companion
calls and passes the same spatial gate. See
[the first-batch record](../development/phase8-wave9-validation.md).
The subsequent `Collocation(family, stages)` backend supplies stage counts 2–9,
with a pinned coefficient/nonlinear-step oracle, order/stability identities,
derived/source companions and a Schur sparse-space scale gate. These are
parameterized factories outside the four fixed records; [evidence](../development/higher-collocation-validation.md).
Variable-order source controllers and source filtering remain separate driver
work. Frontend availability is not a gate for implementing those interfaces.

## Wave 10 — multistep extensions (C3/C4, ~16 methods)

Implemented backend batch: `VariableAdams(order, kind)` covers fixed-order
variable-step AB/ABM at orders 3–5, with a pinned coefficient oracle, ramped
startup estimates, transactional adaptive runs and portable history. Classical
fixed AB5/AM5 PECE is registered as `abm5`, with an explicit repository startup/
cache policy rather than a full upstream ABM54 equivalence claim. The fixed
catalog is now 140. [Evidence](../development/variable-adams-validation.md).
MEBDF2 now uses its equivalent one-step SDIRK form; CNLF2 supplies the source
IMEX-Euler start and Crank–Nicolson–Leapfrog history formula. Both have nonlinear
step-body oracles, ownership/gradient/restart tests and known-consumer measurements;
the fixed catalog is 142. [Evidence](../development/implicit-extensions-validation.md).
Fixed-target-order NDF 1–5 now supplies source-checked differences, resampling,
implicit startup, accepted quasi-constant blocks and portable restart.
[Evidence](../development/ndf-backend-validation.md). AdaptiveNDF adds accepted
target-order selection with source-checked candidate mathematics and explicit
package maturity/startup/retry policies; [evidence](../development/adaptive-ndf-validation.md).
AdaptiveAdams adds bounded accepted target orders 3–5 over PECE, with source
neighbor-defect/selection oracles, owned history and known reaction/rotation-heat
consumers; [evidence](../development/adaptive-adams-validation.md). It does not
implement the full VCABM order-1–12/FSAL/startup/controller policy.
Full upstream QNDF retry/discontinuity semantics, Nordsieck/BDF variants and AMF remain separate backend work; frontend availability
is not an implementation gate.

- `bdf.py`: NordsieckBDF (Nordsieck-form BDF), QNDF / QNDF1 / QNDF2
  (quasi-constant-step NDF — inspect order/history semantics), MEBDF2 (modified extended BDF).
- `multistep.py`: ABM54 (fifth-order predictor/corrector, **not** a source 5(4)
  embedded/adaptive pair); VCAB3/4/5 and VCABM/VCABM3/4/5 (variable-step history
  coefficients — fixed-step AB/ABM also support non-autonomous RHS; inspect
  timestep ratios, history and controller state before
  transcribing).
- New small Nordsieck driver: AN5, JVODE (ODEPACK's classic).
- `imexmultistep.py`: CNLF2 (Crank–Nicolson–Leapfrog 2).
- AMF (`OrdinaryDiffEqAMF`, amf.jl + linsolve.jl): **Approximate Matrix
  Factorization**, a Rosenbrock-W wrapper with structured split factors. It needs
  operator/factorization contracts and is not an Adams–Moulton extension.

**Gate:** order at constant dt; cold-start correctness; any estimator must be
derived and validated separately; step/history/order transitions for variable-step
drivers. The source declaration inventory also includes FBDF/ABDF2 and DAE-specific
variants; they are not interchangeable with fixed-coefficient BDF tableaus.

## Wave 11 — stabilized / Krylov extensions (C3, ~16 methods)

The `StabilizedRK` factory supplies RKL1/2 and RKG1/2 with source step-body
oracles, consistent nonautonomous stage times, conservative spectral sizing,
unscaled Euler difference control, owned lifecycle/quantity handling and portable
restart. These are parameterized backends outside the 142 fixed registrations;
[validation](../development/stabilized-backend-validation.md). ROCK/SERK/RKMC and
the linearized/exponential drivers retain separate source/driver/cost gates.

The `Rosenbrock` factory now supplies Rosenbrock23/32 with a qualified correction
to the source third-stage time term, plus source W6S4OS. Full-JVP stages retain
Hessian/RHS-parameter derivatives, coupled quantities and isolated probes;
source, stability, adaptive, ownership and consumer evidence is in
[the linearized validation](../development/linearized-validation.md). The pair's
original nonautonomous path is recorded separately. W6S4OS and Rosenbrock32 are
not A-stable; a small stiff limit alone is not an L-stability certificate.

OrthogonalRK now supplies ROCK2/4, SERK2 and RKMC2 through translated source
recurrences, consistent clocks and owned stage quadratures. High-count source
stability exceptions are retained individually; 1,103 supported configurations
have numerical spectral screening. Known heat modes expose both force savings
and unresolved-fast-mode damping limits; [evidence](../development/orthogonal-validation.md).
The remaining wave-11 scope is exponential and approximate-factorization work.

- `rkc.py` family: ROCK2, ROCK4 (Rodriguez regularized-Chebyshev), SERK2 (stabilized
  exponential RK 2), RKG1, RKG2, RKL1, RKMC2 (verify each method's construction from
  source — several are variants our driver does not currently parameterize).
- `rosenbrock.py`: Rosenbrock23, Rosenbrock32 (tableau forms), RosenbrockW6S4OS,
  Tsit5DA (a HybridExplicitImplicitRK instance for mass-matrix DAEs, not a plain
  stiff fifth-order Rosenbrock tableau), HybridExplicitImplicitRK (new driver).
- `exponential.py`: the remaining ODE exponential-RK members (ETDRK2 / EXPRB53
  candidates — macro-generated source members include ETDRK2/3/4, HochOst4,
  Exp4, EPIRK4/5, EXPRB53s3 and Exprb32/43; explicit struct grep alone missed them).
- `generalized_alpha.py`: classical second-order GeneralizedAlpha (Chung–Hulbert).
  Implemented as a parameterized fixed-step factory, separate from the fixed
  catalog; [consumer and driver evidence](../development/generalized-alpha-validation.md).
  The pinned Newmark constructor's parameter and acceleration-refresh issues
  are recorded in the independent structural source audit.

**Gate:** real-axis stability interval per method (the selling point of this family);
semilinear split tests; GMRES cost accounting where a solve is involved.

The [reproducible Phase 8 source inventory](../development/evidence/phase8-source-inventory.json)
records all nine package versions, declarations/order expressions and source hashes
for waves 10–11. These are inventories, not coefficient/equivalence approvals.
GeneralizedAlpha now has weighted second-order residuals and a parameter/memory contract;
stabilized methods need recurrences/stage selection, and exponential methods need
phi/Krylov formulas. None is assumed to be a plain-tableau transcription.

## Wave 12 — RKN + second-order-ODE family (C4, ~17 methods, new general RKN driver)

A general Runge–Kutta–Nystrom driver (a, b, b* tableaus on the position/velocity
form) replaces the hand-rolled `Nystrom5th` as the family driver:
RKN4, Nystrom4 (+ Nystrom4VelocityIndependent), DPRKN12, DPRKN4, DPRKN5, DPRKN6,
DPRKN6FM, DPRKN8, FineRKN4, FineRKN5, ERKN4/5/7 (source embedded/trigonometrically
designed explicit RKN tableaus; the pinned implementations use constant RKN
coefficients, not a phi-Krylov driver), IRKN3/4 (**improved**, fixed-step
velocity-independent RKN with history, not implicit JFNK). Plus **PDIRK44**
(`OrdinaryDiffEqPDIRK`): **parallel** DIRK for ordinary first-order ODEs, with
two independent pairs of implicit stages. The earlier partitioned-mechanics
classification was incorrect. ParallelDIRK now executes the source map through
the serial DIRK driver; [validation](../development/parallel-dirk-validation.md).
No internal-threading implementation or performance claim is implied.

The [IRKN source-path audit](../development/irkn-source-audit.md) executes actual
mutable kernels and records unusable constant-cache paths. Source time-dependent
startup only achieves order two; corrected-history controls isolate the cache
seed problem. ImprovedNystrom now implements the corrected seed and source warm
recurrences, with nonlinear/time-dependent order, owned history, gradients,
lifecycle and known-consumer force measurements;
[implementation evidence](../development/improved-nystrom-validation.md).

**Gate:** order on the oscillator/kepler second-order problems; Nystrom5th
regression (the new driver must reproduce the existing hand-rolled method's order);
history/startup/order for improved IRKN; source/order/stability/solve and consumer
gates for PDIRK44. Stage-threading cost belongs to the execution backend.

## Wave 13 — new medium drivers (C4, ~11 methods)

`Extrapolation(base, order)` supplies fixed-target Euler orders 2–8 and even
midpoint orders 2–10 with source fixed-column maps, estimator/quantity/lifecycle
contracts, adaptive timestep and physical composition. Source adaptive-order/work
controllers and internal threading remain distinct. [Evidence](../development/extrapolation-validation.md).

- Extrapolation family (`OrdinaryDiffEqExtrapolation`, 7 methods): AitkenNeville,
  ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
  ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
  ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation — Richardson/
  extrapolation tables over a base method; classical math, new driver.
- RKIP (`OrdinaryDiffEqRKIP`): Runge–Kutta in the interaction picture, requiring
  forward/backward exponential actions; the original Rosenbrock classification
  was incorrect.
- IRKC (`OrdinaryDiffEqStabilizedIRK`): stabilized implicit RK via Chebyshev
  polynomials — performs nonlinear implicit split stages. The pinned constant-cache map fails
  the retained order-two control and is excluded from production promotion.
- FunctionMap (`OrdinaryDiffEqFunctionMap`): direct discrete state map, evaluated
  at t+dt in the inspected step body; scale_by_time adds dt*f to the old state.
  FunctionMap/StateMap now implements the explicit map contract with separate
  result transition metadata, coupled endpoint lifecycle and portable restart;
  the scaled branch is an explicit map wrapper. [Evidence](../development/function-map-validation.md).
- Composite (`OrdinaryDiffEqCore.CompositeAlgorithm`): **algorithm switching**
  with choice/cache transitions, not the physical flow composition implemented
  in Phase 7. AlgorithmSwitch now supplies pure accepted-selection state, owned
  active-child history, cold-start transitions, delegated adaptive control and
  portable tensor restart; source cache-transfer/controller semantics remain
  distinct. [Evidence](../development/algorithm-switch-validation.md).

**Gate:** per-driver; extrapolation convergence-order tests on the standard problems;
IRKC stability-interval measurement.

## Wave 14 — new interfaces, large (C5, ~14 methods)

- Linear family (`OrdinaryDiffEqLinear`): CayleyEuler, LinearExponential,
  MagnusAdapt4 — all require the user to supply the system matrix A (a new RHS
  protocol member, e.g. `linear=`/`A=` on the RHS), plus eigendecomposition / Cayley /
  Magnus machinery. Protocol design first.
- Multirate (`OrdinaryDiffEqMultirate`): MIS, MRAB, MREEF, MRIGARKERK22a/b,
  MRIGARKERK33a, MRIGARKERK45a — components integrated at different rates; requires
  a split-dt state structure + split RHS. GARK (generalized additive RK) structure.
- MultirateImplicit (`OrdinaryDiffEqMultirateImplicit`): MREIL,
  MRIGARKESDIRK34a, MRIGARKESDIRK46a, MRIGARKIRK21a — same structure, implicit
  halves.

**Gate:** design review of the new protocols before implementation; correctness on
toy split problems; cost accounting (the whole point is per-component dt choice).

## Wave 15 — Taylor series (C5, 3 methods, most novel)

ExplicitTaylor, ExplicitTaylor2, ExplicitTaylorAdaptiveOrder
(`OrdinaryDiffEqTaylorSeries`): Taylor-coefficient integration via repeated JVP of
the RHS. In this stack that means Warp custom-gradient chains of depth = order —
expensive and genuinely new. **Design first:** cost model (JVPs per step vs RK
evaluations at comparable order), then a prototype at order 3–4 before committing to
the adaptive-order variant.

## Wave 16 — LowStorageRK screening (C6, ~60 methods)

`OrdinaryDiffEqLowStorageRK`: AlJahdali* (16), CarpenterKennedy2N54, CFRLDDRK64,
CKLLSRK* (15), DGLDDRK* (3), NDBLSRK* (3), ORK256,
ParsaniKetchesonDeconinck* (9), RDPK3Sp* (6), RK46NL, SHLDDRK* (2), TSLDDRK74.
All 60 named constructors are now screened using their actual source step bodies.
ExpandedRK exposes 54 main maps and 19 qualified companions outside the fixed
registry, using full-stage storage. Six main source methods and one companion
have individual deferred/excluded dispositions. Canonical A,b,c comparison finds
no matches against 60 fixed explicit tableaus; the earlier presumption that most
were redundant was not established. This screen alone does not prove inequivalence
under every representation. Native low-storage execution remains separate.
See [source and consumer validation](../development/expanded-rk-validation.md).

## Out of scope (documented in the family page, not added)

- SIMD RK variants (MER5v2, MER6v2, RK6v4) — performance re-implementations of known
  math; no new method, and Warp is already the SIMD substrate.
- StochasticDiffEq* (all variants), DelayDiffEq, GlobalDiffEq (Glee) — different
  problem classes.
- DAE variants (DABDF2, DFBDF, FBDF, DNordsieckBDF, DImplicitEuler) — this library is
  not a DAE library.
- ImplicitDiscreteSolve / SimpleImplicitDiscreteSolve — discrete-event machinery.

## Cross-cutting

- **One commit per wave**, full suite green at each wave (repo convention).
- **CI runtime:** the convergence + stability shards grow with every wave. Re-measure
  the six-shard balance after waves 6, 7, 9, 12 and rebalance per the sharding design
  (`.github/workflows/tests.yml`); consider a `slow`-style marker for the highest-stage
  convergence tests if a shard drifts past ~9 min.
- **TVD classifier scan cost:** grows with stage count (SSPRK83: 8, SSPRK932: 9,
  Feagin14: 14) — the `classify_all` test may need a CFL-grid cap bump, measured not
  guessed.
- **Naming:** adopt ODE's names verbatim for new schemes (SSPRK53, Vern8, Tsit5, ...);
  our two existing near-collisions stay (our `SSPRK3` = ODE `SSPRK33`, our
  `SSPRK104` = ODE `SSPRK104`).
- **Effort is not linear in method count:** waves 6–11 are mostly transcription
  (drivers exist); waves 12–15 are driver work and dominate the real cost; wave 16
  is screening. The ~200-method headline overstates the work: a realistic landing
  target is 120–150 registered schemes plus documented-redundant notes.
- **Memory:** the verified per-family catalog lives in project memory
  `ordinarydiffeq-full-catalog.md` (project memory dir), the SSPRK gap with exact
  SSP coefficients in `ordinarydiffeq-ssprk-gap.md`.

## Final candidate outcomes

The execution plan's Phase 8 candidate review is complete with explicitly bounded
production backends and individually qualified source dispositions. It does not
claim complete upstream controller equivalence. See the
[exponential evidence](../development/exponential-validation.md) and
[remaining backend decisions and prototypes](../development/backend-backlog-validation.md).
The latter includes 35 individual source outcomes, exact Nordsieck primitive
controls, an IRKC order counterexample, and operator/multirate/Taylor/storage
experiments. Deferred controller/operator projects retain concrete revisit
conditions and are independent of frontend availability.
