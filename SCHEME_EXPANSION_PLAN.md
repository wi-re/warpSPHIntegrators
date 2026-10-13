# Scheme expansion plan — OrdinaryDiffEq gap (staged by family and complexity)

**Status: waves 1–7 done (2026-10-13); waves 8–16 not started.** Registry at 108
schemes (64 at planning time).

Scope: the schemes present in local **OrdinaryDiffEq v7.8.1** but missing from this
library, staged into 16 waves ordered by complexity. Waves 1–5 cover the SSPRK gap;
waves 6–16 cover the rest of the ODE catalog. Every wave is independently shippable
(one bundled commit, full suite green) — stop at any wave boundary.

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

## Per-scheme checklist (applies to waves 1–14)

1. `butcher.py`: `getButcherTableau` branch + `butcherScheme(...)` (citation comment:
   ODE file + paper). Embedded pairs: `b` as `(b_main, b_embedded)` tuple.
2. `enums.py`: `IntegrationSchemeType` member + family-enum member.
3. `integration.py`: import + `IntegrationSchemes.append(...)`.
4. `tests/test_tvd.py`: pin the SSP coefficient (SSP-family schemes) against the ODE
   `alg_utils.jl` value at the existing 2e-3 tolerance; `mu_max=` override when C
   exceeds the scan cap (SSPRK104/SSPRK932 precedent).
5. `scripts/tvd_classifier.py`: refresh the maintained verdict table.
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

**Gate:** measured order == registered order (automatic); `convex_combination_cfl`
== ODE C (2e-3); per-step TVD sweep; real-axis stability interval recorded in the
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

**Notes:** SplitEuler is ODE's name for symplectic Euler as a splitting — already in
our registry (`symplecticEuler`); document the duplicate, do not register. The
remaining LowOrderRK members (Euler, Heun, Midpoint, Ralston, RK4, DP5, BS3) are
already ours. **Gate:** standard checklist + embedded-rate tests for the pairs.

**Done 2026-10-13 — 18 registered (registry 79 → 97), one bundled commit.**
Anas5(w) and FRK65(omega) deferred: parameter-dependent tableaus (a free
w / omega trades SSP against accuracy) — the pRRK situation, not fixed
tableaus. SplitEuler confirmed the `symplecticEuler` duplicate (documented,
not registered). The PSRK* names read as propagated-order/stages/
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

### Wave 9 — FIRK block tableaus (`OrdinaryDiffEqFIRK`, 6 tableaus on the existing block driver)

Gauss-Legendre s=3, 4, 5; Radau IIA s=3, 4, 5, 9.

**Gate (hard dependency):** the coupled-block s=3 scale proof from the
`IMPLICIT_ROADMAP` open items (s=2 gate = 256 DOF / 1536 unknowns). This wave does
not start until that proof lands. ODE v7.8.1 has **no Lobatto III A/B** (verified —
FIRK = AdaptiveRadau, GaussLegendre, RadauIIA3/5/9), so there is no Lobatto gap to
fill. Symplecticity of odd-s Gauss-Legendre for separable Hamiltonians: extend the
`tests/test_hamiltonian.py` measurement per s.

## Wave 10 — multistep extensions (C3/C4, ~16 methods)

- `bdf.py`: NordsieckBDF (Nordsieck-form BDF), QNDF / QNDF1 / QNDF2 (quasi-Newton
  NDF — fewer implicit solves per step), MEBDF2 (modified-energy BDF).
- `multistep.py`: ABM54 (5(4) embedded multistep → adaptive-capable); VCAB3/4/5 and
  VCABM/VCABM3/4/5 (variable-coefficient variants for non-autonomous problems —
  verify the history-structure difference from our autonomous AB/ABM before
  transcribing).
- New small Nordsieck driver: AN5, JVODE (ODEPACK's classic).
- `imexmultistep.py`: CNLF2 (Crank–Nicolson–Lefort–Fortunato 2).
- AMF (`OrdinaryDiffEqAMF`, amf.jl + linsolve.jl): Adams–Moulton with a frozen linear
  operator solved exactly — matches our semilinear IMEX/ARK pattern (enumerate the
  alg from source at implementation time).

**Gate:** order at constant dt; cold-start correctness; the ABM54 embedded rate
test; restart-on-dt-change.

## Wave 11 — stabilized / Krylov extensions (C3, ~16 methods)

- `rkc.py` family: ROCK2, ROCK4 (Rodriguez regularized-Chebyshev), SERK2 (stabilized
  exponential RK 2), RKG1, RKG2, RKL1, RKMC2 (verify each method's construction from
  source — several are variants our driver does not currently parameterize).
- `rosenbrock.py`: Rosenbrock23, Rosenbrock32 (tableau forms), RosenbrockW6S4OS,
  Tsit5DA (Rosenbrock-type 5th order), HybridExplicitImplicitRK (verify structure).
- `exponential.py`: the remaining ODE exponential-RK members (ETDRK2 / EXPRB53
  candidates — enumerate from `lib/OrdinaryDiffEqExponentialRK` at implementation
  time; the local struct grep surfaced only the ETD2 family, so the full list is
  unverified here).
- `newmark.py`: GeneralizedAlpha (Hughes–Liu).

**Gate:** real-axis stability interval per method (the selling point of this family);
semilinear split tests; GMRES cost accounting where a solve is involved.

## Wave 12 — RKN + second-order-ODE family (C4, ~17 methods, new general RKN driver)

A general Runge–Kutta–Nystrom driver (a, b, b* tableaus on the position/velocity
form) replaces the hand-rolled `Nystrom5th` as the family driver:
RKN4, Nystrom4 (+ Nystrom4VelocityIndependent), DPRKN12, DPRKN4, DPRKN5, DPRKN6,
DPRKN6FM, DPRKN8, FineRKN4, FineRKN5, ERKN4/5/7 (exponential RKN — phi-Krylov, same
pattern as `exponential.py`), IRKN3/4 (implicit RKN — JFNK). Plus **PDIRK44**
(`OrdinaryDiffEqPDIRK`): partitioned DIRK for second-order ODEs — split-state driver
(position and velocity each run a DIRK half), comparable in scale to the ARK pair
machinery.

**Gate:** order on the oscillator/kepler second-order problems; Nystrom5th
regression (the new driver must reproduce the existing hand-rolled method's order);
block-solve scale for IRKN/PDIRK at s>2.

## Wave 13 — new medium drivers (C4, ~11 methods)

- Extrapolation family (`OrdinaryDiffEqExtrapolation`, 7 methods): AitkenNeville,
  ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
  ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
  ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation — Richardson/
  extrapolation tables over a base method; classical math, new driver.
- RKIP (`OrdinaryDiffEqRKIP`): Krylov-implicit Rosenbrock-type (enumerate members
  from source at implementation time).
- IRKC (`OrdinaryDiffEqStabilizedIRK`): stabilized implicit RK via Chebyshev
  polynomials — needs a linear (Krylov) solver per stage; rides the frozen-operator
  GMRES pattern from `rosenbrock.py`.
- FunctionMap (`OrdinaryDiffEqFunctionMap`): integration composed with a fixed map.
- Composite (`OrdinaryDiffEqCore.CompositeAlgorithm`): composition of registered
  schemes — orchestration layer, no new math.

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
Most are low-storage re-representations of methods already in or added by earlier
waves → in this driver they are algebraically redundant (no storage benefit: our
`RungeKuttaB` allocates a fresh stage state per stage). Batch screen: for each,
transcribe, compare the one-step map against the full-storage counterpart, register
only the genuinely new final maps (candidates: ParsaniKetchesonDeconinck3S105,
ParsaniKetchesonDeconinck3S173/184/205 — high-stage SSP-family methods with no
full-storage twin here; some CKLL* variants). Document the rest as redundant with a
pointer to ODE.

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
