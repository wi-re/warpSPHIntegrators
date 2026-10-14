# Method dispositions and coefficient provenance

Original audit at `5eb45e0`, supplemented by the Phase 8 first wave-9 batch on
14 October 2026. The [JSON ledger](method-dispositions.json)
records all 139 registered identifiers plus standalone, equivalent, deferred and
screening candidates. It includes implementation/source constructors, modifications,
test files, numerical evidence and intended consumers. Method implementation and
primary-source verification are separate fields: legacy methods remain implemented
even where their original primary-source revision is not yet pinned.

Parameterized physical splitting factories and the dense linear flow added in
execution Phase 7 are outside the fixed registration ledger. Their contracts,
Yoshida coefficient source and toy validation are recorded in
[the Phase 7 record](../development/phase7-foundation-validation.md).
The Phase 8 `GeneralizedAlpha(rho_inf)` factory is also outside that fixed ledger;
its classical parameter source, accepted acceleration, structural consumer and
numerical/work gates are recorded in [the driver validation](../development/generalized-alpha-validation.md).
The `Collocation(family, stages)` factory extends Gauss/Radau stages 2–9 outside
the fixed ledger. Its pinned coefficient/step oracle and Schur spatial backend
are recorded in [the higher-stage validation](../development/higher-collocation-validation.md).

## Source pin and transformations

Expansion waves use OrdinaryDiffEq **7.8.1**, metapackage git tree
`bae9b8e5f1d00e51c554648e5094e215164a19f8`. The
[Julia manifest](../../scripts/coefficients/julia/Manifest.toml) pins subpackage
versions and git trees. [Promoted extraction/oracle tools](../../scripts/coefficients/README.md)
preserve source dumps and independent results. Baseline implementations retain
their primary citations in source/family pages; unknown primary revisions are
explicitly marked awaiting verification, not assigned the expansion source pin.

- Source error differences are converted to companion solution weights where
  valid (`b_embed = b_main - btilde`); the estimate exponent is separately cataloged.
- SSPRK932 ships fixed-step: its source companion failed normalization/rate checks.
  PSRK names do not certify embedded pairs. DP8 and Feagin10/12/14 ship without
  estimates; their source estimators need supported constructions, not guessed weights.
- Wave 8 uses pure implicit tableaus. Kvaerno node corrections, row-sum abscissae,
  low-precision source coefficients, Cash4/SFSDIRK weight adjustments and embedded
  conventions are detailed in [section 3.28](../research/integrators/notes-3-28.md).
- Working/FSAL/companion stages are retained on the ordinary driver; low-storage
  source provenance does not imply reduced allocations here.
- ARS222 and ARS232 share an implicit map; their identifiers remain public.
  KenCarp3 shares the propagated ESDIRK324L2SA core, with a distinct companion.
  SSPSDIRK2 is a symplectic two-midpoint composition with `R(infinity)=+1`.

## Registered methods

| Method | Wave | Local callable/source | Evidence |
|---|---:|---|---|
| Forward Euler | baseline | [`forwardEuler`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Midpoint | baseline | [`midpoint`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Heun's Method (2nd order) | baseline | [`heunsMethod`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Ralston's Method (2nd order) | baseline | [`ralston`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| RK3 | baseline | [`RK3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Heun's Method (3rd order) | baseline | [`Heun3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Ralston's Method (3rd order) | baseline | [`ralston3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Wray's Method (3rd order) | baseline | [`Wray3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SSP RK3 | baseline | [`SSPRK3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SSPRK(10,4) | baseline | [`SSPRK104`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SSPRK53 | 1 | [`SSPRK53`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| SSPRK63 | 1 | [`SSPRK63`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| SSPRK73 | 1 | [`SSPRK73`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| SSPRK83 | 1 | [`SSPRK83`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| SSPRK54 | 1 | [`SSPRK54`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| KYKSSPRK42 | 1 | [`KYKSSPRK42`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| KYK2014DGSSPRK_3S2 | 1 | [`KYK2014DGSSPRK_3S2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-21.md) |
| SSPRK43 | 2 | [`SSPRK43`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-22.md) |
| SSPRK432 | 2 | [`SSPRK432`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-22.md) |
| SSPRK932 | 2 | [`SSPRK932`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-22.md) |
| SSPRK53_2N1 | 5 | [`SSPRK53_2N1`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-25.md) |
| SSPRK53_2N2 | 5 | [`SSPRK53_2N2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-25.md) |
| SSPRK53_H | 5 | [`SSPRK53_H`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-25.md) |
| RK4 | baseline | [`RK4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| RK4 (alternative) | baseline | [`RK4alt`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Nystrom 5th order | baseline | [`Nystrom5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Bogacki-Shampine 3(2) | baseline | [`BogackiShampine`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Dormand-Prince 5(4) | baseline | [`DormandPrince`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Cash-Karp 5(4) | baseline | [`CashKarp`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Tsit5 5(4) | 6 | [`Tsit5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| OwrenZen3 3(2) | 6 | [`OwrenZen3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| OwrenZen4 4(3) | 6 | [`OwrenZen4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| OwrenZen5 5(4) | 6 | [`OwrenZen5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| BS5 5(4) | 6 | [`BS5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Stepanov5 5(4) | 6 | [`Stepanov5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Alshina2 2(1) | 6 | [`Alshina2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Alshina3 | 6 | [`Alshina3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Alshina6 | 6 | [`Alshina6`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| MSRK5 | 6 | [`MSRK5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| MSRK6 | 6 | [`MSRK6`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| SIR54 | 6 | [`SIR54`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Ralston4 | 6 | [`Ralston4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| RKO65 | 6 | [`RKO65`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| RKM | 6 | [`RKM`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| PSRK4p7q6 | 6 | [`PSRK4p7q6`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| PSRK3p6q5 | 6 | [`PSRK3p6q5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| PSRK3p5q4 | 6 | [`PSRK3p5q4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-26.md) |
| Vern6 6(5) | 7 | [`Vern6`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Vern7 7(6) | 7 | [`Vern7`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Vern8 8(7) | 7 | [`Vern8`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Vern9 9(8) | 7 | [`Vern9`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| RKV76IIa 7(6) | 7 | [`RKV76IIa`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| TanYam7 7(6) | 7 | [`TanYam7`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| TsitPap8 8(7) | 7 | [`TsitPap8`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| DP8 | 7 | [`DP8`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Feagin10 | 7 | [`Feagin10`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Feagin12 | 7 | [`Feagin12`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Feagin14 | 7 | [`Feagin14`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-27.md) |
| Leap Frog | baseline | [`leapFrog`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Symplectic Euler | baseline | [`symplecticEuler`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Velocity Verlet | baseline | [`velocityVerlet`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| PEFRL | baseline | [`PEFRL`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| VEFRL | baseline | [`VEFRL`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| EPEC | baseline | [`midpoint`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| EPEC Modified | baseline | [`heunsMethod`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| TVD RK3 | baseline | [`TVDRK3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| TVD RK2 | baseline | [`TVDRK2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Semi-Implicit Euler | baseline | [`<lambda>`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Explicit Euler | baseline | [`<lambda>`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Backward Euler (implicit) | baseline | [`backwardEuler`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Implicit Midpoint | baseline | [`implicitMidpoint`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Trapezoidal (Crank-Nicolson) | baseline | [`trapezoidal`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SDIRK2 | baseline | [`SDIRK2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| TR-BDF2 | baseline | [`TRBDF2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| ESDIRK3(2)4L[2]SA | baseline | [`ESDIRK324L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| ESDIRK4(3)6L[2]SA | baseline | [`ESDIRK436L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Kvaerno3 | 8 | [`Kvaerno3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Kvaerno4 | 8 | [`Kvaerno4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Kvaerno5 | 8 | [`Kvaerno5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| KenCarp3 | 8 | [`KenCarp3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| KenCarp4 | 8 | [`KenCarp4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| KenCarp47 | 8 | [`KenCarp47`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| KenCarp5 | 8 | [`KenCarp5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| KenCarp58 | 8 | [`KenCarp58`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK3(2)5L[2]SA | 8 | [`ESDIRK325L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK4(3)6L[2]SA2 | 8 | [`ESDIRK436L2SA2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK4(3)7L[2]SA | 8 | [`ESDIRK437L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK5(4)7L[2]SA2 | 8 | [`ESDIRK547L2SA2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK5(4)I8L[2]SA | 8 | [`ESDIRK54I8L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ESDIRK6(5)9L[2]SA | 8 | [`ESDIRK659L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SSPSDIRK2 | 8 | [`SSPSDIRK2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ARS222 | 8 | [`ARS222`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ARS232 | 8 | [`ARS232`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ARS343 | 8 | [`ARS343`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| ARS443 | 8 | [`ARS443`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| BHR553 | 8 | [`BHR553`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Cash4 | 8 | [`Cash4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| CFNLIRK3 | 8 | [`CFNLIRK3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Hairer4 | 8 | [`Hairer4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Hairer42 | 8 | [`Hairer42`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SFSDIRK4 | 8 | [`SFSDIRK4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SFSDIRK5 | 8 | [`SFSDIRK5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SFSDIRK6 | 8 | [`SFSDIRK6`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SFSDIRK7 | 8 | [`SFSDIRK7`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| SFSDIRK8 | 8 | [`SFSDIRK8`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-28.md) |
| Gauss-Legendre 2 | baseline | [`gaussLegendre2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Radau IIA s=2 | baseline | [`radauIia2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Gauss-Legendre 3 | 9 (first batch) | [`gaussLegendre3`](../../src/warpSPHIntegrators/fullyimplicit.py) | [validation](../development/phase8-wave9-validation.md) |
| Radau IIA s=3 | 9 (first batch) | [`radauIia3`](../../src/warpSPHIntegrators/fullyimplicit.py) | [validation](../development/phase8-wave9-validation.md) |
| ARK3(2)4L[2]SA | baseline | [`ARK324L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| ARK4(3)6L[2]SA | baseline | [`ARK436L2SA`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Newmark | baseline | [`newmark`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| BDF1 | baseline | [`BDF1`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| BDF2 | baseline | [`BDF2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| BDF3 | baseline | [`BDF3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| BDF4 | baseline | [`BDF4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| BDF5 | baseline | [`BDF5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| IMEX Euler | baseline | [`IMEXEuler`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SBDF2 | baseline | [`SBDF2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SBDF3 | baseline | [`SBDF3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| CNAB2 | baseline | [`CNAB2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth 2 | baseline | [`AB2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth 3 | baseline | [`AB3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth 4 | baseline | [`AB4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth 5 | baseline | [`AB5`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth-Moulton 2 (PECE) | baseline | [`ABM2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth-Moulton 3 (PECE) | baseline | [`ABM3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Bashforth-Moulton 4 (PECE) | baseline | [`ABM4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| SSPRKMSVS32 | 3 | [`SSPRKMSVS32`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-23.md) |
| SSPRKMSVS43 | 3 | [`SSPRKMSVS43`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-3-23.md) |
| Adams-Moulton 2 (implicit) | baseline | [`AM2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Moulton 3 (implicit) | baseline | [`AM3`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| Adams-Moulton 4 (implicit) | baseline | [`AM4`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| RKC1 | baseline | [`RKC1`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| RKC2 | baseline | [`RKC2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| RKL2 | baseline | [`RKL2`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| ROS3P | baseline | [`integrateROS3P`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| ETD2RK | baseline | [`integrateETD2RK`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |
| EXPRB32 | baseline | [`integrateEXPRB32`](../../src/warpSPHIntegrators/integration.py) | [record](../research/integrators/notes-index.md) |

## Other dispositions

| Candidate | Status | Reason |
|---|---|---|
| pRRK22 / pRRK33 / pRRK54 | implemented | Standalone drivers depend on kappa and dt; no fixed registry entry |
| SplitEuler | equivalent | Forward Euler on combined `f1+f2`; old Symplectic Euler equivalence corrected by source inspection |
| ARS232 implicit side | equivalent | Stored implicit tableau equals ARS222; both public registrations retained; upstream additive pairs differ |
| Anas5 / FRK65 | deferred | Parameter-dependent fitted tableaus need a runtime parameter/step contract |
| PFRK87 | deferred | perform_step! computes nu=omega*dt and runtime coefficients; fixed cache alone does not define the phase-fitted map |
| SDIRK22 | excluded | Algorithm constructor returns Trapezoid; compact tableau exists but no trustworthy algorithm oracle |
| ExplicitTableaus.DormandPrince | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.RKF5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.RungeFirst5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Cassity5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Lawson5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.LutherKonen5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.LutherKonen52 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.LutherKonen53 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.PapakostasPapaGeorgiou5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.PapakostasPapaGeorgiou52 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Tsitouras5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.BogakiShampine5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.SharpSmart5 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.BogakiShampine3 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.CashKarp | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.RKF4 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Butcher63 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Butcher6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Butcher62 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Verner6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.DormandPrince6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.SharpVerner6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Verner9162 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Verner916 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerRobust6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerEfficient6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Verner7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Verner8 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Papakostas6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Lawson6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TsitourasPapakostas6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.DormandLockyerMcCorriganPrince6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaKasugaYamashitaYazaki6D | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaKasugaYamashitaYazaki6C | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaKasugaYamashitaYazaki6B | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaKasugaYamashitaYazaki6A | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.MikkawyEisa | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Chummund6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Chummund62 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Huta62 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Huta6 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.EnrightVerner7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerRobust7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerEfficient7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.SharpVerner7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.SharpSmart7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaYamashitaEfficient7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TanakaYamashitaStable7 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.CooperVerner8 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.CooperVerner82 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Curtis8 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.EnrightVerner8 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.dverk78 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.TsitourasPapakostas8 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerRobust9 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.VernerEfficient9 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Sharp9 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Tsitouras9 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |
| ExplicitTableaus.Tsitouras92 | awaiting verification | Enumerated primary-source candidate; map equivalence, source estimator and a useful niche have not been established |

All four planned ExplicitTableaus files have now been enumerated. Similar names
are not proof of map equivalence; each candidate remains awaiting verification
until algebraic/numerical screening and a consumer justify a disposition.
PFRK87 is deferred because the inspected `perform_step!` evaluates `nu=omega*dt`
and constructs phase-fitted coefficients at runtime. It needs a parameter/step
contract and small-phase limit tests, like Anas5/FRK65, before registration.
The [current execution plan](../../REPOSITORY_PLAN.md) owns scheduling.

Gauss/Radau three-stage prototypes remain unregistered after the failed Phase 6
diffusion scale gate; see [the decision and evidence](../development/phase6-validation.md).
