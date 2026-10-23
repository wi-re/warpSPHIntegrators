# Source candidate inventory

Reconciled 23 October 2026 from the existing method ledger and expansion design.
This preserves declaration inventory and fitted-method contracts while the old
expansion schedule is retired. Enumeration does not establish coefficients,
equivalence, order, a usable estimate or a useful implementation niche. The
[method ledger](../plans/method-dispositions.json) retains individual statuses;
[the execution plan](../../REPOSITORY_PLAN.md) owns scheduling.

## Source pin and verification scope

Expansion source: OrdinaryDiffEq 7.8.1, metapackage tree
`bae9b8e5f1d00e51c554648e5094e215164a19f8`. The
[Julia manifest](../../scripts/coefficients/julia/Manifest.toml) pins subpackages;
[coefficient tools](../../scripts/coefficients/README.md) retain independent
oracles and transformations. The original source-member enumeration dates to
12 October 2026. This reconciliation preserves that evidence level and does not
claim a fresh upstream coefficient audit. Legacy primary-source gaps remain
separate qualifications in registered entries.

## Extra explicit tableau constructors

All 59 entries below remain **awaiting verification**. Counts by source file are
16 in order 5, 25 in order 6, 10 in order 7 and eight in orders 8/9. Names in
the data package can overlap methods already available through other upstream
drivers. Before promotion, compare full main/companion maps and clock conventions
against the existing catalog, validate order and estimator behavior, and identify
a concrete consumer benefit. No additional registration follows from this list.

| Candidate | Retained source declaration |
|---|---|
| `ExplicitTableaus.DormandPrince` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, DormandPrince(T,T_time)` |
| `ExplicitTableaus.RKF5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, RKF5(T,T_time)` |
| `ExplicitTableaus.RungeFirst5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, RungeFirst5(T,T_time)` |
| `ExplicitTableaus.Cassity5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, Cassity5(T,T_time)` |
| `ExplicitTableaus.Lawson5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, Lawson5(T,T_time)` |
| `ExplicitTableaus.LutherKonen5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, LutherKonen5(T,T_time)` |
| `ExplicitTableaus.LutherKonen52` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, LutherKonen52(T,T_time)` |
| `ExplicitTableaus.LutherKonen53` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, LutherKonen53(T,T_time)` |
| `ExplicitTableaus.PapakostasPapaGeorgiou5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, PapakostasPapaGeorgiou5(T,T_time)` |
| `ExplicitTableaus.PapakostasPapaGeorgiou52` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, PapakostasPapaGeorgiou52(T,T_time)` |
| `ExplicitTableaus.Tsitouras5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, Tsitouras5(T,T_time)` |
| `ExplicitTableaus.BogakiShampine5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, BogakiShampine5(T,T_time)` |
| `ExplicitTableaus.SharpSmart5` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, SharpSmart5(T,T_time)` |
| `ExplicitTableaus.BogakiShampine3` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, BogakiShampine3(T,T_time)` |
| `ExplicitTableaus.CashKarp` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, CashKarp(T,T_time)` |
| `ExplicitTableaus.RKF4` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order5.jl, RKF4(T,T_time)` |
| `ExplicitTableaus.Butcher63` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Butcher63(T,T_time)` |
| `ExplicitTableaus.Butcher6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Butcher6(T,T_time)` |
| `ExplicitTableaus.Butcher62` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Butcher62(T,T_time)` |
| `ExplicitTableaus.Verner6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Verner6(T,T_time)` |
| `ExplicitTableaus.DormandPrince6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, DormandPrince6(T,T_time)` |
| `ExplicitTableaus.SharpVerner6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, SharpVerner6(T,T_time)` |
| `ExplicitTableaus.Verner9162` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Verner9162(T,T_time)` |
| `ExplicitTableaus.Verner916` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Verner916(T,T_time)` |
| `ExplicitTableaus.VernerRobust6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, VernerRobust6(T,T_time)` |
| `ExplicitTableaus.VernerEfficient6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, VernerEfficient6(T,T_time)` |
| `ExplicitTableaus.Verner7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Verner7(T,T_time)` |
| `ExplicitTableaus.Verner8` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Verner8(T,T_time)` |
| `ExplicitTableaus.Papakostas6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Papakostas6(T,T_time)` |
| `ExplicitTableaus.Lawson6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Lawson6(T,T_time)` |
| `ExplicitTableaus.TsitourasPapakostas6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, TsitourasPapakostas6(T,T_time)` |
| `ExplicitTableaus.DormandLockyerMcCorriganPrince6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, DormandLockyerMcCorriganPrince6(T,T_time)` |
| `ExplicitTableaus.TanakaKasugaYamashitaYazaki6D` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, TanakaKasugaYamashitaYazaki6D(T,T_time)` |
| `ExplicitTableaus.TanakaKasugaYamashitaYazaki6C` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, TanakaKasugaYamashitaYazaki6C(T,T_time)` |
| `ExplicitTableaus.TanakaKasugaYamashitaYazaki6B` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, TanakaKasugaYamashitaYazaki6B(T,T_time)` |
| `ExplicitTableaus.TanakaKasugaYamashitaYazaki6A` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, TanakaKasugaYamashitaYazaki6A(T,T_time)` |
| `ExplicitTableaus.MikkawyEisa` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, MikkawyEisa(T,T_time)` |
| `ExplicitTableaus.Chummund6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Chummund6(T,T_time)` |
| `ExplicitTableaus.Chummund62` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Chummund62(T,T_time)` |
| `ExplicitTableaus.Huta62` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Huta62(T,T_time)` |
| `ExplicitTableaus.Huta6` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order6.jl, Huta6(T,T_time)` |
| `ExplicitTableaus.EnrightVerner7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, EnrightVerner7(T,T_time)` |
| `ExplicitTableaus.VernerRobust7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, VernerRobust7(T,T_time)` |
| `ExplicitTableaus.VernerEfficient7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, VernerEfficient7(T,T_time)` |
| `ExplicitTableaus.SharpVerner7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, SharpVerner7(T,T_time)` |
| `ExplicitTableaus.SharpSmart7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, SharpSmart7(T,T_time)` |
| `ExplicitTableaus.TanakaYamashitaEfficient7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, TanakaYamashitaEfficient7(T,T_time)` |
| `ExplicitTableaus.TanakaYamashitaStable7` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, TanakaYamashitaStable7(T,T_time)` |
| `ExplicitTableaus.CooperVerner8` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, CooperVerner8(T,T_time)` |
| `ExplicitTableaus.CooperVerner82` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, CooperVerner82(T,T_time)` |
| `ExplicitTableaus.Curtis8` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order7.jl, Curtis8(T,T_time)` |
| `ExplicitTableaus.EnrightVerner8` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, EnrightVerner8(T,T_time)` |
| `ExplicitTableaus.dverk78` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, dverk78(T,T_time)` |
| `ExplicitTableaus.TsitourasPapakostas8` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, TsitourasPapakostas8(T,T_time)` |
| `ExplicitTableaus.VernerRobust9` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, VernerRobust9(T,T_time)` |
| `ExplicitTableaus.VernerEfficient9` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, VernerEfficient9(T,T_time)` |
| `ExplicitTableaus.Sharp9` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, Sharp9(T,T_time)` |
| `ExplicitTableaus.Tsitouras9` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, Tsitouras9(T,T_time)` |
| `ExplicitTableaus.Tsitouras92` | `OrdinaryDiffEq 7.8.1, lib/OrdinaryDiffEqExplicitTableaus/src/tableaus_order8_9.jl, Tsitouras92(T,T_time)` |

## Runtime fitted tableau contracts

**PFRK87 remains deferred.** The pinned HighOrderRK `perform_step!` computes
`nu = omega * dt` and runtime phase-fitted coefficients. A fixed constant-cache
dump is not the complete method. Activation requires a frequency/parameter and
timestep contract, source step-body comparisons across those parameters,
nonautonomous/main-order checks, and independently qualified error estimation
if an adaptive pair is proposed. Preserve its identity separately from DP8 and
other high-order fixed maps.

**Anas5 / FRK65 remain deferred.** Their parameter-dependent coefficients require
an explicit runtime parameter/step contract. They are not completed by adding
another fixed tableau. The [order-5 source record](../research/integrators/notes-3-26.md)
retains their original screening, alongside pRRK as an example of a deliberately
parameterized driver.

## Existing map and estimate distinctions

SSPRK932, DP8 and Feagin methods retain their documented absent/unqualified
companions; pseudo-symplectic names do not imply embedded pairs. SplitEuler is
Forward Euler on the combined RHS; ARS222/232 share an implicit map while their
additive pairs differ. SDIRK22 remains excluded because its algorithm constructor
returns Trapezoid despite a separately stored tableau. Source-specific node,
weight and estimator corrections remain in the [method ledger](../plans/method-dispositions.md)
and [numbered numerical records](../research/integrators/notes-index.md).
