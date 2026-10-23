# Numerical measurement index

The original NOTES sections are preserved below, including their numbered anchors.
These are dated evidence, not an active task list. See [the current plan](../../../REPOSITORY_PLAN.md)
and [architecture](../../development/architecture.md) for current behavior.

The root NOTES redirect was removed in the [23 October reference migration](../../development/repository-cleanup-batch3-2026-10-23.md).
The numbered anchors below remain the stable entry points for historical
measurements; [the migration map](../../development/migration-map.json) records
retired filenames and their retained destinations.

<a id="warpsphintegrators--architecture-notes-and-open-work"></a>
- [`warpSPHIntegrators` — architecture notes and open work](baseline-audit-2026-08-05.md#warpsphintegrators--architecture-notes-and-open-work)

<a id="0-status"></a>
- [0. Status](baseline-audit-2026-08-05.md#0-status)

<a id="still-open"></a>
- [Still open](baseline-audit-2026-08-05.md#still-open)

<a id="214-housekeeping-pass--done-2026-09-15"></a>
- [2.14 Housekeeping pass — done 2026-09-15](baseline-audit-2026-08-05.md#214-housekeeping-pass--done-2026-09-15)

<a id="1-what-this-library-actually-is"></a>
- [1. What this library actually is](baseline-audit-2026-08-05.md#1-what-this-library-actually-is)

<a id="11-the-four-layers"></a>
- [1.1 The four layers](baseline-audit-2026-08-05.md#11-the-four-layers)

<a id="12-the-generic-update-kernel"></a>
- [1.2 The generic update kernel](baseline-audit-2026-08-05.md#12-the-generic-update-kernel)

<a id="13-one-step-end-to-end-rk4"></a>
- [1.3 One step, end to end (RK4)](baseline-audit-2026-08-05.md#13-one-step-end-to-end-rk4)

<a id="2-open-work-warp-backend-and-sph-awareness"></a>
- [2. Open work: Warp backend and SPH-awareness](baseline-audit-2026-08-05.md#2-open-work-warp-backend-and-sph-awareness)

<a id="21-allocation-per-stage-fights-warps-execution-model"></a>
- [2.1 Allocation-per-stage fights Warp's execution model](baseline-audit-2026-08-05.md#21-allocation-per-stage-fights-warps-execution-model)

<a id="22-no-differentiability-story-for-warp"></a>
- [2.2 No differentiability story for Warp](baseline-audit-2026-08-05.md#22-no-differentiability-story-for-warp)

<a id="23-nothing-in-the-api-is-sph-aware"></a>
- [2.3 Nothing in the API is SPH-aware](baseline-audit-2026-08-05.md#23-nothing-in-the-api-is-sph-aware)

<a id="24-packaging--done"></a>
- [2.4 Packaging — done](baseline-audit-2026-08-05.md#24-packaging--done)

<a id="3-multistep-and-implicit-methods"></a>
- [3. Multistep and implicit methods](baseline-audit-2026-08-05.md#3-multistep-and-implicit-methods)

<a id="30-constraints-from-the-surrounding-simulation"></a>
- [3.0 Constraints from the surrounding simulation](notes-3-0.md#30-constraints-from-the-surrounding-simulation)

<a id="31-headline-both-are-cheaper-than-they-look-and-only-one-wall-is-left"></a>
- [3.1 Headline: both are cheaper than they look, and only one wall is left](notes-3-1.md#31-headline-both-are-cheaper-than-they-look-and-only-one-wall-is-left)

<a id="32-what-a-fixed-point-solve-can-and-cannot-do"></a>
- [3.2 What a fixed-point solve can and cannot do](notes-3-2.md#32-what-a-fixed-point-solve-can-and-cannot-do)

<a id="33-what-implicit-does-buy-even-without-stiffness"></a>
- [3.3 What implicit *does* buy, even without stiffness](notes-3-3.md#33-what-implicit-does-buy-even-without-stiffness)

<a id="34-newton-without-forward-mode-ad"></a>
- [3.4 Newton without forward-mode AD](notes-3-4.md#34-newton-without-forward-mode-ad)

<a id="35-shared-groundwork-prerequisite-for-both--done-2026-08-24"></a>
- [3.5 Shared groundwork (prerequisite for both) — DONE 2026-08-24](notes-3-5.md#35-shared-groundwork-prerequisite-for-both--done-2026-08-24)

<a id="36-valid-schemes-and-what-each-costs"></a>
- [3.6 Valid schemes and what each costs](notes-3-6.md#36-valid-schemes-and-what-each-costs)

<a id="diagonally-implicit-rk-sequential-1-stage-solves--all-seven-shipped-four-2026-08-24-the-rest-2026-09-09"></a>
- [Diagonally implicit RK (sequential 1-stage solves) — all seven shipped (four 2026-08-24, the rest 2026-09-09)](notes-3-6.md#diagonally-implicit-rk-sequential-1-stage-solves--all-seven-shipped-four-2026-08-24-the-rest-2026-09-09)

<a id="fully-implicit-rk-needs-a-coupled-sn-unknown-solve--a-different-solver-shape"></a>
- [Fully implicit RK (needs a *coupled* s·N-unknown solve — a different solver shape)](notes-3-6.md#fully-implicit-rk-needs-a-coupled-sn-unknown-solve--a-different-solver-shape)

<a id="linear-multistep--adams-bashforth-moulton-shipped-2026-08-24"></a>
- [Linear multistep — Adams-Bashforth/-Moulton shipped 2026-08-24](notes-3-6.md#linear-multistep--adams-bashforth-moulton-shipped-2026-08-24)

<a id="imex--additive-rk--the-right-answer-for-sph-and-the-most-work"></a>
- [IMEX / additive RK — the right answer for SPH, and the most work](notes-3-6.md#imex--additive-rk--the-right-answer-for-sph-and-the-most-work)

<a id="37-pain-points-and-limitations"></a>
- [3.7 Pain points and limitations](notes-3-7.md#37-pain-points-and-limitations)

<a id="38-phased-plan"></a>
- [3.8 Phased plan](notes-3-8.md#38-phased-plan)

<a id="39-the-probes-behind-these-numbers"></a>
- [3.9 The probes behind these numbers](notes-3-9.md#39-the-probes-behind-these-numbers)

<a id="310-phase-8-the-nonlinearstiff-benchmark-suite--done-2026-09-09"></a>
- [3.10 Phase 8: the nonlinear/stiff benchmark suite — done 2026-09-09](notes-3-10.md#310-phase-8-the-nonlinearstiff-benchmark-suite--done-2026-09-09)

<a id="311-phase-13-tvd--ssp-classification-of-the-registered-schemes--done-2026-09-10"></a>
- [3.11 Phase 13: TVD / SSP classification of the registered schemes — done 2026-09-10](notes-3-11.md#311-phase-13-tvd--ssp-classification-of-the-registered-schemes--done-2026-09-10)

<a id="312-phase-14-the-structured-rhs-interface--done-2026-09-10"></a>
- [3.12 Phase 14: the structured RHS interface — done 2026-09-10](notes-3-12.md#312-phase-14-the-structured-rhs-interface--done-2026-09-10)

<a id="313-phase-12-rkc--rkl--stabilized-explicit-super-timestepping--done-2026-09-11"></a>
- [3.13 Phase 12: RKC / RKL — stabilized explicit super-timestepping — done 2026-09-11](notes-3-13.md#313-phase-12-rkc--rkl--stabilized-explicit-super-timestepping--done-2026-09-11)

<a id="314-phase-7-ros3p-rosenbrock-w--done-2026-09-11"></a>
- [3.14 Phase 7: ROS3P (Rosenbrock-W) — done 2026-09-11](notes-3-14.md#314-phase-7-ros3p-rosenbrock-w--done-2026-09-11)

<a id="315-phase-7-etd2rk-exponential-integrator--done-2026-09-12"></a>
- [3.15 Phase 7: ETD2RK (exponential integrator) — done 2026-09-12](notes-3-15.md#315-phase-7-etd2rk-exponential-integrator--done-2026-09-12)

<a id="316-phase-7-exprb32-exponential-rosenbrock--done-2026-09-13"></a>
- [3.16 Phase 7: EXPRB32 (exponential Rosenbrock) — done 2026-09-13](notes-3-16.md#316-phase-7-exprb32-exponential-rosenbrock--done-2026-09-13)

<a id="317-adaptive-step-control-and-dense-output-roadmap-phase-11"></a>
- [3.17 Adaptive step control and dense output (roadmap Phase 11)](notes-3-17.md#317-adaptive-step-control-and-dense-output-roadmap-phase-11)

<a id="318-phase-6-coupled-fully-implicit-rk-gauss-legendre-2-radau-iia-s2--done-2026-09-15"></a>
- [3.18 Phase 6: coupled fully implicit RK (Gauss-Legendre 2, Radau IIA s=2) — done 2026-09-15](notes-3-18.md#318-phase-6-coupled-fully-implicit-rk-gauss-legendre-2-radau-iia-s2--done-2026-09-15)

<a id="319-phase-12-imex-linear-multistep-sbdf2-sbdf3-cnab2--done-2026-09-15"></a>
- [3.19 Phase 12: IMEX linear multistep (SBDF2, SBDF3, CNAB2) — done 2026-09-15](notes-3-19.md#319-phase-12-imex-linear-multistep-sbdf2-sbdf3-cnab2--done-2026-09-15)

<a id="320-phase-12-follow-up-ssprk104-shus-order-4-ssp-method--done-2026-09-15"></a>
- [3.20 Phase 12 follow-up: SSPRK(10,4), Shu's order-4 SSP method — done 2026-09-15](notes-3-20.md#320-phase-12-follow-up-ssprk104-shus-order-4-ssp-method--done-2026-09-15)

<a id="321-scheme-expansion-wave-1-the-seven-plain-ssprk-tableaus--done-2026-10-12"></a>
- [3.21 Scheme-expansion wave 1: the seven plain SSPRK tableaus — done 2026-10-12](notes-3-21.md#321-scheme-expansion-wave-1-the-seven-plain-ssprk-tableaus--done-2026-10-12)

<a id="322-scheme-expansion-wave-2-the-ssprk-32-pairs-and-ssprk932--done-2026-10-12"></a>
- [3.22 Scheme-expansion wave 2: the SSPRK 3(2) pairs and SSPRK932 — done 2026-10-12](notes-3-22.md#322-scheme-expansion-wave-2-the-ssprk-32-pairs-and-ssprk932--done-2026-10-12)

<a id="323-scheme-expansion-wave-3-ssprkmsvs3243----shus-ssp-multistep----done-2026-10-13"></a>
- [3.23 Scheme-expansion wave 3: SSPRKMSVS32/43 -- Shu's SSP multistep -- done 2026-10-13](notes-3-23.md#323-scheme-expansion-wave-3-ssprkmsvs3243----shus-ssp-multistep----done-2026-10-13)

<a id="324-scheme-expansion-wave-4-prrk223354----parametric-relaxation-ssprk----done-2026-10-13"></a>
- [3.24 Scheme-expansion wave 4: pRRK22/33/54 -- parametric-relaxation SSPRK -- done 2026-10-13](notes-3-24.md#324-scheme-expansion-wave-4-prrk223354----parametric-relaxation-ssprk----done-2026-10-13)

<a id="325-scheme-expansion-wave-5-ssprk53_2n1_2n2_h----low-storage-third-order-ssp----done-2026-10-13"></a>
- [3.25 Scheme-expansion wave 5: SSPRK53_2N1/_2N2/_H -- low-storage third-order SSP -- done 2026-10-13](notes-3-25.md#325-scheme-expansion-wave-5-ssprk53_2n1_2n2_h----low-storage-third-order-ssp----done-2026-10-13)

<a id="326-scheme-expansion-wave-6-tsit5--the-loworderrk-workhorses----done-2026-10-13"></a>
- [3.26 Scheme-expansion wave 6: Tsit5 + the LowOrderRK workhorses -- done 2026-10-13](notes-3-26.md#326-scheme-expansion-wave-6-tsit5--the-loworderrk-workhorses----done-2026-10-13)

<a id="327-scheme-expansion-wave-7-the-high-order-tableaus-verner--highorderrk--feagin--dp8----done-2026-10-13"></a>
- [3.27 Scheme-expansion wave 7: the high-order tableaus (Verner / HighOrderRK / Feagin / DP8) -- done 2026-10-13](notes-3-27.md#327-scheme-expansion-wave-7-the-high-order-tableaus-verner--highorderrk--feagin--dp8----done-2026-10-13)

<a id="328-scheme-expansion-wave-8-the-remaining-dirk-tableaus-kvaerno--kencarp--esdirk--ars--cash--hairer--sfsdirk----implemented-2026-10-13-validation-gate-2026-10-14"></a>
- [3.28 Scheme-expansion wave 8: the remaining DIRK tableaus (Kvaerno / KenCarp / ESDIRK / ARS / Cash / Hairer / SFSDIRK) -- implemented 2026-10-13; validation gate 2026-10-14](notes-3-28.md#328-scheme-expansion-wave-8-the-remaining-dirk-tableaus-kvaerno--kencarp--esdirk--ars--cash--hairer--sfsdirk----implemented-2026-10-13-validation-gate-2026-10-14)

<a id="reproducing-the-results"></a>
- [Reproducing the results](reproduction.md#reproducing-the-results)

<a id="scriptsstep_reuse_convergencepy-committed"></a>
- [`benchmarks/step_reuse_convergence.py` (committed)](reproduction.md#scriptsstep_reuse_convergencepy-committed)

<a id="scriptsadaptive_benchmarkpy-committed"></a>
- [`benchmarks/adaptive_benchmark.py` (committed)](reproduction.md#scriptsadaptive_benchmarkpy-committed)

<a id="scriptsblockrk_benchmarkpy-committed"></a>
- [`benchmarks/blockrk_benchmark.py` (committed)](reproduction.md#scriptsblockrk_benchmarkpy-committed)

<a id="tests-committed"></a>
- [`tests/` (committed)](reproduction.md#tests-committed)
