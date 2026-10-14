# Phase 8 — expanded low-storage-source RK maps

ExpandedRK supplies 54 screened main maps, including 19 checked companions,
through the existing owned full-stage RungeKuttaB engine. The fixed catalog
remains 142. LowStorageRK provenance does not imply low-storage execution.

## Source and screening

OrdinaryDiffEq 7.8.1 tree bae9b8e5f1d00e51c554648e5094e215164a19f8 pins
LowStorageRK 3.4.0. The stdlib Julia oracle executes all 60 named coefficient
constructors and the actual initialize!/perform_step! kernels with minimal
state/stats/error scaffolding and broadcast/muladd annotations removed. There
are 61 branches because SHLDDRK_2N alternates five/six-stage phases. Unit-RHS
bases recover affine stage coefficients, weights, times and initial-state scales.
Separate runs record 122 autonomous/forced nonlinear maps and 360 scalar
trajectories on three refinement grids. Re-execution reproduces the tracked JSON.

The generator enumerates all rooted-tree order conditions through advertised
order five (1, 1, 2, 4, 9 trees), checks stage row sums against source clocks and
checks initial-state scales at tolerance 1e-10. It checks comparison orders
separately. Duplicate stage rows are merged; a physical endpoint is reconstructed
and the standard increment-form RK engine preserves zero-rate offsets exactly.
Actual qualified nonlinear main outputs agree with source within 9.22e-15.
Coefficient arrays have immutable backing storage. No bitwise equivalence is
claimed after binary64 affine normalization.

| Pinned source candidate | Disposition and reason |
|---|---|
| CKLLSRK54_3C | Excluded from general-ODE factory: autonomous order four passes, but c5 differs from its row sum by .1441458752; b*c=.5502589357 instead of .5. Source y'=t errors halve with refinement, giving order one. |
| CKLLSRK95_4C | Excluded: advertised order five fails order two by .0169723981. Source decay errors on 20/40/80 steps are 3.0553e-4, 1.5445e-4, 7.7639e-5, observed order .9923. |
| ORK256 | Deferred: rounded coefficients miss order two by 5.29288e-7. |
| SHLDDRK52 | Deferred: consistency defect 5.42105e-7. |
| SHLDDRK64 | Deferred: consistency defect 1.59303e-7. |
| SHLDDRK_2N | Deferred: alternating branches miss consistency by 6.53610e-8 / 3.23573e-7; source forced refinement approaches a precision floor. |
| CKLLSRK85_4FM_4R comparison | Main order five qualifies; advertised fourth-order companion satisfies only order one. Only the fixed-step main map is exposed. |

Rounded-source deferral is a binary64 qualification decision, not a claim that
these algorithms have no useful lower-precision regime. No allow_unqualified
flag or invented companion is added. The canonical A,b,c screen finds no matches
against 60 fixed explicit tableaus or within these 54 maps. Such a nonmatch is
not a proof against every alternative representation or non-tableau factory.

## Runtime and understood consumers

The existing RK driver retains initialized stageWeights, injected stageIndex,
compatible-energy rerunRequest, copied physical endpoint and all-field weights.
Factory inputs/history and returned stage buffers are owned. Tests cover all
source maps, formal/observed orders, nineteen estimates, adaptive rejection,
parameter/state gradients, immutable coefficients, time forcing, coupled heat,
lifecycle reruns, physical composition, switching and portable checkpoints.
There are 243 focused passing tests. Tightening earlier explicit-absolute
pytest comparisons with rel=0 also leaves 714 numerical tests passing.

The benchmark records 448 fixed and 80 adaptive known reaction/spiral-heat cases
at N=64/4096. On N=4096 reaction, T=1, h=.1 and rates .1–10,
AlJahdaliAdv3S165 uses 170 RHS entries with 1.17e-9 max error; DP5 uses 61
with 8.99e-7. At rtol=1e-8, atol=1e-10, RDPK3Sp510 uses 385 entries
with 2.38e-11 error versus DP5's 511 with 4.40e-11. On the damped
spiral/heat problem at that tolerance, RDPK3Sp510 uses 539 entries with
2.02e-8 max error; DP5 uses 742 with 2.00e-8. Equal tolerances or grids
are not equal-accuracy timing comparisons. There is no universal speed,
SSP, arbitrary conservation or native-cache equivalence claim.

Full-stage storage is O(sN), including autograd graphs. Native low-storage
buffers, fused updates, FSAL and upstream controller behavior remain separate.
These measurements provide numerical baselines for that later allocation work.

## Validation

The final fingerprint-matched CPU gate passes **8,139 tests, 639 skipped**, no
failures/errors, in 322.17 seconds of pytest time / 323.4 seconds with orchestration.
Eight workers use one numerical thread each. The installed 0.6.1 wheel passes
all public imports, 135 fixed tensor drivers and the 54-name/adaptive factory
smoke outside the checkout with isolated project imports and shared numerical
dependencies. Historical editable metadata is 0.5.0; source/wheel are 0.6.1.

Evidence: [source oracle](../../scripts/coefficients/wave16/oracle.json),
[coefficient screen](../../scripts/coefficients/wave16/screen.json),
[consumer measurements](evidence/expanded-rk-backend.json),
[full CPU gate](evidence/expanded-rk-full.json),
[installed wheel](evidence/expanded-rk-wheel.json).

```bash
python scripts/coefficients/wave16/generate.py --check
python -m pytest -q tests/test_expanded_rk.py
python benchmarks/expanded_rk_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/expanded-rk-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/expanded-rk-wheel
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave16/oracle.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
```
