# Fixed RKC/RKL clock and finalization correction

The three fixed relaxed-Chebyshev entries used equally spaced stage times j/s,
although their recurrence states have different first-order nodes. The result
was nonautonomous order loss for RKC2/RKL2. Their autonomous formulas, declared
orders, selection/parity rules and s-call cost remain unchanged by this fix.

## Independent before/after references

The [audit](../../benchmarks/rkc_clock_audit.py) compiles the actual historical
coefficient/driver functions from local Git revision f6c0edf against the current
shared state helpers. It compares them with the corrected registered methods.
It requires that baseline revision locally; it does not reconstruct the complete
historical Python/dependency environment. The
[result](evidence/rkc-clock.json) keeps source hashes and all component errors.

For y'=t from zero to T=1, with five stages and 20/40/80 steps:

| Method | Before errors | Before order | After errors |
|---|---|---:|---|
| RKC2 | .0234621 / .0117311 / .00586553 | 1.000 | rounding floor, 5.0e-16 at 80 |
| RKL2 | .0241533 / .0120767 / .00603833 | 1.000 | rounding floor, 3.3e-16 at 80 |

On the nonlinear manufactured exp(-t) trajectory, the measured last-pair
orders change from 1.020/1.018 to 2.008/2.008 for RKC2/RKL2. At 80 steps their
errors change from 6.30e-4 / 6.47e-4 to 4.96e-6 / 4.59e-6. RKC1 retains its
declared first order; its changed time nodes change the error constant rather
than its order. The audit records that behavior too.

## Construction

For a constant derivative, stage increments are h*c_j, so consistency requires

```
c0 = 0
c1 = mtilde1
c[j] = mu[j]*c[j-1] + nu[j]*c[j-2] + mtilde[j] + gtilde[j]
```

The stage clock is now t+h*c_j. Tests independently compare actual stage values
with their clocks on a constant-rate trajectory, including backward stepping,
minimum counts and fifty stages. Nonlinear time-dependent order is checked
separately; autonomous stability/property tests remain at their existing tolerances.

Reverse propagation through the linear recurrence yields all s final derivative
weights. They are delivered on `IntegrationResult.quadrature` and to finalize,
replacing the unusable empty weight list. Coupled quantity tests reconstruct each
field from the physical stage rates. There is no extra endpoint call: copied
fields still use the last evaluated stage, preserving the existing contract.

Affine state combinations use base+sum(weight*(stage-base)) instead of a rounded
weighted sum of complete states. This preserves zero-rate fields exactly, with
large/small offsets in float32/64, without changing the mathematical affine map.
State gradients and existing source-polynomial/stability tests still pass.

## Gates

Focused RKC, previous/new stabilized backends and gradient suite: **505 passed**.
The full CPU gate reports
**8,655 passed, 639 skipped, 157 warnings**, 341.01 seconds with eight workers
and one numerical thread per worker. The [full report](evidence/rkc-clock-full.json)
fingerprint matches the final runtime/tests/configuration. The
[installed-wheel gate](evidence/rkc-clock-wheel.json) includes the corrected
registered RKC2 time-force path and passes outside the checkout. All 57 published
Python blocks execute; 13 sketches remain labeled. Catalog and repository-link
checks pass, and Docusaurus builds successfully. Hardware limits remain the
suite's recorded skips; no extra device/graph coverage is inferred.
