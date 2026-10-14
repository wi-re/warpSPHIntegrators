# Improved Nyström with corrected startup

`ImprovedNystrom(3/4)` implements the explicit fixed-step IRKN warm recurrences
for velocity-independent second-order mechanics. It is separate from the 142
fixed registrations. The package seed intentionally corrects the pinned source's
nonautonomous startup defect. This is an explicit source transformation, not a
claim of unmodified IRKN source equivalence.

## Numerical construction and source checks

For q'=v, write a_n=f(t_n,q_n), with old substage forces B and C. IRKN3 uses
B_n=f(t_n+h/2,q_n+h*v_n/2+h²*a_(n-1)/8), then

```
q_(n+1) = q_n + h*(3*v_n/2-v_(n-1)/2) + h²*5*(B_n-B_(n-1))/12
v_(n+1) = v_n + h*(2*a_n/3+a_(n-1)/3+5*(B_n-B_(n-1))/6)
```

IRKN4 uses B_n=f(t_n+h/4,q_n+h*v_n/4+h²*a_n/32) and
C_n=f(t_n+3*h/4,q_n+3*h*v_n/4+9*h²*B_n/32), then

```
q_(n+1) = q_n + h*(3*v_n/2-v_(n-1)/2)
          + h²*(7*(B_n-B_(n-1))/24+(C_n-C_(n-1))/8)
v_(n+1) = v_n + h*(19*a_n/18-a_(n-1)/18
                   -(B_n-B_(n-1))/6+11*(C_n-C_(n-1))/18)
```

The first accepted step uses the fourth-order velocity-independent Nyström
method. Its old physical acceleration is a_0, its new physical acceleration
is a_1, and old B/C are evaluated at their own old positions and times.
The independent [exact Taylor-jet gate](../../scripts/coefficients/wave12/irkn_order.py)
treats the partial derivatives of a general f(t,q) as independent symbols and
checks that every position/velocity residual coefficient through the declared
order vanishes. It includes IRKN3's n-2 acceleration in the old substage.
[Zero residuals](../../scripts/coefficients/wave12/irkn_order.json) establish
local consistency; the fourth-order bootstrap supplies an
adequate initial anchor. Reusing an old-position force at a shifted time as the
physical old acceleration fails this construction, as the source audit shows.

The [original audit](irkn-source-audit.md) pins OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8 and RKN 2.2.0. All 24 corrected-seed
analytic oscillator/time-force/forced-oscillator trajectories match the package.
The new [Julia oracle](../../scripts/coefficients/wave12/irkn_corrected.jl)
reuses the actual mutable source cache/step bodies and changes only the seed
slots after bootstrap. It checks a nonlinear time-dependent force with exact
q=exp(.7t), running 16/32/64/128 steps. Every accepted q/v matches the package
within 2e-13; raw trajectories and source calls remain in
[the source result](../../scripts/coefficients/wave12/irkn_corrected.json).
Source-only extra startup evaluations are retained in that call record and
are not claimed to equal the package's corrected cold-call schedule.

Final component refinement rates in that oracle are 3.485/2.974 for IRKN3
and 4.178/4.135 for IRKN4. The IRKN3 position rate has cancellation; the
velocity rate controls the trajectory. A separate nonlinear manufactured
exp(t) test checks combined position/velocity order without assuming that
every component has the same leading error constant. The linear q''=t
reference is exact to roundoff after correction. These are finite-grid
qualification results; they do not assert symplecticity or unconditional stability.

The independent [oscillator amplification gate](../../scripts/coefficients/wave12/irkn_stability.py)
includes all retained force-history variables. At zero force there are two
physical unit roots and three/four zero history roots; physical drift coupling
is O(h), giving bounded powers on finite time intervals. The measured spectral
radius at omega*h=.5 is .998321 for IRKN3 and 1.00003135 for IRKN4; at omega*h=1
it is .916465 and 1.00278838. IRKN4 therefore slightly amplifies undamped
oscillations even at small positive steps. This limits its long-time mechanical
usefulness despite good finite-horizon force/accuracy results. All roots are
retained in [the stability report](../../scripts/coefficients/wave12/irkn_stability.json).

## Ownership and runtime boundaries

History stores physical old/end rates, old substages, previous velocity and
the accepted state's values/time/layout/UID. Any mismatch or a timestep change
restarts before consumption. `restart_history=True` handles external RHS
parameter changes. Snapshotting preserves gradient graphs while avoiding
aliasing between input history, returned stages and quadrature metadata.

Both policies check q'=v and passive quantities at actual evaluations. Default
structure probing also checks local velocity independence, including detached
velocity dependence. Declared structure is a caller assertion. There is no
embedded error estimate, adaptive mode or physical subflow integration.
Algorithm switching can retain an active IRKN child's history and cold-start
transitions. In-memory mechanical restart is supported; portable custom-system
serialization is application-owned.

`ImprovedNystromWeights` pairs old/current stages with component weights and
an explicit previous-velocity contribution. Custom finalize must opt into
`irkn_compatible=True` and preserve integrated fields/time/IDs. Tests reconstruct
both q/v from the delivered quadrature, count initialize/finalize calls, check
copied endpoint fields/passive quantities, failure rollback, gradients with
respect to the initial state and force parameters, and history invalidation.

## Known consumer and cost

The [benchmark](../../benchmarks/improved_nystrom_backend.py) measures 60 CPU
float64 trajectories at N=64 and 4096, with frequencies .5–3 and an exact
oscillatory solution, both linear and nonlinear forced acceleration. It compares
IRKN3/4 with fourth-order one-step Nyström, RK4 and Dormand–Prince. Structure
is declared; probe overhead is deliberately excluded and explicitly labeled.

At N=4096, nonlinear forcing and h=.0625, sixteen accepted steps give:

| Method | Full RHS calls | max position error | max velocity error | seconds |
|---|---:|---:|---:|---:|
| Improved Nyström 3 | 35 | 4.66e-4 | 1.99e-4 | .0184 |
| Improved Nyström 4 | 51 | 1.33e-6 | 4.91e-6 | .0230 |
| Nyström fourth-order VI | 64 | 1.08e-5 | 3.32e-5 | .0172 |
| RK4 | 64 | 3.13e-5 | 7.63e-5 | .0135 |
| Dormand–Prince 5(4) | 97 | 1.31e-7 | 6.36e-7 | .0348 |

Warm steps cost two/three full RHS calls, cold steps five/six. Default probes
add six entries per step. IRKN4 has a useful force/accuracy tradeoff on this
consumer; history ownership/checks make its cheap-RHS wall time slower than
one-step Nyström/RK4. No equal-accuracy speed, SPH conservation or general
performance claim follows from these measurements. The complete
[consumer report](evidence/improved-nystrom-backend.json) retains every case and
implementation hashes.

## Gates

Focused mechanics/history/switch suite: **181 passed**. Full CPU gate:
**8,458 passed, 639 skipped, 157 warnings**, 318.58 seconds with eight workers
and one numerical thread each. The [full report](evidence/improved-nystrom-full.json)
fingerprint matches the final runtime/tests/configuration. The
[installed-wheel gate](evidence/improved-nystrom-wheel.json) passes outside the
checkout, including IRKN continuation through the installed runner. All 53
published Python blocks execute; 13 caller-defined sketches are labeled. Catalog
and repository-link gates pass and Docusaurus builds successfully. GPU/device
coverage remains limited to the suite's available hardware and recorded skips.
