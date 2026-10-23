# Browser coefficients and relative CFL controls — 2026-10-23

The laboratory inspector now opens and exports a coefficient record for all 280
browser configurations. Tableaus come directly from the generated catalog;
Adams/BDF/ROS3P native weights are also exported from Python. The remaining maps
use the browser drivers' native conventions with explicit notation and source
paths. Parameterized recurrences and structural maps capture the current stage
count, beta/gamma or rho-infinity. A modal snapshot cannot change under a hover.

The CFL inspector distinguishes documented SSP constants from rounded upwind
model convexity measurements. Unknown constants are not assigned zero. A known
constant and an appropriate FE foundation show the dimensional bound and h/bound.
No FE monotonicity bound is claimed for centered acoustics or viscous Burgers.

The left input converts both directions between h and transport/acoustic CFL or
diffusion ratio. Burgers uses the initial max speed and says so. ODEs report no
CFL definition. Both h and relative-number buttons, plus keyboard arrows, use
magnitude-dependent small increments; h buttons clamp at the existing limits.

Validation: browser tests cover all exports, source weights, structural parameter
and recurrence snapshots, SSP/model/unknown distinctions, problem conversions
and microsecond-scale increments. Existing numerical and source-reference checks
remain unchanged. Local browser QA checked SSPRK(10,4) viewing, JSON download and
clipboard copy; 0.0000125 → 0.0000135; CFL 0.8 → 0.81; CFL 1.6 ↔ h 0.025 on
64 upwind cells; and generalized-alpha coefficients at rho-infinity 0.8.
