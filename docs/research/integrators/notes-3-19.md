> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.19 Phase 12: IMEX linear multistep (SBDF2, SBDF3, CNAB2) — done 2026-09-15

Phase 5's IMEX family went from IMEX Euler straight to the two ARK pairs, with
nothing multistep in between — and the multistep schemes are what PDE and fluid
codes actually run. This lands the three classical semi-implicit multistep
schemes: a BDF2 / BDF3 / trapezoidal *implicit* backbone for the stiff part, the
smooth part *extrapolated explicitly* to the endpoint from its own previous
endpoint values.

**The methods** (`src/warpSPHIntegrators/imexmultistep.py`, registered as
`SBDF2`, `SBDF3`, `CNAB2`):

| scheme | implicit backbone (stiff part) | explicit extrapolation (smooth part) | order |
|--------|--------------------------------|--------------------------------------|-------|
| SBDF2  | BDF2 (L-stable)                | `2Nⁿ − Nⁿ⁻¹`                         | 2     |
| SBDF3  | BDF3 (A(α) 86.03°)             | `3Nⁿ − 3Nⁿ⁻¹ + Nⁿ⁻²`                 | 3     |
| CNAB2  | trapezoidal (A, not L)         | AB2 increment `3Nⁿ/2 − Nⁿ⁻¹/2`       | 2     |

**The form, and the β trap.** SBDFp is

    y^{k+1} = Σ_j c_j y^{k−j} + β·dt·[L(t^{k+1}, y^{k+1}) + E(N)],

where the `c_j` are the BDF state weights, `β` the BDF derivative weight
(3/2 for BDF2, 12/5 for BDF3), and `E` the `p`-point Lagrange polynomial through
the known endpoint values extrapolated one point ahead (weights `[2, −1]` and
`[3, −3, 1]` — exact for every polynomial of degree `< p`, which characterizes
them). The key consistency point: **`β` scales the *whole* endpoint derivative,
implicit and extrapolated-explicit part alike.** Drop `β` from the explicit part
and that part is amplified by `1/β` and the combined method is first order
(measured as an O(1) error plateau, pinned as a regression guard in
`tests/test_imexmultistep.py`). CNAB2 carries the same consistency in the
trapezoidal rule's own weights: `y^{k+1} = y^k + (dt/2)[L^k + L^{k+1}] +
dt[(3/2)N^k − (1/2)N^{k−1}]` — trapezoidal on the implicit part, AB2 on the
explicit part.

**Pure limits (the Phase 5 compatibility rule, verified).** Handed an ordinary
RHS callable (no split), each member runs its pure-implicit limit: **SBDF2/3 are
bit-exact against the registered BDF2/BDF3** (same formula, same solve path, same
initial guess) and CNAB2 matches the registered trapezoidal scheme to solver
tolerance (~1e-5 — same fixed point, different solve starts). The reverse limit —
the implicit part zeroed — gives the *effective explicit method*: CNAB2's is
**bit-exact against the registered Adams-Bashforth 2**, while SBDF2/3's is *not*
the Adams-Bashforth method of the same order but the matching
Lagrange-extrapolation multistep method (the zero-stable companion of the BDF
state combination). All three effective explicit methods are zero-stable.

**The explicit half is the practical restriction.** The pure-explicit limits'
stability regions are pinned in `tests/test_imexmultistep.py`:

- zero-stable (Dahlquist root condition at `z = 0`; the root at 1 simple);
- negative-real-axis boundaries, exact: **SBDF2 = 4/3**, **SBDF3 = 20/21**
  (measured 0.952381 — the rational 20/21), **CNAB2 = 1** (AB2's own);
- the imaginary axis is *not* a stability direction at O(1) — `|R(i)| > 1` for
  all three — and the region crosses the axis only in a whisker with O(y⁴)
  per-step growth. The practical restriction is `dt·max|λ_E| ≲ O(0.5)`;
- end-to-end on the actual problem (implicit part zeroed, the n = 64 Burgers
  grid where the convective spectrum is `|z| ≤ 64·dt`): the SBDF2 run at
  `dt = 0.008` (`|z| ≤ 0.512`) stays bounded for 400 steps (peak |u| ~ 0.98)
  while `dt = 0.01` (`|z| ≤ 0.64`) runs to NaN.

This is why the family's showcase runs (viscous_burgers_demo.ipynb §7) step at
`dt = 0.008` (SBDF2, CNAB2) and `dt = 0.004` (SBDF3) instead of the ARK pair's
0.02 — the *diffusion* is no longer the constraint, the convective spectrum is.

**History and cold start follow the BDF family exactly.** `history=` is threaded
with state snapshots of the previous `order − 1` steps (the `steps` metadata is
the history depth, the BDF convention); a cold call runs one Dormand-Prince 5(4)
step and records a snapshot, so a run is never silently wrong — just at the
starter's cost until the history fills (S3.7 pain point 1). The explicit part is
*re-evaluated at the history snapshots* every step rather than stored (no
history-format change); a `dt` change invalidates the snapshot history and the
scheme re-bootstraps (the BDF contract, `tests/test_bdf.py`); `priorStep` is
refused with the standard warning — multistep reuse is the `history=` mechanism,
not first-stage splicing.

**Diagnostics.** One solve per step (the implicit backbone), and
`SolveDiagnostics.rhs_evaluations` is increased by the step's non-solve
evaluations (the explicit-part evaluations, CNAB2's implicit start-point
evaluation, and the endpoint evaluation recorded for the history), so the
Phase 1 work-unit invariant `total = rhs_evaluations + gmres_iterations` covers
the full step cost. (The plain BDF family counts the solve only; SBDF/CNAB pay
for the split, and this makes that visible.)

**Measured.** Orders on the canonical split (diffusion implicit, convection
explicit, the n = 64 viscous Burgers problem, `T = 0.4`, exact-Jacobian solver
`tol = newton_tol = 1e-12`): SBDF2 → 2.03, SBDF3 → 3.02, CNAB2 → 2.01
(viscous_burgers_demo.ipynb §10; `tests/test_imexmultistep.py` pins the same
orders against the claimed ones, and pins the pure-implicit limits' orders on
the non-autonomous `forced` problem, which catches grid-time handling). Showcase
cost (sine IC, `T = 0.4`): SBDF2 @ 0.008 — 50 steps, 343 RHS evals, 370 GMRES
iters (total 713); SBDF3 @ 0.004 — 100 steps, 784/507 (1291); CNAB2 @ 0.008 —
50 steps, 343/330 (673). Their per-step price is the lowest in the demo's cost
table (the implicit system is the fixed linear diffusion operator, not the full
nonlinear `f`), but the small `dt` multiplies it out: at this `n = 64` size,
where the RHS is cheap, fully-implicit BDF2 (393 at `dt = 0.02`) still wins on
raw cost — the split pays off when the nonlinear part is expensive relative to
the linear solve (the demo's §11 has the full accounting).

**New tests:** 34 in `tests/test_imexmultistep.py` — the Lagrange extrapolation
weights (exactness for degree `< p`, the degree-`p` defect, rejection of
unsupported orders); the measured orders on the split problem and on the
pure-implicit limits; the pure-limit equivalences (bit-exact BDF2/BDF3,
trapezoidal to tolerance, AB2 bit-exact); the pure-explicit-limit stability
(zero-stability, the exact 4/3 / 20/21 / 1 negative-real-axis boundaries, the
weak imaginary-axis whisker, the end-to-end bounded/NaN pair on the convective
spectrum); and the driver contract (cold start, state snapshots, dt-change
re-bootstrap, priorStep refusal, the split-evaluation diagnostics count).

**Still open (gated, Phase 12's remaining families):** generalized-alpha /
HHT-alpha (structural/solid SPH), SSPRK(5,4) (SSPRK(10,4) landed 2026-09-15 —
§3.20), high-order symplectic composition and one-step RKN (both gated on a
*separable* Hamiltonian downstream, which an SPH momentum equation with
velocity-dependent forces is not — the Verlet family measures first order under
such forces, S3.6).
