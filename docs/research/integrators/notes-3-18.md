> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.18 Phase 6: coupled fully implicit RK (Gauss-Legendre 2, Radau IIA s=2) — done 2026-09-15

Phase 6 of `docs/plans/implicit.md` lands the two coupled fully implicit RK
tableaus the library could not run before: **Gauss-Legendre 2** (order 4,
A-stable, symplectic for separable Hamiltonians, *not* L-stable) and **Radau
IIA s=2** (order 3, L-stable, stiffly accurate). A DIRK solves its `s` stage
equations as `s` *sequential* solves because `a_ij == 0` for `i != j`; both
shipped tableaus have nonzero off-diagonal entries, so stage `i` references the
still-unknown stage `j` and the `s` equations are one `s`-by-`s` coupled
system. That is a different *solver shape*, not more tableau data — hence the
new driver, `src/warpSPHIntegrators/fullyimplicit.py`, and the new product
type `fields.BlockState`.

**`BlockState` and the `fields.py` bridges.** A `BlockState` holds `s` full
tagged sub-states (one per stage). The four state↔vector bridges
(`flatten_integrated`, `unflatten_integrated`, `integrated_field_names`,
`replace_integrated_fields`) recurse over the substates in substate-major
`'i:name'` layout — each sub-state keeps its full field structure, so `copied`
fields, tags and `apply_*_update` work per stage and the stacking is only at
the flatten boundary the DIRK solver already crosses. `state_difference` /
`state_norm` accept a block against a block and pool the per-substate
contributions. `tests/test_fullyimplicit.py` pins the round-trip, the pooling,
and the mismatched-shape refusals.

**The driver contract.** `fullyImplicitBlock` runs the whole step as one
`JFNKSolver.solve` call (JFNK by default; a `FixedPointSolver` works but is
not a stiff solver). The residual map `step_fn` evaluates `f` at *all* `s`
stage states per call — one map evaluation is `s` RHS evaluations — so
`IntegrationResult.solver_diagnostics` carries **one entry per step with the
counters scaled by `s`** (`rhs_evaluations` and `gmres_iterations`), mirroring
how the DIRK driver appends one entry per stage. `priorStep` (first-stage
reuse) is refused for every coupled tableau — there is no explicit first stage
to splice a previous step's derivative into — and `warmStart=` takes the
previous step's `IntegrationResult.stages` as the block solve's *initial
guess* instead: `Y_i ~= y^n + dt·Σ_j a_ij k_j(previous)`. It is exposed as a
plain flag (not reuse) because it changes the iteration count, never the
answer: on stiff problems with similar consecutive steps it is a materially
better starting point than the cold guess `y^n`, and where it does not help it
costs nothing (`test_warm_start_changes_the_guess_not_the_answer`). `history=`
is bookkeeping only.

**The tableaus, hand-derived.** Neither exists in SUNDIALS ARKODE v7.9.0's
published set, so the coefficients come from the collocation derivation with
the order conditions and the stability functions checked symbolically and
numerically (the SDIRK2 "verified by hand" precedent):

- **Radau IIA s=2** — collocation at the Radau points `{1/3, 1}` (the roots of
  `3x² − 2x − 1`, right endpoint pinned, so the `s = 1` limit is exactly
  backward Euler). `c = (1/3, 1)`,
  `a = [[5/12, −1/12], [3/4, 1/4]]`, `b = (3/4, 1/4)`. Order 3, A- and
  L-stable, stiffly accurate (`b = a`'s last row, `c_2 = 1`), coupled
  (`a_12 = −1/12` — a DIRK driver structurally cannot take it). Stability
  function `R(z) = (1 + z/3)/(1 − 2z/3 + z²/6)`: `R(−1) = 4/11`,
  `|R(−10)| = 0.095890` (pinned exactly in
  `tests/test_fullyimplicit.py`), `|R(−100)| ≈ 0.019` — L-damping
  `~ C/|z|`.
- **Gauss-Legendre 2** — collocation at the Gauss points `1/2 ± √3/6`.
  `c = (1/2 − √3/6, 1/2 + √3/6)`, `b = (1/2, 1/2)`,
  `a = [[1/4, (3−2√3)/12], [(3+2√3)/12, 1/4]]`. Order 4, A-stable with
  `|R(iy)| = 1`; **not** L-stable: `R(−100) = 2353/2653 ≈ 0.88692` and
  `R(z) → +1` as `z → −∞` (it bounds the solution but does not damp it — the
  same class as the trapezoidal rule, whose `|R(−100)| = 49/51`). Not
  stiffly accurate (`c_2 ≠ 1`). `R(−10) = 0.302326`. **The tableau is not
  symmetric** (`a_12 = (3−2√3)/12 ≈ −0.0387 ≠ a_21 = (3+2√3)/12 ≈ 0.5387`) —
  the "A-symmetric" claim that circulates for Gauss-Legendre is false for the
  collocation tableau, and the symplecticity below is therefore pinned by
  *measurement*, not asserted from a tableau property.

**The error estimate: a correction to the roadmap.** The roadmap item said
"Radau IIA s=2 has the standard Hairer-Wanner estimator". It does not, as a
2-stage-only pair: for *both* shipped tableaus the 2-stage-only order-2
quadrature pair is degenerate — the unique order-2 weights over `(k_1, k_2)`
are `b` itself. Both schemes therefore ship the **null-stage order-2
companion**: the min-norm order-2 quadrature over `(k_0, k_1, k_2)` with the
null stage `k_0 = f(t^n, y^n)` at `c_0 = 0` — Radau
`b̄ = (2/7, 9/28, 11/28)`, Gauss-Legendre
`b̄ = (1/6, (5−√3)/12, (5+√3)/12)`. `companion_b[0] ≠ 0` costs one extra RHS
evaluation per step; it is evaluated *before* the converged block is
re-evaluated, so the step's last RHS evaluation is the last *stage* — the
`copied()` field contract ("copied from the last substep in finalize",
`tests/test_copied_fields.py`) pins exactly that, and the companion is an
estimator detail that must not reorder the method's stages. The estimate is
main-minus-companion `O(dt^3)` for *both*: measured rate exactly `3.0` (±0.02) on
`oscillator`/`forced`/`kepler` at `T = 0.5` and `T = 2.0`
(`test_companion_estimate_converges_at_rate_three`). In the Phase 11 `q`
language that is `q = order − 1 = 3` for Gauss-Legendre 2 and `q = order = 3`
for Radau IIA s=2.

**Symplecticity, measured.** On the linear oscillator, preserving `dq ∧ dp`
is `det(DΦ_h) = 1`. With the default JFNK tolerances Gauss-Legendre 2's area
defect is `4.29e-06` — that is solver noise, not method error: the strict
solve (`tol = gmres_tol = newton_tol = 1e-12`, `fd_eps = 1e-6`) recovers a
defect of **`3.05e-13`**, i.e. area-preserving to machine precision, and the
direct nonlinear Kepler symplectic-form check passes at `4.87e-09`
(`< 2e-6`) with the *default* solver — Gauss-Legendre 2 joins
`NONLINEARLY_SYMPLECTIC` in `tests/test_hamiltonian.py`. The same strict-solve
caveat shows up in the long-run energy test: on `kepler` the default solve
stagnates in a `~5e-8` drift band (`6.9e-09` at `T = 10`, `5.5e-08` at
`T = 80`, growth `8.0` — just above the `1e-8` noise floor, so the
growth-ratio test misreads it as dissipative), while the strict solve is flat
at `4.33e-12` on both horizons (growth `1.001`). The test therefore carries a
documented skip for Gauss-Legendre 2 on `kepler` only (the oscillator run
stays under the floor at `4.6e-10`), with the symplecticity pinned by the two
direct area tests instead. Radau IIA is deliberately *not* in the symplectic
sets: its measured defects are `2.44e-02` (linear area) and `3.65e-05`
(Kepler form) — L-damping is its job, and `dissipation=True` is the right
flag for it.

**JVP matvec and the scale gate.** The block Jacobian-vector product is the
forward-mode JVP seeded across the `s` substates (via
`replace_integrated_fields` on dual clones) — `JFNKSolver(matvec='jvp')` is
the registered default, and it agrees with the finite-difference matvec to
`5.3e-09` relative at 256 DOF, is linear to `5.7e-14` and deterministic, and
matches the hand-computed dense block Jacobian
(`test_jvp_matvec_matches_the_hand_computed_dense_jacobian`). The scale gate
(`test_scale_gate_at_256_dof`: 256-DOF advection, CFL = 100, step IC,
`u_0 = 1.0`, `JFNKSolver(matvec='jvp', max_iterations=50,
gmres_restart=1536)`) resolves to `termination == 'tolerance'` for both
schemes — Radau residual `7.5e-10`, `max|u| = 1.11` (bound `1.5·u_0`);
Gauss-Legendre residual `1.7e-10`, `max|u| = 1.69` (bound `2.0·u_0`). The
Gauss-Legendre overshoot is the *true method value*, not a solve defect: its
negative `a_12` makes the implicit stage solution non-monotone, so a step
initial condition excites a bounded Gibbs-like overshoot even though every
per-mode amplification `|R(z)| ≤ 1` (the exact flow is row-stochastic and
keeps `max|u| = 1`). No dense Jacobian is ever allocated — the gate runs
matrix-free at 1536 block unknowns.

**The GMRES finding behind the gate.** At 256 DOF, the *default*
`JFNKSolver(matvec='jvp')` (GMRES restart 30) stagnates on this system —
Radau at residual `6735` (a garbage answer the old gate would have accepted
as `stagnation`), Gauss-Legendre at `4.75e-06`. The diagnosis, pinned in the
test docstring: the block Jacobian `J_G` for the 256-DOF advection step is a
`1536 × 1536` matrix with condition number `88.56` and `|λ| ∈ [0.931, 82.47]`
— **128 complex conjugate pairs spread over a 2-D annulus** plus an
eigenvalue-1 cluster, and restarted GMRES (restart 30 or 60) fails on that
spread spectrum while full GMRES (restart ≥ n) converges in 256 iterations to
residual `1.1e-12`. It is not a bug in `jfnk.gmres` or the matvec: SciPy's
own `gmres` fails identically (restart 30, `maxiter = 1536` → `info = 1536`,
residual `1.1e-03`; no restarts → `info = 0`, `8.3e-13`), and SciPy's early
per-iterate trajectory is actually *slower* than ours. The test-side fix is
`gmres_restart = 2·3·256` (the flat block dimension, so no restarts) plus
`termination == 'tolerance'` asserted — a `stagnation` exit on this system is
a failed solve, not a floor.

**Stiffness and TVD.** Both are A-stable, so on the stiff nonlinear
relaxation (`rate = 100`, `dt = 0.1`, `z = −10`,
`tests/test_stiff.py::STATE_SPACE_IMPLICIT`) the stiff transient decays as
`|R(−10)|^n` — Radau to `~1e-10` in one step (`0.095890^10 ≈ 7e-11`),
Gauss-Legendre in ~4 (`0.302326^4 ≈ 0.008`) — while the slow `tanh`
component is tracked at full order. In the Phase 13 TVD classification both
land at `(ssp=None, tvd=5.0, unconditional=True)`: no tableau attribute for
the SSP convex-combination scan (a `BlockTableau` is not a Butcher/DIRK/ARK
tableau), and A-stability makes every per-mode amplification `≤ 1`, so the
measured sweep finds no TV increase to the CFL cap.

**Order, cost, and the benchmark.** Measured order on the linear oscillator:
Gauss-Legendre `3.98`, Radau `3.00`; on the nonlinear `kepler`/`forced`
problems the driver order holds (`test_driver_order_on_nonlinear_problems`).
At equal order the block pair carries the smallest error constants in the
registry (at `dt = 0.0125`, `T = 2`: Gauss-Legendre `4.5e-09` vs RK4 `2.7e-08`
and ESDIRK4(3)6 `2.7e-09`; Radau `1.9e-06` vs ESDIRK3(2)4 `3.5e-06` and
TR-BDF2 `2.1e-04`). The full comparison is
[`benchmarks/blockrk_benchmark.py`](../../../benchmarks/blockrk_benchmark.py) →
[`docs/assets/blockrk_benchmark.png`](../../assets/blockrk_benchmark.png): (1) max energy
drift vs horizon on the oscillator — the symplectic set (Velocity Verlet,
PEFRL, Gauss-Legendre) stays in its band while RK4 grows secularly exactly
linearly in `T`; (2) the stiff transient decay at `z = −10` — L-stable
schemes hit their order floor in a few steps (BDF3-5 carry the shared DP5
startup spike, §3.7), A-stable-not-L bounds it at `|R(−10)|` per step;
(3) the order ladder.

**New tests / artifacts:** 32 in `tests/test_fullyimplicit.py` (the block
bridges, both tableaus' structure and stability functions, driver order on
nonlinear problems, the companion rate, warm start, JVP vs FD vs dense, the
scale gate, the reuse refusal, the scaled diagnostics); the two schemes join
`tests/test_embedded.py` (estimate rate `3.0`), `tests/test_adaptive.py`
(`q = 3`), `tests/test_stiff.py` (the stiff relaxation, the Dahlquist origin),
`tests/test_hamiltonian.py` (the symplectic sets, the strict-solve area, the
Kepler-form, the `kepler` skip), `tests/test_gradients.py` (gradient flow,
bitwise forward invariance), and `tests/test_tvd.py` (the verdict table);
`benchmarks/blockrk_benchmark.py` → `docs/assets/blockrk_benchmark.png`.

**Still open (gated):** Gauss-Legendre s=3 and Radau IIA s=3 wait on the
block solver and preconditioner being proven at scale; Lobatto IIIA-IIIB
waits on a concrete partitioned/separable-Hamiltonian downstream. The
`warmStart` flag is the block analogue of a good initial guess, not first-stage
reuse — the reuse machinery stays DIRK/explicit-only.

---
