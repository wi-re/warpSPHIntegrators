> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.13 Phase 12: RKC / RKL — stabilized explicit super-timestepping — done 2026-09-11

The roadmap's "strongest omission": an explicit method that attacks parabolic
stiffness with no nonlinear solver at all. RKC (relaxed Chebyshev) and RKL (relaxed
Lobatto/Legendre) spend `s` matrix-free right-hand-side evaluations per step to buy a
real-axis stability interval that grows like `O(s²)` — so `dt` can be set by the
physics (the convective scale), not by the stiffest diffusive eigenvalue. For SPH
viscosity that is frequently the right answer over implicit, and it attacks exactly the
regime the Phase 2 preconditioning work and the `diffusion_problem(n)` benchmark exist
for.

Construction. Each method's one-step stability polynomial is a scaled, shifted
orthogonal polynomial in `z = dt·λ`: `R_s(z) = T_s(1 + z/s²)` for RKC1, and
`a_s + b_s·Φ_s(1 + w₁z)` for RKC2 / RKL2 (`Φ_s` the Chebyshev `T_s` or Legendre
`P_s`; the constant offset `a_s` is what lifts the order from 1 to 2). The real-axis
stability intervals are

    RKC1  K = 2s²            RKC2  K = 2(s²−1)/3           RKL2  K = (s²+s−2)/2

and all three are realised by a three-term (plus a cached `Y_0`) recurrence, so a step
costs exactly `s` right-hand-side evaluations and a few state buffers — no linear
solve, no matrix. The stage weights are chosen so that every *intermediate* stage
polynomial is itself bounded by 1 on `[−K, 0]` (no stage overshoots). RKL2 is the
scheme of Meyer, Balsara & Aslam, *J. Comput. Phys.* 257 (2014) 594–626 (Eqs. 15–19);
RKC2 is the identical construction with `T_j` in place of `P_j`; RKC1 is the order-1
Chebyshev limit (Ruuth, *J. Comput. Phys.* 169 (2001) 162–175). The full recurrence
and coefficient rules live in the `rkc.py` module docstring.

What landed:

- **`rkc.py`** — one shared core `integrateRelaxedChebyshev(state, dt, f, family, …)`
  and three thin drivers, `RKC1` / `RKC2` / `RKL2` (enum 52/53/54; registry order
  1/2/2, explicit, `dissipation=True`, no reuse). `stage_count(dt·|λ_max|, family)`
  returns the smallest admissible stage count, clamped to the family's minimum (1 for
  RKC1, 3 for RKC2/RKL2); the driver takes `s=` directly or `lambda_max=` and picks
  `s` — neither, and it raises. `priorStep` is refused with the standard warning.
- **Generic-suite skips, with reasons** (`conftest.py` `NEEDS_STAGE_COUNT`): the stage
  count is a *per-step* parameter the shared one-step tests do not provide, and the
  stability analysis is on the real axis, which the oscillator's imaginary eigenvalues
  do not exercise (RKC1 is marginally unstable on that problem for any `s`). Three
  tests that iterate the registry directly were adjusted for it: `test_rhs.py` skips
  schemes absent from the pre-Phase-14 bit-identical golden; `test_hamiltonian.py`
  skips the family in the oscillator area-property test; and `tvd_analysis.py`
  classifies it as "not applicable (parabolic super-timestepping)" — the hyperbolic
  TVD/SSP question does not apply to a method built for parabolic terms.
- **`tests/test_rkc.py`** (35 tests) — where the family is actually exercised:
  registered order on the diffusion, the stability boundary on a multi-mode random IC,
  the stage-count rule (smallest admissible, tight at every `K(s)` boundary up to
  `s = 43`, `s²` growth), `s=` ↔ `lambda_max=` equivalence, minimum clamping, the
  missing-`s` error, the `priorStep` warning, exactly `s` RHS evaluations per step,
  stage times inside `[t, t+dt]` opening at `t`, no caller-state mutation, and a
  semilinear-RHS (viscous Burgers) run that conserves mass to round-off.

Verification. The recurrences were validated *before* the driver was written (scalar
stability polynomial to round-off, intermediate-stage boundedness, order on the
semilinear diffusion); the tests pin the same through the real machinery. Measured on
the semi-discrete diffusion (n = 32, `s = 12`, T = 0.02):

- **Order**: 0.930 (RKC1), 2.137 (RKC2), 2.126 (RKL2). The order is independent of
  the stage count; `s` only sets the stability interval.
- **Stability boundary** (random multi-mode IC, n = 16, `s = 8`): the run is bounded
  just inside and blows up just outside the predicted interval — the measured blow-up
  sits at `dt·|λ_max| / K(s) = 1.007–1.011`, i.e. the `K(s)` prediction holds to about
  one percent.

The benchmark (the roadmap gate). `benchmarks/rkc_benchmark.py` →
`docs/assets/rkc_benchmark.png`: total RHS evaluations (finite-difference JFNK matvecs
included) to reach max error ≤ 1e-2 on `diffusion_problem`, T = 0.1:

| n (|λ_max|) | RKC1 | RKC2 | RKL2 | TR-BDF2 | BE | BDF2 |
|-------------|------|------|------|---------|-----|------|
| 16 (1146)   | 32   | 40   | 48   | 89      | 268 | 224  |
| 32 (4346)   | 64   | 80   | 88   | 147     | 399 | 896  |
| 64 (16890)  | 128  | 144  | 168  | 271     | 710 | unstable |

Two findings:

1. **The family is the cheapest way to a fixed error at every n.** RKC/RKL reach
   1e-2 with fewer total RHS evaluations than any implicit baseline — the implicit
   cost is dominated by the nonlinear solve (finite-difference Jacobian matvecs in
   the default JFNK), which RKC/RKL do not have. Order-1 RKC1 is cheapest overall
   (fewest stages); RKC2/RKL2 add an order of accuracy at a modest stage-count
   premium (12–50% more total evaluations depending on `n`).
2. **BDF2's practical stability on this benchmark is solver-limited, not
   method-limited.** BDF2 is A-stable, yet its default JFNK (finite-difference
   matvec, no preconditioner) fails from the first step at coarse `dt` (n = 16,
   dt = 0.05: max|u| 1.44 → 1336, with z ≈ 11.8 deep inside the A-stable region), at
   n = 32 for dt ≥ 0.001563, and at n = 64 at every `dt` tested. The benchmark marks
   such runs unstable and reports BDF2's n = 64 as "no stable point reaches the
   target." That solver ceiling — not the stability region — is the wall the Phase 2
   preconditioner work is meant to lower on the implicit side, and it is the regime
   RKC/RKL sidestep entirely.

Caveats. The stage times are the autonomous-domain approximation (stage `j` at
`t + dt·j/s`); for a non-autonomous right-hand side this is an approximation. RKL2
and RKC2 prefer an *odd* stage count — for even `s` the smallest-wavelength mode is
only weakly damped (Meyer et al. 2014); the driver clamps to the minimum but does not
force parity, so pass `s=` (or a slightly larger `lambda_max=`) when you want odd.

Suite: 2365 → 2400 passing (the 35 new tests are `tests/test_rkc.py`).
