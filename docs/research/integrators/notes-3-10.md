> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.10 Phase 8: the nonlinear/stiff benchmark suite — done 2026-09-09

The roadmap's Phase 8 (stability and nonlinear benchmark suite) is complete.
What landed:

- **Five new problem factories in `testing.py`** (the `PROBLEMS` registry now has
  nine). `stiff_relaxation_problem(rate, forcing, sign)` is Prothero–Robinson in
  both signs: the stable variant (eigenvalue `−rate`, exact solution the moving
  target `s(t)`) and the unstable-PR variant (`sign=−1`, eigenvalue `+rate`, exact
  still `s(t)`). `stiff_damped_oscillator_problem(omega, c)` separates high
  frequency (`ω=50, c=1`) from true dissipative stiffness (`ω=1, c=50`) and has
  the closed-form exact solution in all three damping regimes. `van_der_pol_problem(mu)`,
  `robertson_problem()`, and `diffusion_problem(n)` round it out; the last starts on
  Laplacian eigenvectors 1 and 5 so the *excited* spectrum — not the full one — is
  what the stability boundary has to contain.
- **Damped-oscillator amplification matrices** in `stability.py`
  (`damped_oscillator_amplification_matrix`, `damped_oscillator_spectral_radius`)
  for leapfrog, velocity Verlet, symplectic Euler, and Newmark (β = 1/4 and 1/6),
  cross-checked against the registered schemes' actual one-step maps
  (basis-state numerical matrices agree to 1e-8/1e-10) and against the undamped
  Phase 6 results at `ζ = 0` (1e-12).
- **`tests/test_benchmarks.py`** — 52 tests, all small and deterministic (25 s):
  the registry, PR in both signs, the three damping regimes, van der Pol at
  μ = 2 and μ = 10, Robertson invariants and scheme agreement, diffusion, three
  explicit-wall tests, and the amplification-matrix checks.
- **`benchmarks/stiff_benchmark_suite.py` → `docs/assets/stiff_benchmark_suite.png`** —
  error vs `dt` on the four stiff benchmarks, per-step JFNK cost panels, and a
  van der Pol energy-vs-time panel (bounded bands for the settling orbits, the
  diverging orbits leave the capped axis within `t < 2`); all parameters and
  solver settings are stated in the figure caption.
- **`benchmarks/oscillator_stability_gallery.py`** gained the damping-ratio figure
  (`docs/assets/oscillator_stability_damped_newmark_verlet.png`): `log10(ρ)` over
  `(h·ω, ζ)` for the five methods with the `ρ = 1` boundary contoured.

Findings worth keeping:

- **Robertson needs a bootstrap, and BE has a hard ceiling on it.** The initial
  quasi-steady layer (width ~1e-3) is not crossable from `y(0) = (1, 0, 0)` at
  `dt ≥ 0.02`; ten `dt = 0.001` steps put the solution on the QSS manifold and the
  main `dt` takes over. Backward Euler at `dt ≥ 2` then converges *every* JFNK
  solve to an unstable discrete fixed point of the stiff quadratic and the map
  diverges ~2×/step — BE inaccuracy at `dt` above the fast scale (fast eigenvalue
  ~2.2e3), not a solver failure. `dt = 0.2` is strictly positive for all components
  over T = 100.
- **ESDIRK6's stability function has a pole at `z = 1`.** The stiffly-accurate
  `γ = 1` stage puts a pole in `R(z)`, so on the *unstable* PR (`z = +5`) ESDIRK6
  diverges (|x| ~ 9e29 in five steps) while A-stability says nothing about the
  positive real axis. DP5 amplifies the unstable mode by `R(+5) = 117.5` per step.
  Only BE/BDF-type methods track the unstable PR, where BE is the natural choice
  (`R(z) → 1/2` as `z → +∞`).
- **Damping reshapes the Verlet-family stability pictures in opposite directions.**
  Velocity Verlet at `h·ω = 2.5` (undamped eigenvalues −0.25 and −4.0, `ρ = 4`)
  has a stable window `0.5055 < ζ < 0.8` — at `ζ = 0.6` the pair is complex with
  `|λ| = √det = 0.5`. Leapfrog's damping term sits at the *old* velocity, so it is
  itself an explicit update unstable for `c = 2ζ·h·ω > 2`: no damping ratio
  rescues it. Newmark (average acceleration) is neutrally stable undamped for
  every `h·ω` and becomes asymptotically stable for any `ζ > 0`.
- **Van der Pol at μ = 10 has a discrete-attractor wall.** ESDIRK6 tracks the
  amplitude-~2 cycle at `dt ≤ 0.2` (errors 0.53 / 1.22 / 3.4 vs a `dt = 0.01`
  reference at `dt = 0.05 / 0.1 / 0.2`) and diverges at `dt = 0.4` — the energy
  overflows by `t ≈ 2` and the state reaches NaN well before T = 50; DP5 at
  `dt = 0.5` leaves the O(1) band within a few steps. At μ = 2, by contrast,
  DP5 tracks the cycle at `dt = 0.05` — the explicit restriction is not active
  until μ grows.
- **Per-step JFNK cost (PR, rate = 100, `dt = 0.01`, 100 steps, FD matvecs,
  GMRES tol 1e-8):** GMRES iterations per step are exactly the stage-solve counts
  (1.00 / 0.99 / 2.00 / 5.00 for BE / BDF2 / TR-BDF2 / ESDIRK6) with zero
  line-search backtracks; RHS evaluations per step are 3.40 / 2.96 / 4.50 / 10.72.
  ESDIRK6's per-step cost is ~3× BE's, so its accuracy advantage on these
  benchmarks is bought at a real price. (This metric counts only the implicit
  stages' `rhs_evaluations`; the DIRK explicit first stage is recorded as `None` and
  is excluded, so Phase 10's first-stage reuse — which saves that explicit evaluation
  — does not move these numbers. See §3.6's Phase 10 note.)
