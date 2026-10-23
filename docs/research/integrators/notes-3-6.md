> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.6 Valid schemes and what each costs

Effort is *marginal*, on top of the groundwork and the driver its group needs.
"tableau only" means the scheme is a data entry in `getButcherTableau` plus a
registry line — the same one-line cost that adding Dormand–Prince was.

#### Diagonally implicit RK (sequential 1-stage solves) — all seven shipped (four 2026-08-24, the rest 2026-09-09)

The probe driver is ~60 lines; budget ~150 for a registered one with the solver
protocol, verbose output and scheme metadata wired in. **Landed as `dirk.py`
(~340 lines including the 2026-09-09 tableaus) + seven registry entries**, reusing
`butcher.py`'s `_weighted_update`,
`_error_estimate` and `finalizeSystem` machinery directly rather than duplicating it,
per the design sketched here.

| Scheme | Order | Stages | Stability | Symplectic | Status |
|---|---|---|---|---|---|
| **Implicit midpoint** (Gauss–Legendre s=1) | 2 | 1 | A | **yes, exactly-solved** | **Done.** Registered `dissipation=True` — see the finding below. |
| **Backward Euler** | 1 | 1 | L | no | **Done.** |
| **Trapezoidal / Crank–Nicolson** (Lobatto IIIA-2) | 2 | 2 | A, not L | symmetric | **Done.** Explicit first stage (`a11=0`) exercises the driver's non-Picard branch. |
| **SDIRK2** (Ellsiepen, γ=1−√2/2) | 2 | 2 | L | no | **Done.** |
| **TR-BDF2** | 2(3) | 3 | L | no | **Done 2026-09-09.** Embedded pair is SUNDIALS ARKODE's published (2, 3) pair. |
| **ESDIRK3(2)4L[2]SA** (Kennedy–Carpenter) | 3(2) | 4 | L | no | **Done 2026-09-09.** Coefficients from SUNDIALS ARKODE v7.9.0; verified three ways (below). |
| **ESDIRK4(3)6L[2]SA** | 4(3) | 6 | L | no | **Done 2026-09-09.** Same source and verification as ESDIRK3(2)4. |

**TR-BDF2 and both ESDIRKs landed 2026-09-09**, once the transcription concern
below was resolved by a public, byte-for-byte reproducible coefficient source:
SUNDIALS ARKODE v7.9.0's `src/arkode/arkode_butcher_dirk.def` (the Kennedy–Carpenter
tableaus, entries `ARKODE_TRBDF2_3_3_2`, `ARKODE_ESDIRK324L2SA`,
`ARKODE_ESDIRK436L2SA`). The original concern still stands as the reason: both are
embedded, higher-stage tableaus whose published coefficients are easy to transcribe
wrong in a way a smoke test would not obviously catch — an order-2/3 convergence
measurement looks the same whether the low-order embedded weights are exactly right
or merely close, since only the *high-order* weights drive the propagated solution.
Each of the three was therefore verified beyond convergence: (1) the embedded
weights satisfy the low branch's order conditions exactly (e.g. TR-BDF2's `d`:
sum(d) = 1, d·c = 1/2, d·c² = 1/3); (2) one step expanded against the exact Taylor
solution via a symbolic Taylor model gives the claimed main and embedded order
(exact for the 4-stage ESDIRK3(2)4; numeric local-error rates against five
closed-form ODEs for the 6-stage ESDIRK4(3)6, where the exact-√2 arithmetic is too
slow symbolically); and (3) the stability function, computed exactly as a rational
function, is A-stable (|R(z)| ≤ 1 on the left half-plane, on a sweep fine enough
that grid artifact is ruled out) and L-stable (|R(-100)| = 2.65e-2 for
ESDIRK3(2)4, 7.57e-2 for ESDIRK4(3)6, 4.41e-2 for TR-BDF2; decay ~ C/|z|). One
naming note: the published names say "L[2]", but the exact stability functions
decay as O(1/|z|), so these register `stability='L'` and no O(z⁻²) claim is made.
The four 2026-08-24 tableaus were each verified two ways: by hand against their own
order conditions before writing any code, and empirically via `testing.convergence`
after (all four reach their claimed order to within 0.01 on `oscillator`/`forced`/
`damped` — `tests/test_dirk.py`).

**`reuse.py` was deliberately *not* touched, and the DIRK schemes deliberately do not
expose `.butcherTableau`.** That attribute is what `reuse._tableau_of` looks for to
run the explicit-scheme substitution analysis, whose reasoning assumes a stage is a
plain function of already-known states — not true for an implicit stage, whose value
depends on `dt` through the very solve reuse would try to skip. Leaving the attribute
off makes `step_reuse_analysis` correctly fall through to "no tableau, no recorded
reuse behaviour" rather than silently misapplying the explicit-tableau formula to an
implicit one. DIRK did not implement first-stage reuse at all then; `priorStep` was
rejected with the same warning every other non-reuse scheme in this library gives.

**Update (2026-09-10, IMPLICIT_ROADMAP Phase 10):** that "not yet" is now implemented
for the four stiffly accurate tableaus with an explicit first stage — Trapezoidal,
TR-BDF2, `ESDIRK3(2)4L[2]SA`, `ESDIRK4(3)6L[2]SA`. Each satisfies both halves of the
implicit FSAL condition: `a[0,0] == 0` / `c[0] == 0` (the first stage is explicit) *and*
`c[-1] == 1` / `a[-1] == b` (stiff accuracy), so the previous step's last stage *is*
`y^{n+1}` and its converged derivative is exactly `f(t^{n+1}, y^{n+1})` — the value this
step's first stage needs. Reuse is therefore lossless, saving one (cheap, explicit) RHS
evaluation per step. The split mirrors the design note above: `dirk.py` decides reuse
from the tableau alone (`_dirk_reuses_first_stage`, so the call-time check cannot drift
from the registration), while `reuse.py` gained a separate `dirk_reuse_analysis` — *not*
an extension of the explicit-tableau formula, which still does not apply to implicit
stages. It gates on the registered `stiffly_accurate` flag (that metadata's first real
consumer) with the same tableau check as a drift guard, and `step_reuse_analysis`
dispatches to it via `.dirkTableau` (now copied onto the reuse-guarded wrapper alongside
`.butcherTableau`). SDIRK2 (`a[0,0] != 0`, implicit first stage) and the single-stage
backward Euler / implicit midpoint still reject `priorStep` with the standard warning.
Measured (`tests/test_dirk.py`): reuse preserves the measured order on both problems, the
reuse run agrees with the no-reuse run to ~1e-11 on `oscillator` and exactly on `forced`
(the gap is JFNK's last-bit sensitivity to the reused initial guess, not a correctness
issue), and the saving is one explicit evaluation per reusable step within JFNK
iteration noise. **A subtlety for the Phase 8 cost panels:** those panels count
`SolveDiagnostics.rhs_evaluations`, which the DIRK driver records as `None` for explicit
stages, so the 4.50/step (TR-BDF2) and 10.72/step (ESDIRK4) figures do *not* include the
explicit first stage and do not move under reuse. The "4.50 → ~3.50" saving is a raw
`f`-call count, not a change in that panel number.

**A finding significant enough to change a registration flag, not just an
implementation detail:** implicit midpoint is the textbook symplectic Gauss-Legendre
s=1 method *only when its stage equation is solved to convergence*. The shipped
default — `FixedPointSolver(iterations=2)`, per this section's own "2 iterations
reach full order" measurement — leaves a real, uncorrected residual in the stage
equation, and that residual's effect on long-run energy conservation is not the same
question as its effect on local convergence order. Measured directly
(`tests/test_dirk.py::test_implicit_midpoint_energy_drift_bound_recovers_with_more_iterations`):
at the shipped default, max relative energy error grows secularly, ~9x over an
8x-longer run (`oscillator`, `dt=0.05`) — the same signature `test_hamiltonian.py`
uses to detect a *dissipative* scheme, not a symplectic one. Configuring the solver
with enough iterations to actually converge (~16, confirmed by a Cauchy check on the
iterate sequence) recovers the textbook bound: drift → ~1e-15, flat with `T`, matching
this section's own originally-cited numbers almost exactly. **Both things are true at
once and were conflated in this section's first draft**: "implicit midpoint is
symplectic" (a property of the exact method) and "the shipped 2-iteration default
gives you that property for free" (false — it gives you the *convergence order* for
free, not the *qualitative long-run energy behaviour*, which is arguably the scheme's
main selling point per §3.3). Registered `dissipation=True` to describe measured
default behaviour honestly; a caller who wants the textbook property back passes
`solver=FixedPointSolver(iterations=...)` explicitly, at the cost of more force
evaluations per step. This same distinction likely also affects the L-stable
tableaus' *stability* claims under Picard specifically — see the stiff-divergence
finding two paragraphs down — though stability (staying bounded) turned out to be a
smaller casualty here than symplecticity (staying bounded *and* non-dissipative) was.

**Note on the fixed-count default specifically (2026-08-24):** `FixedPointSolver`
already accepts `tol=`/`norm=` for a bounded, convergence-checked variant instead of
running a hard-coded iteration count — this was built into it from the start (§3.5
S4), not added for this note. "Ship a fixed count" is the right *default* because it
is what CUDA-graph capture and a fixed-unroll-depth training graph need (§3.7 pain
point 12), but neither of those is a universal requirement: a purely forward
simulation, or one whose gradients come from a separately-managed adjoint rather than
backprop through this solve, doesn't need either, and can pass a much larger
`iterations` cap with a real `tol` today to recover the textbook property above
without waiting on JFNK. The energy-drift recovery measured above used a bare
iteration-count sweep, not this `tol`-based path specifically; confirming the two give
the same answer is a small, concrete piece of follow-up if it matters to a caller
before `docs/research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md`'s work lands.

**A second finding, matching an existing §3.2 claim exactly rather than contradicting
it:** L-stability is a property of the *exact* method, not of a truncated Picard
solve — already stated in §3.2 for backward Euler specifically, and now confirmed for
the registered DIRK path directly (`test_picard_diverges_on_a_stiff_problem...`,
oscillator at `k=1e6`, `dt·ω=100`): the fixed 2-iteration default returns a finite but
badly wrong answer (`x≈-9999`, not the ~0 an L-stable method should give), and running
more Picard iterations at the same stiffness makes the divergence explicit and
exponential (`10^4` at 2 iterations → `10^39` at 20), matching this section's own
`3.7e+239`-at-`dt·ω=10` figure in shape. This is not a new problem Phase 2 introduced;
it is the reason the ladder in §3.4 treats Picard as strictly the non-stiff rung and
gates JFNK on "a downstream that is actually stiff" rather than on "a downstream that
uses an L-stable tableau" — the tableau's stability class only pays off once the stage
equation is actually being solved, which Picard alone does not guarantee.

**One implementation bug caught by the existing test suite, not a design gap:** the
Picard loop's `step_fn` returns a *fresh clone* as the next iterate (via
`updateStateEuler(..., copyState=True)`), so the object that actually ran
`preprocess()` — and therefore carries `copied`-behaviour fields like a summation
density — was being discarded rather than passed to `finalizeSystem` as
`lastStageSystem`. `tests/test_copied_fields.py`'s existing parametrized suite (12
tests, extended for free the moment the four DIRK schemes joined the `scheme`
fixture) caught this immediately: `density` arrived as `None`, the exact `copied`→
`ephemeral` regression that section originally guarded against for the explicit path.
Fixed by capturing the actual evaluated stage buffer in `step_fn`'s closure instead of
using the solver's returned iterate for anything but the stage's own derivative.

**A third, narrower finding, in the pre-existing test suite rather than the new
code:** `test_hamiltonian.py`'s growth-ratio heuristic ("dissipative ⟹ energy error
grows >2x over an 8x-longer run") assumed every dissipative scheme dissipates
*slowly enough to keep growing* — true for the six RK schemes it was written against,
false for an L-stable scheme fast enough to hit its own floor (all initial energy
gone) before the short horizon even ends. Backward Euler saturates near 100% relative
energy loss well inside `T_SHORT`, so its growth ratio reads ≈1.0 — "bounded", the
same signature a symplectic scheme gives, for the opposite reason. Added a
`SATURATES_EARLY` exclusion (mirroring the file's existing `UNSTABLE` one) rather than
weakening the heuristic itself, since the heuristic is still correct for every scheme
slow enough for it to apply to.

`reuse.py`'s `tableau_reuse_analysis` was **not** extended to understand "stiffly
accurate" (`a[-1] == b`) as an implicit FSAL analogue — correctly so, since DIRK reuse
was not implemented at all this round (see above), so there is nothing yet for that
extension to serve.

**Update (2026-09-10, Phase 10):** it now is, as a *separate* `dirk_reuse_analysis`
rather than an extension of the explicit-tableau formula — the substitution argument
still does not apply to implicit stages, so the implicit case is decided on the DIRK
terms (`a[0,0] == 0` + stiff accuracy, gated by the registered `stiffly_accurate`
flag) instead. See the Phase 10 note above for the mechanism and the measured results.

#### Fully implicit RK (needs a *coupled* s·N-unknown solve — a different solver shape)

| Scheme | Order | Stages | Stability | Symplectic | Marginal effort |
|---|---|---|---|---|---|
| **Gauss–Legendre s=2** | 4 | 2 | A | **yes** | 4–6 d (block solve) |
| **Gauss–Legendre s=3** | 6 | 3 | A | **yes** | +1 d after the above |
| **Radau IIA s=2 / s=3** | 3 / 5 | 2 / 3 | L, stiffly accurate | no | +2 d — the gold standard for stiff ODEs |
| **Lobatto IIIA–IIIB pair** | 2s−2 | s | A | partitioned-symplectic | 3 d, and only pays off for separable Hamiltonians |

Not recommended until something downstream demands it. The coupled solve is a genuine
step up in solver machinery, not more tableau data — and its dense/block Jacobian is
the same scale trap as §3.4's caution, worse: an `s·N × s·N` system rather than
`N × N`.

**Update (2026-09-15, Phase 6):** the first row of each column landed —
Gauss-Legendre s=2 and Radau IIA s=2, through the coupled block driver
(`fullyimplicit.py` + `fields.BlockState`, matrix-free JVP matvec, no dense
`s·N` Jacobian ever formed). The "dense/block Jacobian" trap above is avoided
by construction (the JVP/FD matvec is matrix-free at 1536 block unknowns,
§3.18's scale gate); what *did* bite at 256 DOF instead is the spread complex
spectrum of the block operator and restarted GMRES — see the GMRES finding in
§3.18. The "not recommended until a downstream demands it" scoping is obsolete:
the phase was de-gated on general-ODE-library grounds (2026-09-10) and shipped.
Gauss-Legendre s=3, Radau IIA s=3 and Lobatto remain as in the table, gated.

#### Linear multistep — Adams-Bashforth/-Moulton shipped 2026-08-24

| Scheme | Order | Evals/step | History | Implicit | Status |
|---|---|---|---|---|---|
| **Adams–Bashforth 2–5** | k | **1** | k−1 updates | no | **Done.** |
| **ABM predictor–corrector (PECE)** | k | 2 | k−1 updates | no (fixed corrections) | **Done**, for orders 2-4. |
| **Adams–Moulton 2–4** as a true corrector | k | solve | k−1 updates | yes | **Done 2026-09-09** (IMPLICIT_ROADMAP Phase 4): `multistep.AdamsMoulton` — the DIRK/JFNK solver applied to the multistep residual exactly as feared, with an optional matching-AB predictor; AM2 = trapezoidal (A-stable), AM3/4 bounded region; beats same-order PECE by ~10x on the stiff nonlinear relaxation (`tests/test_am.py`). |
| **BDF1–2** | 1, 2 | solve | k states | yes, A-stable | **Done** (`bdf.py`, JFNK-closed; BDF1 = backward Euler). |
| **BDF3–5** | 3–5 | solve | k states | yes, A(α)-stable only (86.03° / 73.35° / 51.84°) | **Done 2026-09-09** (BDF3 with the JFNK work; BDF4/5 in IMPLICIT_ROADMAP Phase 4 — coefficients from the exact order conditions, zero-stability and the A(α) cone angles measured in `tests/test_bdf.py`; history carries state snapshots, `f` only at the new grid time, so full order on non-autonomous problems). |
| **BDF6** | 6 | solve | 6 states | yes, A(α)-stable only, smaller cone than BDF5 | not done — the A(α) cone keeps shrinking with order, so the stiffness payoff keeps getting worse; gated on a downstream that actually needs order 6 |
| **Störmer–Cowell / multistep Nyström** (`x'' = f(x)`) | k | 1 | k states | no | not done |
| **Gauss–Jackson** (8th-order Störmer–Cowell) | 8 | 1 | 8 | no | not done — niche, orbital mechanics |

**Implemented as `multistep.py`** (~215 lines): `y^{n+1} = y^n + dt·Σ_j β_j k^{n-j}` reuses
`butcher._weighted_update` directly, exactly the free lunch this section predicted — `ks` doesn't care
whether its entries came from this step's stages or past steps' `StepHistory` entries. Verified
empirically for every order (`tests/test_multistep.py`): AB2-5 and ABM2-4 all reach their claimed order
on `oscillator`/`forced`/`damped` to within 0.05. Bootstraps from Dormand-Prince 5(4) (order 5, at or
above every one of these) for the first `order - 1` steps, taking `result.stages[0].update` — the
FSAL-irrelevant fact that *every* explicit RK scheme's first stage sits at `f(t^n, y^n)` — as this
step's history entry.

**A design question this section didn't anticipate, resolved during implementation: what happens if a
caller never threads `history=`?** Unlike every one-step scheme, where `history=`/`priorStep=` are pure
opt-in bookkeeping, a multistep scheme's past derivatives exist *only* if the caller carries
`IntegrationResult.history` forward. Resolved by making the fallback safe rather than merely
documented: with insufficient history, these schemes keep re-running the Dormand-Prince starter
forever — verified bit-for-bit identical to calling `DormandPrince` directly
(`test_without_history_threading_falls_back_to_the_starter_exactly`). Since DP5 is higher order than
any of these, a caller who forgets to thread history gets a *correct but more expensive* trajectory,
never a silently wrong one — the same "safe but expensive" category as Item 3's DIRK stiff-Picard
finding, not the "silently wrong" category the `history=`-auto-deriving-`priorStep` bug from Phase 0 was.

AB's one evaluation per step is the real prize here: **one force evaluation per step
at order 4**, against RK4's four, with fixed coefficients that are exactly correct
under constant `dt` and a history that never expires under stable indexing (§3.0).
Both of the things that normally make multistep painful are absent. Note that no
linear multistep method is symplectic for a general Hamiltonian (Tang, 1993) —
measured directly for all seven (`dissipation=True`, ~7-9x energy-error growth over
an 8x-longer run, matching every other non-symplectic scheme registered here);
symmetric LMMs applied to `x''=f(x)` do show good long-time energy behaviour, but they
are subject to parasitic-root instability, so don't market them as symplectic.

#### IMEX / additive RK — the right answer for SPH, and the most work

**ARK3(2)4L[2]SA** and **ARK4(3)6L[2]SA** (Kennedy & Carpenter) pair an ESDIRK
tableau for the stiff terms with an ERK tableau for the rest: viscosity, surface
tension or the pressure term implicit, advection explicit. This is what production
stiff-SPH actually wants, and it sidesteps §3.2 — the implicit part is the part with a
tractable, often *linear* operator.

**Landed 2026-09-09 (Phase 5)** as `ark.py`, an additive (two-half) Runge-Kutta
driver. No protocol change was needed: it reuses the `IMEXRHS(explicit=..., implicit=...)`
split that IMEX Euler already shipped (an ordinary RHS stays fully implicit — the pure
ESDIRK limit), and each half is a plain `updateStep` evaluation, so `updateStep` still
returns one update object per callback.

| Scheme | Order | Stages | Stability | FSAL? | Status |
|---|---|---|---|---|---|
| **ARK3(2)4L[2]SA** | 3 | 4 | L[2] | no (implicit half SA) | **Done.** Coefficients from SUNDIALS ARKODE v7.9.0 (ERK + DIRK pair). |
| **ARK4(3)6L[2]SA** | 4 | 6 | L[2] | no (implicit half SA) | **Done.** Same source; the implicit half is a *different* ESDIRK design than the standalone ESDIRK4(3)6. |

Key findings, all verified in `tests/test_ark.py` (and numerically against the SUNDIALS
C driver, `arkode_arkstep.c`):

- **The propagated update is the additive `b`-weighted combination, not the last
  stage.** An ARK method is a *pair*; the two halves are not symmetric under stiff
  accuracy. The implicit (DIRK) half is stiffly accurate (`b_imp == A_i[last]`) but the
  explicit (ERK) half is not (`b_exp != A_e[last]`), so the combined method is **not**
  FSAL and `y^{n+1} != z_last`. SUNDIALS's own `IsStifflyAccurate` check (requires both
  halves) falls through to the additive update. Using the last stage instead drops the
  method to first order (measured); the additive combination keeps the full 3/4. This is
  why both schemes register `stiffly_accurate=False` (this codebase's flag tracks FSAL)
  and reject `priorStep`.
- **`b_exp == b_imp` elementwise for both pairs.** The "two weight vectors" reduce to a
  single vector (summing to 1) applied to `f_exp + f_imp`; the additive form is written
  per half so the driver stays general.
- **The ARK-context tableaus differ from the standalone ESDIRKs.** ARK3(2)4's implicit
  `a/b/c` equal the registered ESDIRK3(2)4, but the ARK pair shares one embedded `d`
  across both halves, and that `d` is *not* the standalone ESDIRK3(2)4's. ARK4(3)6's
  implicit half is an entirely different 6-stage ESDIRK (nodes 83/250, 31/50, 17/20 vs
  the standalone's (2−√2)/4, 5/8, 26/25), so its coefficients are carried in full.
- **Copied-field ownership when both callbacks are active.** Each callback runs
  `updateStep` on its own state object (one preprocess→f→postprocess each): the implicit
  callback owns the stage buffer (the JFNK solve and post-convergence re-eval run on it,
  the `dirk.DIRK` convention) and the explicit callback runs on a throwaway clone of the
  converged buffer. The final state's copied fields come from the implicit buffer.
- **Two-parameter IMEX stability** `R(z_exp, z_imp)` is computed exactly
  (`stability.imex_stability_function`); the pure limits recover the component stability
  functions, the implicit axis is L-stable, and a slice figure is generated by
  `benchmarks/stability_gallery.py` (`docs/assets/imex_stability_slices.png`).

**Update (2026-09-15, IMPLICIT_ROADMAP Phase 12):** the multistep gap this
subsection left — IMEX Euler jumping straight to the two ARK pairs — is closed
by `imexmultistep.py`: **SBDF2** (BDF2 backbone, order 2), **SBDF3** (BDF3
backbone, order 3) and **CNAB2** (trapezoidal backbone, order 2), each with the
smooth part extrapolated explicitly to the endpoint (Lagrange weights for
SBDF, the AB2 increment for CNAB2) and the BDF derivative weight `β` scaling
the whole endpoint derivative (dropping it from the extrapolation makes the
combined method first order — measured, pinned). The Phase 5 compatibility rule
holds: an ordinary RHS runs the pure-implicit limit, bit-exact against the
registered BDF2/BDF3 for SBDF2/3 and to solver tolerance against the
registered trapezoidal scheme for CNAB2; zeroing the implicit part instead
gives the effective explicit method (bit-exact AB2 for CNAB2), whose
narrow stability lobe (negative-real-axis boundaries 4/3, 20/21, 1; the
imaginary axis only in a weak whisker) is why these run at a smaller `dt`
than the ARK pair on a problem whose explicit part has a convective
spectrum. Full record: §3.19.
