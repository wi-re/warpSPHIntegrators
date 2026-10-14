> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.8 Phased plan

Each phase is independently shippable and independently useful.

| Phase | Content | Effort | Gate |
|---|---|---|---|
| **0** | S1–S5 + S2g groundwork (§3.5) — **done 2026-08-24** | 5–6 d | none — S1 also unblocks §2.3's adaptive `dt` |
| **2** | DIRK driver + fixed-count `FixedPointSolver` + all seven DIRK tableaus — **done** (four 2026-08-24; TR-BDF2 + both ESDIRKs 2026-09-09, §3.6) | 4–5 d | Phase 0 |
| **1** | Explicit multistep: AB2–5 + ABM PECE, DP5 starter, `dt`/`uid` guards — **done 2026-08-24** | 2–3 d | Phase 0 |
| **3** | JFNK with FD matvecs + user `solve_linear` hook + particle masking | 1–1.5 wk | Phase 2, **and** a downstream that is actually stiff |
| **4** | BDF1–6 + true Adams-Moulton 2–4 (IMPLICIT_ROADMAP Phase 4) — **BDF4–5 and AM2–4 done 2026-09-09** (BDF1–3 already landed with the JFNK driver); BDF6 stays gated | 3–4 d | Phase 3 |
| **5** | IMEX / ARK, split right-hand side — **done 2026-09-09** (`ark.py`, ARK3(2)4L[2]SA + ARK4(3)6L[2]SA from SUNDIALS ARKODE v7.9.0, §3.6) | 1 wk | Phase 3 + a downstream with a split RHS |
| **6** | Fully implicit: Gauss–Legendre, Radau IIA — **done 2026-09-15** (`fullyimplicit.py` + `BlockState`, Gauss-Legendre s=2 + Radau IIA s=2, §3.18; de-gated 2026-09-10 on general-ODE-library grounds) | 1–1.5 wk | de-gated; symplectic order 4 + no-compromise L-stability |

Phases 1 and 2 are listed out of numeric order deliberately: Phase 1's old gate —
"measure the resort cadence first" — is void, because there is no resorting (§3.0),
but Phase 2 has the stronger standalone case, so it should land first.

**Recommendation: Phase 0 → 2 → 1, ~2 weeks total, then stop and reassess.**
**All three are done as of 2026-08-24 (§3.5, §3.6). Stop and reassess, as planned:
Phases 3-6 stay gated on a downstream that needs them (see below), none appeared this
session, and nothing here was building a solver contract with no user in the meantime.**

- **Phase 2 is the headline, with one caveat confirmed after implementing it.**
  Implicit midpoint is symplectic, A-stable, and holds order 2 for velocity-dependent
  forces — which nothing else registered does (§3.3) — *when its stage equation is
  solved to convergence*. At the fixed 2-Picard-iteration default this section
  recommends (no norm, no Jacobian, no branching, a fixed-depth graph that is
  differentiable in both backends by construction and CUDA-graph-capturable), that
  headline property does not actually hold: measured energy drift grows secularly,
  the signature of a dissipative scheme, not a symplectic one (§3.6 has the numbers).
  The order-2 accuracy claim is unaffected and was verified to hold exactly at the
  2-iteration default; only the *qualitative long-run energy behaviour* needs more
  iterations than the "ship 2" recommendation to recover. Four of six planned
  tableaus landed (backward Euler, implicit midpoint, trapezoidal, SDIRK2); TR-BDF2
  and both ESDIRKs landed 2026-09-09 once SUNDIALS ARKODE's published tableaus made
  the embedded-pair transcription risk checkable (§3.6).
- **Phase 1 landed cleanly, with no comparable caveat.** AB4 at one force evaluation
  per step against RK4's four, with exactly-correct fixed coefficients and a history
  that never expires — verified for every order (AB2-5, ABM2-4) against
  `oscillator`/`forced`/`damped`, and the full suite went from 1103 to 1380 passing
  tests with only one pre-existing test needing its exclusion criteria widened (not a
  bug in the new code). Pain point 2's BPTT-depth tradeoff (k-step history deepens
  backprop-through-time by k) is still unmeasured and still a question about a
  training setup this library doesn't have visibility into, not something resolved by
  landing the schemes — weigh it before choosing a multistep scheme in a training loop
  specifically, same as before.
- **Phases 3–6 stay gated on a downstream that is actually stiff.** Nothing here needs
  them yet, and building a solver contract with no user is the same mistake §2.2 warns
  about for gradients. §3.4 records the design so the decision does not have to be
  re-derived when a user appears.

**Testing.** Every phase extends the existing net rather than replacing it: the four
problems in `testing.py` already separate the failure modes (`forced` catches stage
times, `damped` catches the separable-Hamiltonian assumption, `kepler` catches
linear-only errors). Add a stiff problem for Phase 2+ (`stiff_oscillator`, `k=1e4`)
and assert that A-stable schemes stay bounded at `dt·ω = 10` while explicit ones do
not — that test is what stops §3.2 from being quietly forgotten. Phase 1 additionally
needs a test that a `dt` change or a `uid` change forces a restart rather than
consuming stale history; that is the only place a silent wrong answer can enter.
