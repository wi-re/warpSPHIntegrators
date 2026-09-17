# Conservative/Dissipative Operator Splitting

**What this document is.** An assessment of the proposal in `hamiltonian.md` — use the
existing "frozen viscosity" idea as a deliberate operator split, so the conservative half
of an SPH momentum equation becomes a genuinely separable Hamiltonian — followed by a
plan to realize it.

**Verdict up front.** The idea is sound, it is the right idea for this architecture, and
it clears a gate that three roadmap items are currently blocked on. But the write-up in
`hamiltonian.md` overstates the payoff in two specific places, and the corrections change
what should be built:

1. **Strang splitting with a backward-Euler viscous half is order 1, not order 2.** The
   half-step sub-integrator must be *symmetric* for the composition to be second order.
   `hamiltonian.md`'s own diagrams (§"IMEX Strang Split Loop", §"Structural Implementation
   Blueprint") specify `v* = v_n + (dt/2)·ν∇²v*` — backward Euler — while claiming second
   order. Use implicit midpoint, the trapezoidal rule, or the exact exponential flow.
2. **The split does not make higher-order composition unconditionally available.** It
   unlocks order 4 *for the conservative half* unconditionally, which is the real win. It
   unlocks order 4 *for the composite* only in the weak-dissipation regime, because every
   composition method of order > 2 with real coefficients has a negative coefficient, and
   a negative step run against a dissipative operator is an anti-diffusion step.

There is also a precondition neither `hamiltonian.md` nor the roadmap states, and it
decides which of two roadmap phases applies — see §2.3.

---

## 1. Where this sits in the current architecture

### 1.1 The gate this clears

Three entries in `IMPLICIT_ROADMAP.md` Phase 12 are gated on the same thing:

- *High-order symplectic composition* (Yoshida 4/6/8, Suzuki, Blanes-Moan, a general
  `compose()` helper): "**Gated on a separable Hamiltonian downstream**, and that gate is
  real: NOTES already measures the whole Verlet/Forest-Ruth family dropping to first order
  under a velocity-dependent force, which is every actual SPH momentum equation."
- *One-step Runge-Kutta-Nystrom*: "Same separability caveat as composition methods."
- Phase 6's `[?]` *Lobatto IIIA-IIIB*: "only for a concrete partitioned/separable
  Hamiltonian downstream (it needs the component partition, not this additive block solve)."

The gate is read today as *"the whole problem must be separable, and SPH's isn't."* That
reading is what makes it look permanent. The correct reading is weaker: **the composition
scheme must be applied to a separable flow.** Operator splitting manufactures exactly that
flow. It does not require the original problem to be separable — it requires the original
vector field to decompose into a separable piece and a remainder, which an SPH momentum
equation does, cleanly and along a line the solvers already cut.

So the gate is not waiting on a downstream. It is waiting on a `compose()` driver and a
declared split, both of which are library-side work.

### 1.2 What the repo already has

The split lands on more existing machinery than it needs to build:

| Needed | Already in the repo |
|---|---|
| A structured way to declare the split | `rhs.py` — `RHS` + `provides` capabilities, `resolve`, `check_contracts` (Phase 14) |
| Separable-Hamiltonian integrators for the conservative half | `verlet.py` (Leap Frog, Symplectic Euler, Velocity Verlet), `ruth.py` (PEFRL, VEFRL — already order 4) |
| Symmetric A-stable integrators for the viscous half | `dirk.py` — Implicit Midpoint, Trapezoidal |
| A nonlinear solver for a non-Newtonian viscous half | `jfnk.py` — `JFNKSolver`, matrix-free GMRES, FD and forward-AD JVP matvecs, preconditioner hook |
| An **exact** flow for a Newtonian viscous half | `exponential.py` — `krylov_phi`, the matrix-free φ-function build (landed 2026-09-12) |
| A stabilized explicit alternative to an implicit viscous half | `rkc.py` — RKC1/RKC2/RKL2 (landed 2026-09-11) |
| A problem that measures exactly the failure mode | `testing.damped_problem` — `x'' = -kx - cu`, already split as conservative + dissipative by construction |
| Stiff/diffusive benchmarks for the ν-stability gate | `diffusion_problem`, `viscous_burgers_problem` |
| A symplectic-form property check to generalize | `tests/test_hamiltonian.py::_phase_jacobian` — direct `det(DΦ_h) = 1` check |
| Gradient flow through an implicit sub-step | `jfnk._implicit_diff_reattach` (Phase 9) |

What is genuinely missing is one driver (`compose`) and one capability pair on `RHS`.
That is a small surface for what it buys.

### 1.3 The frozen-viscosity bridge, stated precisely

`hamiltonian.md` is right that the existing SPH trick — evaluate `F_visc(q_n, v_n)` once at
the top of the step and hold it constant across the RK sub-stages — is a hidden first-order
split. Worth being exact about *which* one, because it determines what changes:

- Freezing the viscous force and adding it to every stage is the **additive** (ARK-style)
  treatment with a forward-Euler-frozen implicit half: `dv/dt = F_cons(q) + c`, `c` constant.
- Lie-Trotter composition is `Φ^D_h ∘ Φ^H_h` — the viscous flow applied as a separate map.

These are *not the same map*. They agree to `O(h)` and share the same leading error term, so
"the frozen approach is already a Lie-Trotter split" is true at the order that matters and
false as an identity. The practical consequence: a solver switching from frozen-viscosity to
explicit Lie-Trotter will see its trajectory change at `O(h²)` per step, not stay bit-identical.
Switching to Strang changes it at `O(h²)` too — but *converges* at `O(h²)`, which frozen
viscosity does not.

This matters for adoption. The existing solvers are already paying for a first-order split
without getting the naming, the diagnostics, or the option to upgrade. Making the split
explicit costs them one extra viscous evaluation per step at worst, and zero in a long run
(§3.4).

---

## 2. The mathematics

### 2.1 The split

An SPH momentum equation, with `q` positions and `v` velocities:

```
dq/dt = v
dv/dt = F_cons(q) + F_visc(q, v)
```

`F_cons` is pressure plus gravity — derived from a potential, position-dependent only.
`F_visc` is physical or artificial viscosity — dissipative, velocity-dependent. Split the
vector field into two:

**𝓗 — the conservative flow.**

```
dq/dt = v
dv/dt = F_cons(q)
```

This is a separable Hamiltonian system, `H(q,p) = T(p) + V(q)`, exactly the setting
Verlet, PEFRL, VEFRL and Yoshida composition are derived for. Its flow `Φ^H_h` is
symplectic.

**𝓓 — the dissipative flow.**

```
dq/dt = 0
dv/dt = F_visc(q, v)
```

Positions are *frozen*. This is an ODE in `v` alone with `q` as a constant parameter. Three
structural consequences, all of which the implementation leans on:

1. **The flow map is `(q,v) ↦ (q, φ(v))`** — a shear in phase space, with Jacobian block
   structure `[[I, 0], [∂φ/∂q, ∂φ/∂v]]`, so `det DΦ^D_h = det(∂φ/∂v)`.
2. **The neighbour graph is constant across the substep**, because `q` is. This is precisely
   the condition `hamiltonian.md` §"Critical Rules for the JFNK & Forward AD Sub-step"
   identifies — keep the discrete topology outside the Newton loop — and here it is not a
   rule imposed on the solver, it is a property of the split. There is nothing to enforce.
3. **For Newtonian viscosity the flow is linear and exactly solvable.** `F_visc = A(q)·v`
   with `A` the SPH Laplacian discretization, symmetric negative semidefinite for the
   standard formulations. The exact flow is `v(t) = exp(tA)v₀`, and
   `det exp(tA) = exp(t·tr A) < 1` — exactly volume-contracting, which is the physical
   dissipation, measured rather than approximated.

Point 3 is the one `hamiltonian.md` misses, and it is the most useful thing here: the repo
already has matrix-free `exp(hA)v` via `krylov_phi`. For Newtonian viscosity the dissipative
half can be **exact**, not merely implicit.

### 2.2 Order of the composition

Write `𝓗` and `𝓓` for the Lie derivatives of the two vector fields.

**Lie-Trotter** (`Ψ_h = Φ^D_h ∘ Φ^H_h`). BCH gives the modified vector field

```
𝓗 + 𝓓 + (h/2)[𝓗,𝓓] + O(h²)
```

Order 1. The error is `O(h·‖[𝓗,𝓓]‖)`, proportional to the viscosity.

**Strang** (`Ψ_h = Φ^D_{h/2} ∘ Φ^H_h ∘ Φ^D_{h/2}`). The palindromic arrangement kills every
odd power:

```
𝓗 + 𝓓 + h²( −(1/24)[𝓗,[𝓗,𝓓]] + (1/12)[𝓓,[𝓓,𝓗]] ) + O(h⁴)
```

Order 2, and — because only even powers survive — **symmetric**, which is what makes it
eligible as the base method for higher-order composition (§2.5). Two properties of the
leading term are worth extracting:

- It is a **double commutator**, so it vanishes only if viscosity commutes with the
  conservative flow. It never does.
- Its magnitude scales with `‖𝓓‖`, i.e. **with ν**. In the SPH artificial-viscosity regime
  the splitting error constant is small. This is what makes a high-order conservative half
  worth more than the composite order count suggests — see §2.5.

**The symmetry requirement on the sub-integrators.** Both results above assume `Φ^H` and
`Φ^D` are *exact* flows. Replace them with numerical methods and the composition is
symmetric — hence of even order, hence order 2 — only if each sub-method is itself
symmetric and the arrangement stays palindromic. Concretely:

| Viscous sub-integrator | Symmetric? | Stability | Strang order |
|---|---|---|---|
| Backward Euler | **no** (its adjoint is forward Euler) | L-stable | **1** |
| Implicit Midpoint | yes | A-stable, not L | 2 |
| Trapezoidal (Crank-Nicolson) | yes | A-stable, not L | 2 |
| Explicit Midpoint / RK2 | yes | conditionally stable, `h ≲ h_visc` | 2 |
| RKC2 / RKL2 | no | extended real-axis, `h ≲ s²·h_visc` | 1 (order 2 in isolation) |
| Exact `exp(hA)v` via `krylov_phi` | yes (exact flows are) | unconditional, exactly damping | 2 |

This is the correction to `hamiltonian.md`, and it is not a technicality: the two blueprint
diagrams in that document both specify backward Euler and both claim second-order global
accuracy. They give first order.

**The tension this exposes.** For a *stiff* viscous half you want L-stability — infinitely
stiff modes annihilated in one step. For order and for composability you need symmetry. No
one-step method is both: an A-stable symmetric method has `|R(∞)| = 1` (midpoint and
trapezoidal both give `R(∞) = −1`, so the stiffest modes *ring* rather than damp). The
resolution is the exact exponential flow, which is symmetric *and* exactly damping
(`R(z) = e^z → 0`). This is why §4's Phase C treats the exponential path as the target
rather than a refinement — for linear viscosity it dissolves a real trade-off instead of
picking a side of it.

### 2.3 The precondition nobody states: what "conservative" requires

The `(q, v)` subsystem being separable is necessary but not sufficient. SPH carries
thermodynamic state, and *how* it is carried decides whether the conservative half is
actually separable:

- **Density by summation** (`ρᵢ = Σⱼ mⱼ W(qᵢ−qⱼ, h)`) with a per-particle entropy held
  fixed: `ρ = ρ(q)` exactly, so `P = P(ρ(q), s)` and `F_cons = F_cons(q)`. The Hamiltonian
  is `H = Σ pᵢ²/2mᵢ + Σ mᵢ u(ρᵢ(q), sᵢ) = T(p) + V(q)`. **Separable.** This is the
  variational/Hamiltonian SPH formulation (Price 2012 §3; Monaghan 2005) and it is the only
  form in which the symplectic claim was ever true.
- **Density by integrating the continuity equation** (`dρ/dt = −ρ∇·v`): `ρ` becomes an
  independent integrated field whose derivative depends on `v`, and `F_cons` depends on `ρ`.
  The conservative half is then **conservative but not separable** — the force depends on the
  history of the velocity through `ρ`.
- **Integrated internal energy** (`du/dt = −(P/ρ)∇·v`): same problem, same conclusion — see
  §2.3.1, which is the case worth spelling out, because it is the one that looks like it
  should be fine and is not.

#### 2.3.1 Compressible flow: the variable you integrate, not the compressibility

Compressibility is **not** the obstruction. A compressible ideal gas with summation density
is a textbook separable Hamiltonian. What decides separability is *which* thermodynamic
variable is carried as an integrated field.

**Summation density with integrated thermal energy `u` — not separable.** The state is
`(q, v, u)` with `P = P(ρ(q), u)`:

```
dq/dt = v
dv/dt = F_cons(q, u)
du/dt = G(q, v, u)          <- velocity-dependent
```

Both shear maps break, not just one:

- *Kick* (freeze `q`): `v` changes during the kick, so `u` changes, so `F_cons` changes. The
  force is no longer constant over the substep and the kick is not exact.
- *Drift* (freeze `v`): `q` changes, so `du/dt = (Pᵢ/ρᵢ²) Σⱼ mⱼ v_ij·∇ᵢW_ij` is nonzero and
  varying. The drift is not exact either.

This is the same causality dilemma as velocity-dependent viscosity, reaching the scheme
through the thermodynamics instead of through the force.

**Summation density with integrated entropy — separable.** Carry entropy `s` (or the
entropic function `A`) instead. For adiabatic flow `ds/dt = 0`, so `s` is a Lagrangian
constant per particle and `uᵢ = u(ρᵢ(q), sᵢ)`. The potential is then a pure function of `q`:

```
V(q) = Σ mᵢ u(ρᵢ(q), sᵢ) = Σ mᵢ Aᵢ ρᵢ(q)^(γ−1)/(γ−1)        [ideal gas]
```

`H = T(p) + V(q)`, fully separable, compressible, and PEFRL is valid on it. This is the
Springel & Hernquist (2002) entropy formulation (GADGET's entropic variable). Grad-`h` terms
survive it: the `Ωᵢ` factors from adaptive smoothing lengths are still pure functions of `q`.

**The split is the natural home for the entropy formulation, not merely compatible with it.**
Entropy is constant only under the *conservative* flow; viscous heating generates it. Under
the split that lands exactly where it belongs:

| | `ds/dt` | effect on separability |
|---|---|---|
| **𝓗** conservative | `0` — `s` frozen | intact; `V(q)` stays a pure function of `q` |
| **𝓓** dissipative | `(viscous heating)/T ≥ 0` | irrelevant; `q` is frozen here anyway |

`s` is an invariant of the conservative flow and a monotone of the dissipative one. The same
cut that separates the conservative force from the viscous force separates reversible work
from entropy production. That is not a coincidence — it is the same structural line — and it
means the entropy formulation costs the split nothing.

**If integrated `u` is unavoidable, composition is still not closed off.** Two things are
worth separating here, because running them together is what makes the gate look
insurmountable: **the composition theorem needs a symmetric order-2 base method, not a
separable problem** (Hairer-Lubich-Wanner II.4, Thm 4.1). Separability is what licenses
*Verlet* specifically — it is what makes the order-2 base explicit and cheap. It is not what
licenses *Yoshida*.

So with integrated `u`, order 4 by composition is still reachable over a symmetric base
applied to the `(v, u)` block — an iterated predictor-corrector, or implicit midpoint. The
cost moves from a free explicit shear to an implicit or iterated substep. If that base is
itself symplectic (Gauss-Legendre s=2), the composition is symplectic too, which makes this a
**bridge to Phase 6 rather than a competitor to it**: Phase 6 supplies the symmetric
symplectic base that composition then raises in order.

This is not hypothetical for this codebase — `verlet.py:108` records a removed
`integrateDensity` hook inherited from diffSPH, and `ruth.py:20` already warns that PEFRL
"probably won't work well with energy or other integration terms."

**The consequence is a clean decision rule, and it resolves an open question in the
roadmap.** `IMPLICIT_ROADMAP.md:595` notes that Phase 6's trigger "should say
*non-separable* explicitly rather than leaving the cheaper route looking unconsidered."
It should, and now there is a criterion:

| Thermodynamic closure | Conservative half is | Use |
|---|---|---|
| Summation density, fixed entropy (incompressible / isothermal) | separable Hamiltonian | **composition** — Verlet/PEFRL/Yoshida, explicit, order 4-8 cheaply |
| Summation density, integrated **entropy** `s` or `A` (compressible, adiabatic) | separable Hamiltonian | **composition** — same as above; `V(q) = Σ mᵢu(ρᵢ(q), sᵢ)` |
| Summation density, integrated **energy** `u` | conservative, **not** separable | **Phase 6** as the symmetric base, composition on top for order 4 |
| Integrated density (continuity equation) | conservative, **not** separable | **Phase 6** — Gauss-Legendre s=2, implicit, order 4 |

Both roadmap items survive; neither subsumes the other. The split does not make Phase 6
redundant, and Phase 6 does not make composition redundant — in rows 3 and 4 they **stack**,
with Phase 6 supplying the symmetric symplectic base that outer composition raises in order
(§2.3.1).

This also gives Phase A's `classify_separable` certificate something concrete to be right
about on a real SPH state: it should return "separable" for an entropy-carrying compressible
system and "not separable" for the same system switched to integrated `u` — a distinction no
amount of reading the force expression alone would reveal, since `F_cons` looks
position-only in both until you notice what `P` is a function of.

### 2.4 What "symplectic" means for the composite — and what to check instead

The composite is **not symplectic**, and should not be: physical viscosity destroys phase-space
volume, and a method that preserved it would be wrong. `hamiltonian.md` gets this right. What
the split actually buys is sharper and more checkable than "symplectic":

**No *artificial* dissipation.** The conservative half contributes zero secular energy drift —
it conserves a shadow Hamiltonian. Every joule the simulation loses is lost in the dissipative
half, where you can account for it. That is testable directly: **the total energy change over a
run should equal the integrated viscous dissipation, to `O(h²)`.** This is a far stronger
assertion than the current `dissipation` boolean, which only asks whether the energy error
grows over an 8× longer run.

**Conformal symplecticity, exactly.** For the special case `F_visc = −γv` with constant scalar
`γ` (Rayleigh damping, linear drag), the true flow is *conformally symplectic*: it contracts the
symplectic form by an exact factor, `ω(t) = e^{−γt}ω(0)` (McLachlan & Perlmutter 2001; Bhatt,
Floyd & Moore 2016). A split with an exact dissipative flow **preserves this exactly**:

```
Ψ*ω = e^{−γh} ω
```

`tests/test_hamiltonian.py` already computes `det(DΦ_h)` by finite differences on a 1-DOF
phase map and asserts it equals 1. The conformal check is the *same code* with the right-hand
side changed from `1` to `exp(−γh)`. That is the cheapest high-value validation gate in this
plan, and it is a genuine geometric property, not a proxy.

**The `dissipation` flag becomes inadequate.** It is currently a boolean meaning "energy error
grows secularly." A conformally-symplectic split is neither: its energy error grows, but for
the physically correct reason and at the physically correct rate. This wants a third
classification — see Phase F.

### 2.5 Does this unlock higher-order symplectic schemes?

Two questions that `hamiltonian.md` runs together. They have different answers.

**(a) Can PEFRL/VEFRL/Yoshida be used for the conservative half? Yes — subject to §2.3.**

This is the direct answer to the gate. Given the closure condition of §2.3 (summation
density, with entropy rather than energy as the integrated thermodynamic variable — see
§2.3.1), the conservative half *is* a separable Hamiltonian,
so the four schemes currently listed in `tests/test_convergence.py`'s
`SEPARABLE_HAMILTONIAN_ONLY` set — Leap Frog, Velocity Verlet, PEFRL, VEFRL — run at their
registered order inside the split. They measure order 1 on `damped` today for exactly one
reason: `damped` hands them a velocity-dependent force. Under the split they never see one.
PEFRL and VEFRL are already registered at order 4 and already implemented. The split makes
them useful for real SPH instead of being a curiosity that fails on every real momentum
equation.

Two properties are being bundled together here, and §2.7 shows a real scheme (CRKSPH)
that has one without the other: **velocity-independence of `F_cons`** is what makes the
kick exact and so retains the *order*, while **`F_cons = −∇V`** is separately required for
the kick to be *symplectic*. A non-variational force gives an order-preserving,
volume-preserving, non-symplectic split.

**(b) Does the composite reach order 4? Not from (a) alone.**

Raising the conservative half to order 4 removes its contribution to the `O(h²)` error, but
the **splitting error** — the double commutator in §2.2 — is `O(h²)` and remains. So
`Strang(PEFRL)` is order 2, with a smaller error constant than `Strang(Verlet)`, not order 4.

This is still worth doing, and §2.2 says why: the total error behaves like

```
err ≈ C_split · ν · h²  +  C_H · h⁴
```

At small ν the first term is suppressed and the `h⁴` term dominates until `h` grows. There is
a measurable crossover below which `Strang(PEFRL)` beats `Strang(Verlet)` outright, and it
moves with ν. Locating that crossover is a concrete deliverable (Phase E) and exactly the
kind of measured claim the rest of this repo's docs are built on.

To raise the *composite*, compose the whole Strang map — Yoshida's triple jump at the outer
level:

```
Ψ⁽⁴⁾_h = Ψ⁽²⁾_{γ₁h} ∘ Ψ⁽²⁾_{γ₀h} ∘ Ψ⁽²⁾_{γ₁h}
γ₁ = 1/(2 − 2^{1/3}) ≈  1.35120719
γ₀ = −2^{1/3}/(2 − 2^{1/3}) ≈ −1.70241438
```

The composition theorem (Hairer-Lubich-Wanner II.4, Thm 4.1) requires only that `Ψ⁽²⁾` be a
**symmetric one-step method of order 2**. It does *not* require the problem to be Hamiltonian
or separable. So this is legitimate for the full dissipative system — with one obstruction.

**The obstruction: `γ₀ < 0`.** The middle stage runs the dissipative flow *backward*. For a
decaying mode with rate `λ > 0`, a step of `−|γ₀|h` gives amplification `e^{+|γ₀|hλ}` — an
anti-diffusion step. `hamiltonian.md:78` gestures at this correctly. It is not an
implementation detail but a theorem: **no composition method of order > 2 with real
coefficients has all-positive coefficients** (Suzuki 1991; Goldman & Kaper 1996).

Two refinements that make the obstruction quantitative rather than fatal:

- *The net step is fine; the intermediates are not.* With `A` frozen, the exponentials
  telescope — `Πᵢ exp(γᵢhA) = exp(hA)` since `Σγᵢ = 1` — so the composite step is
  analytically harmless. The damage is (i) intermediate states inflated by `e^{|γ₀|hλ_max}`,
  costing that many digits to cancellation, and (ii) genuine instability once `A(q)` varies
  between substeps, which it does, because the conservative flows move the particles. The
  practical bound is therefore `h·λ_max` modest, not `h·λ_max` bounded by the explicit
  viscous CFL — a large relaxation, but not unconditional.
- *Suzuki's coefficients are strictly better here.* The 5-stage fourth-order composition
  `w₁=w₂=w₄=w₅ = 1/(4 − 4^{1/3}) ≈ 0.41449077`, `w₃ = 1 − 4w₁ ≈ −0.65796308` has a largest
  negative step of `0.658h` against Yoshida's `1.702h`. It tolerates roughly **2.6× the
  viscous stiffness** before the backward step amplifies comparably, at the cost of 5 stages
  instead of 3. For a dissipative half this is the right default, and the reason is specific
  enough to be worth recording rather than discovered per-user.

**Escape routes worth naming and not pursuing yet.** Complex coefficients with positive real
part reach order 4 with no backward step (Castella, Chartier, Descombes & Vilmart 2009; Hansen
& Ostermann 2009) — mathematically the clean answer, but it requires complex arithmetic
through the whole state, which is a large change to `fields.py` for a speculative payoff.
Processing/corrector techniques (Blanes, Casas & Murua) do not evade the negative-coefficient
theorem for order ≥ 3 in general. Neither belongs in the first pass.

### 2.6 Summary of what the split does and does not buy

| Claim | Verdict |
|---|---|
| The conservative half is a genuinely separable Hamiltonian | ✅ by construction — *if* §2.3's closure condition holds |
| Verlet / PEFRL / VEFRL run at registered order on that half | ✅ unconditional |
| Strang gives order 2 for the composite | ✅ **only with a symmetric viscous sub-integrator** |
| Lie-Trotter gives order 1 | ✅ — and is what frozen viscosity already is, at leading order |
| The viscous CFL bottleneck disappears | ✅ with an implicit or exponential viscous half |
| The composite is symplectic | ❌ — nor should it be |
| The composite is *conformally* symplectic for constant-γ damping | ✅ exactly, with an exact viscous flow |
| Energy loss equals integrated physical dissipation | ✅ to `O(h²)` — testable, and stronger than the current flag |
| Yoshida/Suzuki raise the composite to order 4 | ⚠️ yes, but with a backward dissipative step — **weak-viscosity regime only** |
| Order 4 *and* unconditional viscous stability, together | ❌ not with real coefficients. Use ARK4 (already registered, not symplectic) or accept the regime limit |

### 2.7 The three schemes of Frontiere, Raskin & Owen (2017)

`literature/1-s2.0-S0021999116306453-main (1).pdf` (JCP 332:160-209) specifies three
compressible schemes: **CRKSPH** (§3), **compSPH** (Appendix E), and **PESPH/PSPH**
(Appendix G, matching Hopkins 2015 Appendix F2, `literature/stv195.pdf`). Two of the
three are implemented in `../warpSPH` (`schemes/compSPH.py`, `schemes/crkSPH.py`), and
the implementations agree with the papers term for term. Applying §2.3's criterion to
each produces three *different* answers, and the deciding factor is different in each
case.

#### The equations

**compSPH** (Eqs. E.1-E.9), Lagrangian-derived, summation density:

```
ρᵢ   = Σⱼ mⱼ Wᵢ                                                              (E.1)
Dvᵢ/Dt = −Σⱼ mⱼ[ (Ωᵢ⁻¹Pᵢ/ρᵢ² + Πᵢ/2)∂ₐWᵢ + (Ωⱼ⁻¹Pⱼ/ρⱼ² + Πⱼ/2)∂ₐWⱼ ]          (E.2)
Duᵢ/Dt =  Σⱼ mⱼ (Ωᵢ⁻¹Pᵢ/ρᵢ² + Πᵢ/2) vᵢⱼ·∂ₐWᵢ                                  (E.3)
Ωᵢ     = 1 − (∂hᵢ/∂ρᵢ) Σⱼ mⱼ ∂Wᵢ/∂hᵢ                                          (E.5)
Πᵢ     = ρᵢ⁻¹(−C_l cᵢ μᵢ + C_q μᵢ²),  μᵢ = min(0, vᵢⱼ·ηᵢ /(ηᵢ·ηᵢ + ε²))       (E.7-8)
```

**CRKSPH** (Eqs. 64, 69-76), *not* Lagrangian-derived — see below:

```
Vᵢ⁻¹ = Σⱼ Wᵢ                                                                  (75)
ρᵢ   = Σⱼ mᵢⱼ Vⱼ W^R_ᵢⱼ / Σⱼ Vⱼ W^R_ᵢⱼ                                        (76)
Dvᵢ/Dt = −(1/2mᵢ) Σⱼ VᵢVⱼ (Pᵢ + Pⱼ + Qᵢ + Qⱼ)(∂ₐW^R_ᵢⱼ − ∂ₐW^R_ⱼᵢ)            (64)
Qᵢ   = ρᵢ(−C_l cᵢ μᵢ + C_q μᵢ²),  μᵢ = min(0, v̂ᵢⱼ·ηᵢ /(ηᵢ·ηᵢ + ε²))            (69-70)
v̂ᵢⱼ  = vᵢ − vⱼ − (φᵢⱼ/4rᵢⱼ)(∂_β vᵢ + ∂_β vⱼ)xᵢⱼ                               (71)
∂_β vᵢ = −Σⱼ Vⱼ vᵢⱼ ∂_β W^R_ᵢⱼ                                                (74)
```

Both then advance thermal energy by the **compatible discretization** (Eqs. 65-68),
*not* by their derivative forms:

```
uᵢ(t+Δt) = uᵢ(t) + Σⱼ Δuᵢⱼ Δt                                                 (65)
Δuᵢⱼ    = (fᵢⱼ/2)[vⱼ(t) + vⱼ(t+Δt) − vᵢ(t) − vᵢ(t+Δt)]·(Dvᵢⱼ/Dt)              (66)
fᵢⱼ     = 1/2, or s_min/(s_min+s_max), or s_max/(s_min+s_max)                 (67)
          branching on sign(Δuᵢⱼ) and sign(sᵢ − sⱼ);   sᵢ = Pᵢ/ρᵢ^γ
```

The paper is explicit that the derivative forms (Eq. 78 for CRKSPH, Eq. E.3 for
compSPH) "are only used to compute intermediate values of `uᵢ` during the time
advancement cycle." `warpSPH` implements exactly this: `dudt` is a genuine field on the
update, and `systems/compSPH.py::finalize` overwrites the result with
`compSPH_deltaU_multistep(dt, ..., butcherTerms, ...)` when `compatibleEnergy` is set.

**PESPH/PSPH** (Eqs. G.1-G.6), pressure by summation, **total** energy integrated:

```
P̄ᵢ  = Σⱼ (γ−1) mⱼ uⱼ Wᵢ                                                       (G.1)
Dvᵢ/Dt = −Σⱼ mⱼ(γ−1)²uᵢuⱼ [ (fᵢⱼ/P̄ᵢ)∂ₐWᵢ + (fⱼᵢ/P̄ⱼ)∂ₐWⱼ ] + q_accᵢⱼ          (G.4)
DEᵢ/Dt = mᵢvᵢ·Dvᵢ/Dt + Σⱼ mᵢmⱼ(γ−1)²uᵢuⱼ (fᵢⱼ/P̄ᵢ) vᵢⱼ·∂ₐWᵢ + vᵢⱼ·q_accᵢⱼ     (G.5)
```

#### The verdict, per scheme

| | compSPH | CRKSPH | PESPH (pressure-energy) |
|---|---|---|---|
| Density | summation, `ρ(q)` ✅ | RK summation, `ρ(q)` ✅ | summation, `ρ(q)` ✅ |
| Volume / corrections | `Ωᵢ(q)` ✅ | `Vᵢ(q)`, `Aᵢ(q)`, `Bᵢ(q)` ✅ | `fᵢⱼ` grad-h ✅ |
| Integrated thermo variable | `u` | `u` | **`E` (total)** |
| `F_cons` velocity-dependent? | **no** | **no** | **yes** ❌ |
| `F_cons` is a gradient `−∇V`? | **yes** (variational) | **no** (App. B) | yes (variational) |
| Energy advance | compatible step-map | compatible step-map | derivative `DE/Dt` ✅ |
| Fits the split | ✅ with the §2.3.1 entropy swap | ✅ for *order*, ❌ for *symplecticity* | ❌ as published |

Three distinct obstructions, and only one of them is the one the roadmap gate is about:

**1. PESPH fails at the force, not the thermodynamics.** Because it integrates *total*
energy, `uᵢ = Eᵢ/mᵢ − vᵢ²/2` is velocity-dependent **by definition**, and `P̄ᵢ` (Eq. G.1)
is a kernel sum over the neighbours' `uⱼ`. So the conservative force itself depends on
velocity, before any viscosity enters. The split cannot isolate this — there is no cut
that puts the velocity-dependence in the dissipative half, because it is in the pressure.
PESPH as published is the one scheme here that genuinely does not fit.

*The fix is the name.* "PESPH" properly means **pressure-entropy** SPH (Hopkins 2013),
where `P̄ᵢ = [Σⱼ mⱼ Aⱼ^(1/γ) Wᵢⱼ]^γ` with the entropic function `Aᵢ` as the integrated
variable and `dAᵢ/dt = 0` for adiabatic flow. Both papers here actually implement and
compare the pressure-**energy** variant. The pressure-entropy variant is separable, for
exactly the reason §2.3.1 gives, and it is the one to build on.

**2. CRKSPH keeps the order but loses the Hamiltonian.** This is a distinction §2.5
should have drawn and did not, and CRKSPH is the case that forces it:

- **Velocity-independence of `F_cons`** is what makes the kick substep exact, and that is
  all that is needed for Verlet/PEFRL to *retain their convergence order*. CRKSPH has
  this: Eq. (64)'s pressure term is `F(q, u)`, and everything velocity-dependent sits in
  `Q` (Eqs. 69-71), which is dissipative and belongs in the `𝓓` half.
- **`F_cons = −∇V`** is what makes the kick *symplectic*. A kick `(q,v) ↦ (q, v + hF(q))`
  has Jacobian `[[I,0],[h∂F/∂q, I]]`, which is always volume-preserving (`det = 1`) but
  satisfies `JᵀΩJ = Ω` only when `∂F/∂q` is symmetric — i.e. only when `F` is a gradient.

Appendix B is explicit that CRKSPH is **not** variational: the RK kernel breaks
`∂ₐWᵢⱼ = −∂ₐWⱼᵢ`, so Eq. (64) is derived by conservative differencing (MLSPH-style
integration by parts), trading consistency for pairwise antisymmetry. It conserves linear
momentum to machine precision and has no potential behind it.

So for CRKSPH the split delivers **order retention and volume preservation, but not
symplecticity** — no shadow Hamiltonian, so no bounded-energy-error guarantee, and the
conformal-symplectic check of §2.4 does not apply. compSPH, being Lagrangian-derived
(Eq. E.5's `Ω` terms *are* the grad-h corrections of the variational formulation), does
get the full symplectic story once entropy replaces `u`.

**3. The compatible energy update is the real obstruction — for both compSPH and CRKSPH.**

This is what makes the "energy balance computation" expensive, and the cost is the least
of it. Eq. (66) is:

- **not a vector field** — it is a step-level algebraic map, using `t` and `t+Δt`;
- **explicitly `Δt`-dependent** — `fᵢⱼ` branches on `sign(Δuᵢⱼ)`, which scales with `Δt`;
- **implicit in `v(t+Δt)`** — it is a midpoint velocity difference;
- **tableau-aware** — `compSPH_deltaU_multistep` consumes the Butcher weights and the
  per-stage pairwise arrays `ap_ij`/`av_ij`/`f_ij` to reconstruct one `Δu`.

Every one of those violates the library's `f(state) → update` contract, and splitting
makes it worse rather than better: sub-flows run at *different* effective step sizes
(`h/2`, `γ₀h`), a `Δt`-dependent update has no consistent meaning across them, and under
an order-4 composition `γ₀h < 0` flips the branch conditions of `fᵢⱼ` outright.

**But the framing that fixes it is already in the equation.** Rewrite Eq. (66) with
`v̄ = (v(t) + v(t+Δt))/2`:

```
duᵢ/dt |_compatible = Σⱼ fᵢⱼ (v̄ⱼ − v̄ᵢ)·aᵢⱼ
```

That *is* a vector field — evaluated at the midpoint velocity. In other words the
compatible update is an **implicit-midpoint evaluation of the energy equation, projected
onto exact discrete energy conservation** by the constraint `fᵢⱼ + fⱼᵢ = 1`. Both halves
of that are things the library already does: symmetric midpoint evaluation is
`implicitMidpoint` + `JFNKSolver`, and a once-per-step constraint projection is a
standard geometric-integration construct (HLW IV.4). Critically, a **symmetric**
projection preserves both the order and the symmetry of the base method (HLW V.4.1), so
the composition theory of §2.5 survives it:

```
Ψ_h = P_{h/2} ∘ D_{h/2} ∘ H_h ∘ D_{h/2} ∘ P_{h/2}
```

with `P` the energy projection. This also *helps* separability rather than hurting it:
because the compatible discretization takes `u` out of the flow entirely and reconstructs
it at the step boundary, `u` can legitimately be held fixed across the `𝓗` substeps,
which makes `F_cons = F(q; u_frozen)` a pure function of `q` — the §2.3.1 obstruction
dissolves without needing the entropy swap at all. The entropy swap is still what buys
*symplecticity*; freezing `u` under a projection buys *order*.

One genuine hazard remains: `fᵢⱼ` (Eq. 67) branches on `sign(Δuᵢⱼ)` and `sign(sᵢ − sⱼ)`,
so it is **piecewise constant and non-differentiable**. A JFNK solve through it will see
a discontinuous residual, and forward-AD JVPs will be wrong at the branch points. `fᵢⱼ`
must be frozen (evaluated once, held across the solve) or smoothed, and that choice
should be made deliberately rather than discovered as a convergence failure.

#### 2.7.1 Making the compatible update substep-local

Frontiere et al. present exactly two options for the pairwise accelerations Eq. (66)
needs: "either retain the pair-wise accelerations (i.e., extra memory) or recompute them
(extra computation)." They chose to retain, and `warpSPH` follows — `ap_ij`, `av_ij` and
`f_ij` are carried on the state across the whole step and consumed in `finalize`. Both
options are `O(N · neighbours)` in memory or force the pairwise data to outlive the
substep that produced it.

**There is a third option, and the split is what makes it available.** The cross-step
coupling exists only because a monolithic scheme has *one* momentum update per step, so
the work accounting must span it. In a composition, every substep is a self-contained
map with its own endpoints — so each substep can close its own energy books.

**The general form.** Take any substep of length `τ` whose velocity update is
`Δvᵢ = τ aᵢ` with `aᵢ = Σⱼ aᵢⱼ` and pairwise antisymmetry `mᵢaᵢⱼ = −mⱼaⱼᵢ`. Define

```
v̄ᵢ  := v⁻ᵢ + (τ/2) aᵢ   ( = (v⁻ᵢ + v⁺ᵢ)/2 )
Δuᵢ := τ Σⱼ fᵢⱼ (v̄ⱼ − v̄ᵢ)·aᵢⱼ
```

Then for any `fᵢⱼ` with `fᵢⱼ + fⱼᵢ = 1`:

```
Σᵢ mᵢ(½|v⁺ᵢ|² − ½|v⁻ᵢ|²) + Σᵢ mᵢ Δuᵢ = 0     exactly
```

*Proof.* The kinetic term is `Σᵢ mᵢ v̄ᵢ·(v⁺ᵢ − v⁻ᵢ) = τ Σᵢ mᵢ v̄ᵢ·Σⱼ aᵢⱼ`. Collect the pair
`(i,j)`: it contributes `−τ[mᵢv̄ᵢ·aᵢⱼ + mⱼv̄ⱼ·aⱼᵢ] = τ mᵢ(v̄ⱼ − v̄ᵢ)·aᵢⱼ` to the required
thermal total, using `mⱼaⱼᵢ = −mᵢaᵢⱼ`. The same pair contributes
`τ mᵢfᵢⱼ(v̄ⱼ−v̄ᵢ)·aᵢⱼ + τ mⱼfⱼᵢ(v̄ᵢ−v̄ⱼ)·aⱼᵢ = τ mᵢ(fᵢⱼ+fⱼᵢ)(v̄ⱼ−v̄ᵢ)·aᵢⱼ` to `Σᵢ mᵢΔuᵢ`.
These are equal iff `fᵢⱼ + fⱼᵢ = 1`. ∎

Note `τ` factors out of both sides, so **this holds for negative `τ` too** — the
identity survives the backward substeps of an order-4 composition (§2.5) unchanged.

**What this buys:**

- **No pairwise storage, ever.** `aᵢⱼ` and `fᵢⱼ` are produced and consumed inside the
  same substep. Nothing outlives it.
- **Tableau-independence.** No Butcher weights enter, so
  `compSPH_deltaU_multistep(dt, …, butcherTerms, …)` disappears and the energy update
  becomes an ordinary per-substep quantity update.
- **Exact conservation per substep**, hence over the whole composed step by telescoping —
  and it resolves the objection raised above that a split momentum update has two
  different discrete works. Each half now accounts for its own, which is *more*
  compatible in Owen's sense, not less.
- **A free energy-budget diagnostic** (Phase F): the `𝓗` substeps' `Δu` is reversible
  `PdV` work, the `𝓓` substeps' is viscous heating. They are now separately visible.
- **It fits the existing call structure exactly.** `verlet.py` and `ruth.py` already
  issue `applyQuantityUpdate(state, k, explicit_step(c·dt))` alongside each
  `applyVelocityUpdate` with the same coefficient — the per-substep energy update slots
  straight into that call, and `updateStep` already threads the substep's own effective
  `dt` into the right-hand side (`leapFrog` passes `dt/2`, `PEFRL` passes `ξ·dt`,
  `χ·dt`, …), which is the `τ` the formula needs.

**What it costs: one extra neighbour loop per substep, and no more.** The subtlety is
that `v̄ⱼ` needs `aⱼ`, particle `j`'s *total* acceleration — a per-particle quantity, not
a per-pair one. So the substep is two passes: pass 1 is the ordinary force loop
(producing `aᵢ`, which the library already returns as the stage update `k`), and pass 2
recomputes `aᵢⱼ` from the unchanged per-particle state to accumulate the work. Memory
goes from `O(N · neighbours)` to `O(N)`; cost goes to roughly `2×` the force loop per
kick. Splitting the sum,

```
Σⱼ fᵢⱼ(v̄ⱼ − v̄ᵢ)·aᵢⱼ  =  Σⱼ fᵢⱼ(v⁻ⱼ − v⁻ᵢ)·aᵢⱼ  +  (τ/2) Σⱼ fᵢⱼ(aⱼ − aᵢ)·aᵢⱼ
```

shows the one-pass part and the `O(τ)` correction that forces the second pass. Dropping
the second term makes it single-pass but **breaks exact conservation** — the identity
above needs the true `v̄`. That is not a trade worth taking: exact conservation is the
entire point of the compatible discretization.

This is precisely the trade Frontiere et al. flag as possibly preferable on modern
hardware — "the more computationally demanding second choice of recomputing the pair-wise
accelerations may see benefits on architectures such as GPU accelerated machines with
limited memory and significant FLOPs to burn." For a Warp backend that is the expected
regime, and §3.0's carried-and-revalidated neighbour list makes the second pass cheap:
the traversal is reused, only the arithmetic repeats.

**One consistency requirement.** For the substep map to be symmetric — which §2.5 needs
for any composition above order 2 — `fᵢⱼ` must be **frozen** across the substep rather
than re-derived from `sign(Δuᵢⱼ)`. Reversing the substep (`τ → −τ`) flips `sign(Δuᵢⱼ)`
and therefore switches `fᵢⱼ`'s branch, which would break time-reversibility. Freezing
`fᵢⱼ` makes the substep exactly reversible, and it is the same fix the JFNK
differentiability hazard above already demands — one decision settles both.

**This is a different method from Eq. (66), not a re-derivation of it.** Both are exactly
conserving and both are consistent with `duᵢ/dt = Σⱼ fᵢⱼ(vⱼ−vᵢ)·aᵢⱼ`, so they agree to
the order of the scheme, but they distribute energy differently at `O(τ)` within a step.
That difference must be measured, not assumed away — it is the validation gate for this
piece.

#### Which path each scheme takes

| Scheme | Path |
|---|---|
| **compSPH + entropy** | **Path A — the full target.** Separable, variational, symplectic `𝓗`; Monaghan `Π` is a clean velocity-only `𝓓` half. Everything in §4 applies, including the conformal check (§2.4) and order-4 composition (§2.5) in the weak-viscosity regime. This is the scheme to prototype on. |
| **compSPH + compatible energy (as published)** | **Path B — substep-local compatible update (§2.7.1).** Per-substep energy accounting with frozen `fᵢⱼ`: no pairwise storage, tableau-independent, exactly conserving, symmetric. Order 2 via Strang, order 4 via outer composition. Costs one extra neighbour loop per kick. |
| **CRKSPH** | **Path B, minus the geometry.** Same structure as compSPH's Path B — order retention, volume preservation, viscous-CFL relief, and the same substep-local energy update — but no shadow Hamiltonian and no conformal-symplectic property, because Eq. (64) is not a gradient (App. B). Worth doing for the `Δt` relief and the order; do not advertise it as symplectic. Its `𝓓` half is *nonlinear* in `v` (Eq. 70's `C_q μᵢ²` plus the Eq. 74 gradient reconstruction), so it takes the **JFNK** dissipative path, not the exponential one — `krylov_phi` does not apply. |
| **PESPH (pressure-energy, as published)** | **Does not fit.** Velocity-dependent conservative force. No split isolates it. |
| **PESPH (pressure-entropy, Hopkins 2013)** | **Path A.** Separable and variational; the `Aᵢ` variable is exactly §2.3.1's entropy argument in its native form. |

The honest summary for the user's instinct about CRKSPH: the energy balance *is* the hard
part, but not for the cost reason. The pairwise-work bookkeeping is `O(neighbours)` and
`warpSPH` already retains `ap_ij`/`av_ij` per pair. What makes it hard is that it is a
`Δt`-dependent, implicit, non-differentiable step-level map in a library whose entire
contract is `f(state) → update`. What makes it tractable is §2.7.1: recast per substep,
it is an ordinary quantity update that stores nothing past the substep, conserves energy
exactly, and composes — trading `O(N · neighbours)` memory for one extra neighbour loop.

---

## 3. Feasibility against this architecture

### 3.1 The interface: one new capability pair

The split is *already expressible* today as `IMEXRHS(explicit=conservative, implicit=viscous)`
— it is additive in exactly the sense `rhs.py` means, with both halves returning a full
update object. So nothing is blocked. But the existing names are **solver-role** names
(explicit/implicit), and a composition driver needs two facts those names do not carry:

- The conservative half is a **separable Hamiltonian vector field** — the licence to hand it
  to PEFRL.
- The dissipative half has **`dq/dt ≡ 0`** — the structural fact behind the frozen neighbour
  graph, the shear-map Jacobian, and the velocity-only solve.

This is the same argument Phase 14 made for adding `linear`/`nonlinear` beside
`explicit`/`implicit`: a different *structure*, not a different *role*, deserves its own
declared capability. So:

```python
ConservativeDissipativeRHS(conservative=..., dissipative=...)
# -> RHS with provides = {'conservative', 'dissipative'}
```

with three contracts for `check_contracts`, all in the style already there:

1. **Additivity** — `conservative + dissipative == f`. Identical in form to the existing
   `explicit + implicit == f` check.
2. **The dissipative half does not move particles** — its position derivative is identically
   zero. Cheap, exact, and it is the structural premise everything else rests on.
3. **The separability certificate** — `∂(dv/dt)_cons/∂v == 0`, probed by forward-mode AD
   (already used in `jfnk.jvp_matvec`) or finite differences on a velocity-perturbed state.

Contract 3 is the one worth being loud about. **The reason Verlet silently dropped to order 1
on every real SPH momentum equation is that nothing ever checked.** A certificate turns
"gated on a separable Hamiltonian downstream" from a hand-wave into a machine-verified
precondition, and it should be built as a *general classifier* in the exact style of Phase
13's `tvd_analysis.classify_tvd` — measure the property, report a verdict with the measured
`‖∂F/∂v‖`, and let `resolve(..., need_conservative=True)` raise a capability error
**before the solve**, the way `need_linear=True` already does. A driver should refuse to run
PEFRL on a half that fails the certificate rather than silently returning first-order results.

A pleasant orthogonality falls out: capabilities are per-part, so the dissipative half can
itself be a `SemilinearRHS` declaring `linear` — which is precisely what the exponential
viscous flow needs to get `A` for `krylov_phi`. Nested structured RHS needs no new mechanism.

### 3.2 The driver, and the one real risk

A `compose` driver is short — a palindromic list of `(sub_scheme, coefficient)` pairs, each
applied to its half of the split, threading the state through. The sub-schemes are the
registered ones, unmodified.

**The risk is lifecycle re-entrancy.** Every registered scheme opens with
`initializeSystem(state, dt, ...)` and closes with `finalizeSystem(...)`. A composition
driver invoking three sub-scheme *functions* per outer step calls `initialize` three times
and `finalize` three times. With `BaseIntegrationSystem`'s no-op defaults that is harmless;
with a real SPH system it is not, and the two hooks want opposite answers:

- `initialize` per substep is arguably **correct** — positions changed, the neighbour list
  wants revalidating. §3.0's "cheap velocity-Verlet-style validity check with rebuild only
  when it fails" is designed for exactly this and makes the extra calls cheap.
- `finalize` per substep is **wrong** — it is the once-per-step assembly hook, and running it
  on an intermediate state would publish a half-finished step.

This needs a deliberate decision, not a default. The minimal version: a `substep=True` flag
that `finalizeSystem` pops and honours, so the composition driver owns the single outer
`finalize`. It is small but it touches the protocol, so it belongs in the plan rather than
being discovered during implementation. This is the one place where the work is architectural
rather than additive.

### 3.3 Cost: Strang is order 2 at Lie-Trotter price

Consecutive Strang steps share adjacent half-steps:

```
[D_{h/2} H_h D_{h/2}] · [D_{h/2} H_h D_{h/2}] · … = D_{h/2} · H_h D_h H_h D_h … H_h · D_{h/2}
```

Over an `N`-step run the interior half-steps merge, so Strang costs **one** dissipative flow
per step — the same as Lie-Trotter, the same as frozen viscosity — plus one extra at each end.
This answers "is it affordable" directly: second order in the viscous term is essentially free
relative to what the solvers already pay.

Two caveats, both precise:

- Merging is **exact** when `Φ^D` is the exact flow (`exp(aA)exp(bA) = exp((a+b)A)`). When
  `Φ^D` is a one-step integrator lacking the group property — implicit midpoint, trapezoidal —
  two half-steps are *not* the same map as one full step. The merged method remains symmetric
  and order 2, but it is a **different method**, related to the unmerged one exactly as
  leapfrog's staggered form relates to its synchronized form. This repo already understands
  that distinction (`verlet.py:15`, "also known as synchronized form"), which is a useful
  precedent to name in the docs.
- Merging requires constant `dt`, which §3.0 guarantees for the target simulation but which
  the adaptive path (Phase 11) would break. Under adaptivity, pay the full two flows.

### 3.4 What is inherited for free

- **Gradients.** The composite is a chain of sub-steps. Autograd flows through if each
  sub-step does, and the implicit dissipative half already reattaches via
  `_implicit_diff_reattach`. Phase 9 coverage is inherited, not re-derived.
- **Error estimation.** `IntegrationResult.error` is already there, and the split offers a
  cheap Milne-style estimator: `Strang − LieTrotter` is an `O(h²)` estimate of the splitting
  error, and both maps are already being computed if the merged form is in use.
- **Solver diagnostics, preconditioning.** The dissipative half runs through `JFNKSolver`
  unchanged, inheriting Phase 1 diagnostics and the Phase 2 `preconditioner(v, state, context)`
  hook.
- **The test problem.** `damped_problem` is `x'' = −kx − cu`: already exactly a conservative
  `−kx` plus a dissipative `−cu`, with an analytic solution. The problem that currently
  *proves* the order collapse becomes the problem that proves the split fixes it, with no
  new problem written. That is the cleanest possible demonstration and it is already in the
  repo.

---

## 4. Realization plan

Phases are ordered so each one's validation gate is measurable before the next begins.
Phases A-B are the minimum that clears the roadmap gate; C-D make it usable for stiff
viscosity; E-F are the high-order and geometric payoffs.

### Phase A — the split as a declared capability

- Extend `rhs.CAPABILITIES` with `conservative` / `dissipative`; add
  `ConservativeDissipativeRHS(conservative=, dissipative=)` as a thin constructor returning
  an `RHS`, exactly as `IMEXRHS` and `SemilinearRHS` are.
- Extend `resolve` with `need_conservative` / `need_dissipative`, raising the pre-solve
  capability error in the existing style.
- Extend `check_contracts` with the three contracts of §3.1.
- Build `classify_separable(rhs, state)` — the velocity-independence certificate — as a
  general classifier reporting a verdict plus the measured `‖∂F_cons/∂v‖`, in the style of
  `tvd_analysis.classify_tvd`. Run it over every registered test problem and record the
  table, the way the TVD verdict table is recorded in NOTES §3.11.
- Give `damped_problem` a declared split (`conservative = −kx`, `dissipative = −cu`) and add
  the same to `viscous_burgers_problem` (inviscid flux vs. `ν∂²u/∂x²`).

**Gate.** `check_contracts` passes on both split problems. `classify_separable` returns
"separable" for the conservative half of `damped` and "not separable" for the *unsplit*
`damped` RHS — i.e. the certificate detects the exact condition that has been silently
breaking Verlet.

**Downstream prototype target (from §2.7): an entropy-carrying scheme**, not CRKSPH —
either compSPH with entropy in place of `u`, or **pressure-entropy PESPH**, which is the
cleaner target of the two and is scoped in `../warpSPH/PESPH_PLAN.md`. Both are
simultaneously velocity-independent in `F_cons`, Lagrangian-derived (so genuinely
symplectic), and free of the compatible-energy step-map. CRKSPH and the
compatible-energy form of compSPH follow on Path B; pressure-*energy* PESPH does not fit
at all.

**Sequencing.** `PESPH_PLAN.md` lands *first* and does not depend on this plan —
pressure-entropy PESPH is a plain `f(state) -> update` that runs with every currently
registered scheme. Building it first isolates debugging (a discrepancy under RK4 has one
candidate cause, not two) and produces the measured baseline — order, energy drift, cost
— that this plan's benefit is then judged against. The one item worth pulling forward
into Phase A is `classify_separable`, because pressure-entropy vs pressure-energy PESPH
is its sharpest test case: the two differ *only* in the integrated thermodynamic variable
yet land on opposite sides of the separability verdict.

### Phase B — the `compose` driver, Lie-Trotter and Strang

- `compose(parts, coefficients)`: a palindromic composition driver over sub-schemes.
- `LieTrotter(H_scheme, D_scheme)` and `Strang(H_scheme, D_scheme)` as the two registered
  entry points, with both sub-schemes as parameters rather than baked in.
- Resolve the lifecycle question of §3.2 — recommend the `substep` flag on `finalizeSystem`,
  with `initialize`/`preprocess`/`postprocess` still running per substep.
- Register the useful combinations rather than the cross product: `Strang(VelocityVerlet,
  ImplicitMidpoint)` as the default, `Strang(PEFRL, ImplicitMidpoint)` as the high-order-H
  variant, `LieTrotter(VelocityVerlet, ExplicitMidpoint)` as the frozen-viscosity analogue.

**Gate.**
- Strang measures **order 2** on split `damped` with any symmetric viscous sub-integrator,
  and **order 1** with backward Euler — pinning §2.2's correction as a test rather than a
  claim in a document.
- Lie-Trotter measures order 1.
- The four `SEPARABLE_HAMILTONIAN_ONLY` schemes each measure their *registered* order on the
  conservative half in isolation — the gate's core assertion.
- `tests/test_convergence.py::test_splitting_schemes_need_a_separable_force` keeps passing
  unchanged on the unsplit problem. The split is an addition, not a fix to existing behaviour.

### Phase C — implicit and exact dissipative flows

- **Implicit path**: implicit-midpoint dissipative half through `JFNKSolver`, solving for
  velocity with positions frozen. Non-Newtonian viscosity works natively — the residual is
  just the SPH viscous loop, as `hamiltonian.md` correctly describes.
- **Exact path**: when the dissipative half declares `linear`, use `exp(hA)v` through the
  existing `krylov_phi`. Symmetric *and* exactly damping — §2.2's trade-off dissolved, and
  the natural default for Newtonian viscosity.

**Gate.** A viscous-CFL sweep on `diffusion_problem` / `viscous_burgers_problem`: the
explicit dissipative half blows up past `h ≈ αh²/ν` while the implicit and exponential halves
stay bounded at 10-100× that step, with order 2 retained throughout. This is the claim that
justifies the whole implicit apparatus, and it is the one `hamiltonian.md` is most confident
about — so it should be measured, not assumed.

### Phase C2 — substep-local compatible energy (§2.7.1)

Only needed for a downstream carrying the compatible discretization (compSPH/CRKSPH as
published); skip it for an entropy-carrying scheme, which needs no energy projection at
all. Independent of Phase C's dissipative solver, so the two can run in parallel.

- A `conservative`/`dissipative` half may declare a **pairwise work** accessor returning
  `Δuᵢ = τ Σⱼ fᵢⱼ(v̄ⱼ − v̄ᵢ)·aᵢⱼ` for its own substep, with `fᵢⱼ` frozen and `v̄` built
  from the substep's own `τ` and total acceleration.
- Route it through the existing `applyQuantityUpdate` call that `verlet.py` and `ruth.py`
  already issue per substep with the matching coefficient.
- A conservation check in the `check_contracts` style: `fᵢⱼ + fⱼᵢ = 1` per pair, probed
  on a sample state.

**Gate.**
- `Σᵢ mᵢ(ΔKEᵢ + Δuᵢ) = 0` to roundoff **per substep**, for positive *and* negative `τ`
  (the order-4 composition case), on a small SPH-shaped test state.
- No pairwise array survives a substep boundary — memory measured, not assumed.
- The substep-local update and the monolithic Eq. (66) reach the **same convergence
  order** on the split `damped`/Burgers problems, and their `O(τ)` difference in energy
  *distribution* is measured and recorded rather than assumed negligible (§2.7.1's
  closing caveat).
- Reversing a substep (`τ → −τ`) with frozen `fᵢⱼ` returns the initial state to solver
  tolerance — the symmetry that Phase E's composition depends on.

### Phase D — half-step merging

- Merged-Strang for constant `dt`, expressed in the existing `priorStep`/FSAL vocabulary.
- Document the staggered-vs-synchronized distinction of §3.3 and the exactness condition.
- Milne-style `Strang − LieTrotter` error estimate into `IntegrationResult.error`.

**Gate.** Merged Strang reaches the same order and the same trajectory (to the exactness
condition of §3.3) at **one** dissipative flow per step — measured in viscous evaluations
per step, the same work-unit metric Phase 7's cost gate uses.

### Phase E — outer order-4 composition, regime-gated

- Yoshida triple-jump and Suzuki 5-stage over the Strang map.
- Measure the stability boundary in `h·λ_visc` for both coefficient sets, and document
  Suzuki's `0.658h` vs Yoshida's `1.702h` largest backward step as the reason to prefer it.
- Measure the `C_split·ν·h² + C_H·h⁴` crossover of §2.5: where does `Strang(PEFRL)` beat
  `Strang(Verlet)`, and how does that boundary move with ν?

**Gate.** Order 4 measured on split `damped` at small `c`; a **documented blow-up threshold**
as `c` rises, with Suzuki's threshold measurably higher than Yoshida's. A negative result here
is still a result — the regime boundary is the deliverable, and publishing it is what stops
the next person from assuming order 4 is unconditional.

### Phase F — the geometric payoff

- **Conformal symplecticity**: generalize `tests/test_hamiltonian.py::_phase_jacobian`'s
  `det(DΦ_h) = 1` to `det(DΨ_h) = e^{−γh}` for constant-γ damping. One changed right-hand
  side, a genuine geometric property.
- **Energy budget**: assert total energy change equals integrated viscous dissipation to
  `O(h²)` — strictly stronger than the current `dissipation` growth-ratio proxy.
- **Reclassify `dissipation`**: the boolean cannot express "dissipates at exactly the
  physical rate." Add `conformal` as a third classification, or a `conformal_factor`
  diagnostic, and update
  `test_bounded_energy_schemes_are_exactly_the_expected_set` to force the deliberate choice
  the way it already does for the boolean.

**Gate.** The conformal determinant holds to solver tolerance for `Strang(Verlet, exact)` on
constant-γ damping, and fails for an unsplit RK4 on the same problem — otherwise the test is
not measuring the property it claims.

### Roadmap consequences

- **Phase 12 "High-order symplectic composition"** — de-gate. The gate was "a separable
  Hamiltonian downstream"; the split *is* that downstream, with `classify_separable` as the
  guard that keeps the claim honest.
- **Phase 12 "One-step Runge-Kutta-Nystrom"** — de-gate on the same grounds, and promote:
  RKN is derived for `q'' = F(q)`, which is exactly the conservative half. It is the natural
  H-scheme and a cheap follow-on to Phase B.
- **Phase 6 trigger** — amend to say *non-separable* explicitly, per `IMPLICIT_ROADMAP.md:595`,
  and record §2.3's decision rule: the thermodynamic closure picks between Phase 6 and
  composition. Both survive.
- **Phase 6 `[?]` Lobatto IIIA-IIIB** — the "concrete partitioned/separable Hamiltonian
  downstream" now exists. Still low priority; note it rather than schedule it.
- **NOTES "A finding, not a defect"** — the entry recording that Leap Frog / Velocity Verlet
  / PEFRL / VEFRL drop to first order under a velocity-dependent force gains a resolution:
  they drop to first order *on an unsplit* velocity-dependent force, and the split is the
  supported way to get their registered order back.

### Explicitly out of scope

- Complex-coefficient compositions (§2.5). Clean mathematics, large change to `fields.py`,
  no consumer yet.
- More-than-two-way splits (conservative / viscous / external-forcing). The two-way split is
  the one the physics and the existing solvers already cut along; `IMPLICIT_ROADMAP.md:754`
  already rules out multi-way additive splits on the same "no consumer" grounds.
- Claiming symplecticity for the composite anywhere in the docs or the scheme table.
  Conformal symplecticity is the true and checkable statement (§2.4).

---

## 5. References

- Hairer, Lubich & Wanner, *Geometric Numerical Integration*, 2nd ed. — II.4 (composition
  methods and the order theorem), II.5 (splitting, Strang, the BCH expansion), III.4
  (backward error analysis for splitting).
- Suzuki (1991); Goldman & Kaper (1996) — no composition of order > 2 with real coefficients
  has all-positive coefficients.
- Yoshida (1990) — the triple-jump coefficients. Suzuki (1990) — the 5-stage fourth-order set.
- Omelyan, Mryglod & Folk (2002) — PEFRL/VEFRL, already cited in `ruth.py`.
- McLachlan & Perlmutter (2001); Bhatt, Floyd & Moore (2016) — conformally symplectic
  integrators.
- Castella, Chartier, Descombes & Vilmart (2009); Hansen & Ostermann (2009) — complex
  coefficients with positive real part.
- Price (2012), *Smoothed particle hydrodynamics and magnetohydrodynamics*, §3; Monaghan
  (2005) — the variational/Hamiltonian SPH formulation and its summation-density requirement.
- Springel & Hernquist (2002), *Cosmological SPH simulations: the entropy equation* — the
  entropy/entropic-function formulation that keeps compressible SPH separable (§2.3.1), and
  the grad-`h` (`Ωᵢ`) terms under adaptive smoothing lengths.
- Weiler, Koschier, Brand & Bender (2018); Bender & Koschier — implicit viscosity in SPH, the
  solvers `hamiltonian.md` cites as the production precedent for the split.
- **Frontiere, Raskin & Owen (2017)**, *CRKSPH — A Conservative Reproducing Kernel Smoothed
  Particle Hydrodynamics Scheme*, JCP 332:160-209 —
  `literature/1-s2.0-S0021999116306453-main (1).pdf`. §3 CRKSPH, §3.3 the compatible energy
  discretization, Appendix B the consistency/conservation trade-off (why CRKSPH is not
  variational), Appendix E compSPH, Appendix G PESPH. Analysed in §2.7.
- **Hopkins (2015)**, *A new class of accurate, mesh-free hydrodynamic simulation methods*,
  MNRAS 450:53-110 — `literature/stv195.pdf`, Appendix F2 for the PSPH equations that match
  Frontiere's Appendix G. Hopkins (2013), MNRAS 428:2840, is the pressure-**entropy**
  variant that §2.7 identifies as the separable one.
- Owen (2014) — the original compatible-energy SPH discretization ([48] in Frontiere et al.),
  the source of the `fᵢⱼ` work-partitioning that §2.7 reframes as a symmetric projection.
- Reference implementations: `../warpSPH/src/warpSPH/schemes/{compSPH,crkSPH}.py`,
  `modules/compSPH/{accel,dudt,balance,multistep}.py`, `modules/crk/{accel,dudt,limiter}.py`.
