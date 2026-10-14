> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.15 Phase 7: ETD2RK (exponential integrator) — done 2026-09-12

The roadmap's Phase 7 **exponential-integrator** sub-phase, and the first
exponential method in the registry. An exponential integrator is the natural home
for a semilinear PDE `y' = L·y + N(y)` with a *stiff linear* part `L` and a *mild*
nonlinear remainder `N`: instead of solving a nonlinear system (DIRK/BDF) or a
frozen-operator linear system (Rosenbrock), it carries `L` **exactly** through the
matrix functions `exp(hL)` and the `phi_k` functions, and only *quadratures* the
mild nonlinear part. This is the strongest fit for the Phase 14 `SemilinearRHS`
split of the three Phase 7 families, and it is the method that consumes the `linear`
accessor most directly.

**Method.** The base unsplit **ETD2RK** (2-stage, **order 2**, **L-stable** for the
linear part), in the form of **Sarumi, arXiv:2601.06849** (unsplit ETD2RK, eq. (2.5));
the `phi` functions per **Caliari & Ostermann, *Appl. Numer. Math.* 59 (2009) 568–581,
eq. (2.4)** (`phi_k(z) = (e^z − P_{k−1}(z))/z`, `P_{k−1}` the Taylor polynomial):

```
y_{n+1} = exp(hL)·y_n + h·phi_1(hL)·N(t_n, y_n)
         + h·phi_2(hL)·( N(t_n + h, w) − N(t_n, y_n) )
w       = exp(hL)·y_n + h·phi_1(hL)·N(t_n, y_n)        (the first stage)
```

Two stages (`w`, `y_{n+1}`), **two `N` evaluations per step** (`N_n`, `N_w`), and
**three matrix-function actions per step** (`exp(hL)`, `phi_1(hL)`, `phi_2(hL)`), each
applied to a *different* vector. `L` is integrated **exactly** (the linear part has no
truncation error — see the verification below); only `N` is quadratured, which is what
makes the method order 2. (The newly-added literature PDF is the *dimension-split,
Sylvester-based* 2D extension of ETD2RK — it factors the 2D diffusion operator as a
Sylvester matrix product to cut the Krylov cost. Our benchmark `L = nu·u_xx` is already
a single 1D operator, so the Sylvester split has nothing to buy here; ETD2RK on a 1D
`L` is the in-scope, in-repo piece.)

**Matrix-free `phi_k(hL)v`.** `exponential.py` builds `krylov_phi(matvec, v, k, m=30,
tol=1e-10)`: one **Arnoldi** build of `m` steps on the operator (the same Givens-free
Hessenberg core as `jfnk.gmres`, with the residual early-stop), then `phi_k(H)·e_1` of
the small dense `(m, m)` Hessenberg matrix by a **short Taylor-vector series**
(`term_{j+1} = H·term_j / (j + k + 1)`, `k = 0` gives `exp`). No dense `L`, no matrix
exponential, no eigendecomposition. One build per distinct right-hand-side vector
(three per step: `exp(hL)·y_n`, `phi_1(hL)·(h·N_n)`, `phi_2(hL)·(h·M)`), because the
build is the expensive part and the `phi` of the Hessenberg is cheap. The total
`L`-matvec (Krylov iteration) count is reported as the step's Krylov-iteration count in
the `SolveDiagnostics` (the `gmres_iterations` field). The `phi` series is a plain
Taylor sum (no scaling-and-squaring), so it is accurate for modest `||h·L||`; the
benchmark's `||h·L|| ~= 3.3` is comfortably in range, and an over-stiff step would need
scaling-and-squaring (a future extension, noted in the module docstring).

**Gradient (in-scope, the point of a differentiable exponential integrator).** Because
`L` is **constant** (independent of `y`), the three matrix-function actions are *fixed*
linear maps; each is re-attached to the autograd tape with its **transpose** as the
adjoint — the transpose `phi_k(hL)^T` actions computed by the same Krylov machinery on
`L^T`. There is no unrolled Krylov and, unlike the Rosenbrock frozen-`W` (which drops
`dW/dy`, a Hessian, giving an O(dt) gradient), **no structural approximation** — the
gradient is **exact up to the Krylov tolerance**. Scope note: the transpose reuses the
forward `linear` accessor because the benchmark `L = nu·u_xx` (periodic central
Laplacian) is **self-adjoint**; a non-self-adjoint `L` would need an `L^T` accessor
(a future extension). This is the cleanest gradient story in the Phase 7 set: it is
exact (up to the Krylov/FD tolerance) on *both* the linear and the semilinear problem,
whereas ROS3P's is O(dt) (a few percent) on the semilinear one.

**Registration.** `etd2rk = 56` in the enum; `IntegrationScheme(integrateETD2RK,
'ETD2RK', ..., order 2, dissipation True, fsal True, implicit=False, steps=1,
stiffly_accurate=False, stability='L')`. It **requires the `linear` accessor**:
`resolve(f, scheme_name='ETD2RK', need_linear=True)` raises a `TypeError` naming the
`linear` part before the solve if the RHS is a plain callable. Not stiffly accurate, so
no first-stage reuse. No embedded pair, so `error is None`.

**Verification** (all pinned in `tests/test_exponential.py`, 10 tests):

- **Order 2 on viscous Burgers** (vs the fine-`dt` RK4 reference, Gaussian IC):
  relative L2 error `[2.34e-01, 2.15e-02, 5.09e-03, 1.24e-03, 3.06e-04]` at
  `dt = [0.08, 0.04, 0.02, 0.01, 0.005]`, measured orders `[3.44, 2.08, 2.04, 2.02]`
  (the coarsest pair is pre-asymptotic; the asymptotic order is **2**, as advertised).
- **Exact on a purely linear problem** (`N = 0`, `x' = L·x`): the error is at the
  Krylov tolerance (~1e-9), **not** `O(h^2)` — the linear part is carried exactly, the
  point of an exponential integrator.
- **L-stable**: a stiff linear mode (`x' = -rate·x`, `rate = 10`, `h = 1`, `rate·h = 10`)
  is damped by `exp(-10) ~= 4.5e-5` (killed, well under 1e-3), in contrast to ROS3P's
  A-stability ceiling (`R(∞) = 1 − √3 ~= -0.732`) which only dampens.
- **Two stages / two `N` evals per step**, the Krylov count is reported, `error is None`.
- **Gradient parity** (autograd vs central finite difference): the linear problem agrees
  to ~1e-9 (the Krylov tolerance) and the semilinear problem to ~1e-5 (the FD error
  floor) — in contrast to ROS3P's O(dt) (a few percent) on the same semilinear problem.
- **`SemilinearRHS` required**: a plain callable is rejected with a `TypeError` before
  the solve; a `priorStep` is rejected with a `RuntimeWarning`.

**Cost** (sine IC, `dt = 0.02`, 20 steps; `SolveDiagnostics` summed over the two
stages) — the roadmap's cost gate is assessed on this:

| family      | scheme  | dt   | steps | `N` evals | `L`-matvecs | total | rel L2 err |
|-------------|---------|------|-------|-----------|-------------|-------|------------|
| implicit    | BDF2    | 0.02 | 20    | 200       | 193         | 393   | 3.16e-3    |
| implicit    | TR-BDF2 | 0.02 | 20    | 400       | 175         | 575   | 7.81e-4    |
| exponential | ETD2RK  | 0.02 | 20    | 40        | 1742        | 1782  | 2.33e-3    |

ETD2RK's 40 `N` evaluations are few (two per step), but its cost is dominated by the
**1742 `L`-matvec Krylov iterations**: each of the three per-step builds runs to the
full `m = 30` because `||h·L|| ~= 3.3` does not drive the Arnoldi residual below
`1e-10` early. On the work-unit metric ETD2RK (1782) is **not** competitive with the
order-2 bars (BDF2 393, TR-BDF2 575). **FLOP caveat:** each `L`-matvec is a cheap
`nu·u_xx` stencil, whereas each JFNK `rhs_evaluation` / GMRES iteration is a full
`N + L` (Jacobian-vector) evaluation, so the work-unit metric **overstates** ETD2RK's
true FLOP cost. The roadmap's exponential **cost gate is therefore assessed at
`exprb32`** (order 3) — the method that should carry the family's cost case — before
the exponential method set expands; the **order gate is met** (order 2). (Assessed in
§3.16: **met** — EXPRB32 is the cheapest order-3 method on the fixed-error work-unit
metric, 640 < ROS3P 706 < ESDIRK3 715 < ARK3 793.)

**Carve-outs.** Because ETD2RK needs the `linear` accessor, the generic *plain-
callable* problems (oscillator, forced, kepler, advection) cannot run it as registered.
`conftest.NEEDS_SEMILINEAR_RHS = {'ETD2RK'}` skips those in the `scheme`-fixture
tests; `tvd_analysis.SEMILINEAR_ONLY_IDENTIFIERS = {'etd2rk'}` records its TVD/SSP
verdict as "not applicable (structured semilinear RHS required)" (the model advection
problem has no `linear` part; Rosenbrock-W is *not* in that set — it falls back to the
combined `f`'s Jacobian); and `test_hamiltonian`'s direct-`IntegrationSchemes` area test
skips it the same way. Its order, driver contract, and gradient are covered in
`tests/test_exponential.py` on the semilinear viscous-Burgers problem instead.

**New tests / artifacts:** 10 in `tests/test_exponential.py`; ETD2RK rows in
`tests/test_tvd.py`, `tests/conftest.py`, `tests/test_hamiltonian.py`;
`benchmarks/exponential_benchmark.py` → `docs/assets/exponential_benchmark.png`; the ETD2RK
section of `examples/notebooks/viscous_burgers_demo.ipynb`.
