---
sidebar_label: Exponential
---

# Exponential integrators (`Exponential`) — 2 schemes

Exponential integrators integrate the **linear part of a stiff problem
exactly** — through the matrix exponential and the entire $\phi$-functions —
and only quadrature the mild nonlinear remainder. The stiffness is carried by
the linear operator, so the method is unconditionally stable for the linear
part and can run comfortably past the explicit wall, with no nonlinear solve
per step (driven by `exponential.py`; NOTES.md §3.15-3.16).

The $\phi$-functions (Caliari & Ostermann, *Appl. Numer. Math.* 59 (2009)
568-581): $\phi_k(z) = \sum_{j\ge 0} z^j / (j+k)!$, so
$\phi_1(z) = (e^z - 1)/z$ and $\phi_2(z) = (e^z - 1 - z)/z^2$.

## ETD2RK (`ETD2RK`)

A two-stage, second-order exponential time-differencing RK method for
**semilinear** RHS $f = L y + N$ (requires the `linear` accessor — a
`SemilinearRHS`), the base unsplit ETD2RK of Sarumi (arXiv:2601.06849, eqs.
(7)-(8)):

$$
N_n = N(t^n, y^n), \qquad
w = e^{\mathrm{dt}\,L} y^n + \mathrm{dt}\, \phi_1(\mathrm{dt}\,L)\, N_n \quad (\text{exp-Euler predictor}),
$$
$$
y^{n+1} = e^{\mathrm{dt}\,L} y^n + \mathrm{dt}\,\phi_1(\mathrm{dt}\,L)\, N_n
+ \mathrm{dt}\,\phi_2(\mathrm{dt}\,L)\, \big(N(w) - N_n\big).
$$

Explicit in $N$ (the predictor $w$ is built from $y^n$ alone), order 2,
**L-stable for the linear part**: $N = 0$ gives $y^{n+1} = e^{\mathrm{dt}\,L}
y^n$ exactly, so stiff linear modes are damped to $|e^{\mathrm{dt}\,\lambda}|$
per step, killed as $\mathrm{dt}\,|\lambda| \to \infty$. Cost: **two** $N$
evaluations + three matrix-function-vector products
($e^{\mathrm{dt}L}$, $\phi_1(\mathrm{dt}L)$, $\phi_2(\mathrm{dt}L)$).

## EXPRB32 (`EXPRB32`)

A two-stage, **third-order exponential Rosenbrock** method with an embedded
order-2 estimator, from Hochbrueck, Ostermann & Schweitzer, *SIAM J. Numer.
Anal.* 47(1) (2009) 786-803 (their section-5 coefficients, in the section-2.3
reformulation). The difference from ETD2RK: **no split is needed** — it
freezes the **full** Jacobian $J_n = D_u f(t^n, y^n)$ and carries *all* of the
stiffness (the linear part *and* the linearization of the nonlinear part)
through the $\phi$-functions of $h J_n$:

$$
c_1 = 0,\ c_2 = 1,\ a_{21} = \phi_1, \qquad
b_1 = \phi_1 - 2\phi_3,\ b_2 = 2\phi_3, \qquad
\hat b_1 = \phi_1 \ \ (\text{embedded order 2}).
$$

In the reformulation this costs **two** $\phi$-Krylov builds per step:

$$
U_2 = y^n + h\, \phi_1(h J_n)\, F_n, \qquad
D_2 = f(t^n + h, U_2) - F_n - J_n (U_2 - y^n),
$$
$$
y^{n+1} = U_2 + 2h\, \phi_3(h J_n)\, D_2, \qquad
\hat y = U_2 \quad (\text{error} = 2h\, \phi_3(h J_n)\, D_2).
$$

The embedded estimate is the correction itself — free, no extra stage. For a
**non-autonomous** RHS the HOS section-6 treatment applies: the stage gains
$h^2 \phi_2(h J_n) v_n$ with $v_n = f_t(t^n, y^n)$ (a third build; probed by
the same one-sided finite difference as the Rosenbrock driver; skipped for
autonomous RHS).

**Properties.** Exact on linear problems ($y^{n+1} = e^{h J_n} y^n$, $D_2 = 0$);
order 3; L-stable for the linear part (scalar stability function $e^{h\lambda}$);
**not** stiffly accurate ($U_2 \neq y^{n+1}$ — no first-stage reuse); fully
explicit in the nonlinearity ($J_n$ is applied only through the $\phi$ builds,
never inverted). Cost: **two** full $f$ evaluations, one $J_n$ application,
**two** $\phi$-Krylov builds (three with the non-autonomous $v_n$ term). There
is deliberately **no** `w='linear'` option: a $J = L$-only frozen operator is
only first order on a genuinely nonlinear problem — the same order drop the
[Rosenbrock](rosenbrock) driver documents.

## Matrix-free $\phi_k(\mathrm{dt}\,L)\,v$

The linear operator is applied **matrix-free** through the Phase 14 `linear`
accessor; no dense matrix is ever formed. Each $\phi_k(\mathrm{dt}\,L)\,v$ (and
$e^{\mathrm{dt}L}v$) is a Krylov approximation (Hochbrueck's method): build
the Arnoldi basis from $v$ (the same Hessenberg process as `jfnk.gmres`),
project $\mathrm{dt}\,L$ to the small dense Hessenberg $H$, and apply $\phi_k$
to that block — $\phi_k(\mathrm{dt}L)v \approx \|v\|\, V\, \phi_k(H) e_1$, a
short Taylor-vector series since $\phi_k$ is entire. One Arnoldi build per
distinct RHS vector (three per ETD2RK step); the $L$-application count is the
step's Krylov-iteration count.

## Gradients

- **ETD2RK**: $L$ is constant, so the three matrix-function actions are
  *constant linear maps* — each is re-attached with its **transposed** operator
  as the adjoint ($e^{(\mathrm{dt}L)^T}$, $\phi_k(\mathrm{dt}L)^T$), by backward
  hooks, with **no structural approximation**. The only adjoint error is the
  Krylov tolerance of the transpose $\phi$-actions. (Self-adjoint $L$, like the
  periodic central-difference Laplacian, lets the adjoint reuse the forward
  accessor.)
- **EXPRB32**: $J_n$ depends on $y^n$, so the $\phi$ actions are *not*
  constant maps; the driver re-attaches the *frozen* $J_n$ linearization (each
  $\phi_k(h J_n)v$ with the transposed frozen operator as adjoint), dropping
  the $dJ_n/du$ Hessian terms. Because every occurrence of the frozen operator
  in the step map carries a factor of $h$, the dropped terms are
  $O(h^2)\,\|dJ_n/du\|$ — one power better than the Rosenbrock frozen-$W$
  $O(h)$. Exact up to the Krylov tolerance on linear problems.

## References

- Sarumi, arXiv:2601.06849 — the unsplit ETD2RK base scheme.
- Caliari & Ostermann, *Exponential integrators for general linear operators*, Appl. Numer. Math. 59 (2009) 568-581 — the $\phi$-functions.
- Hochbrueck, Ostermann, Schweitzer, *Exponential Rosenbrock and adaptive time integrators for semilinear PDEs*, SIAM J. Numer. Anal. 47 (2009) 786-803 — EXPRB32.
- Hochbrueck, *Unconditional stability of a Krylov exponential time differencing integrator* (1997) — the Krylov $\phi_k$ method.

**Tests:** `tests/test_exponential.py` (orders, linearity exactness, the
non-autonomous $v_n$ term, the gradient re-attachment).
