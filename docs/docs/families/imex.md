# Additive (IMEX) methods (`IMEX`) — 6 schemes

For the **split right-hand side** $y' = f_E(t, y) + f_I(t, y)$: the smooth
(non-stiff) part explicit, the stiff part implicit. The split is read by
**capability** through `rhs.resolve` (NOTES.md §3.12): an `IMEXRHS` (or a
structured `RHS` providing `explicit`/`implicit`) activates the split; an
ordinary RHS callable degenerates to the **pure-implicit limit** of the
scheme, so the standard integrator call convention stays valid.

## IMEX Euler (`IMEXEuler`)

One step: explicit Euler on $f_E$ at the start point, one JFNK solve for the
implicit endpoint:

$$
y^{n+1} = y^n + \mathrm{dt}\, f_E(t^n, y^n) + \mathrm{dt}\, f_I(t^{n+1}, y^{n+1}).
$$

Order 1; L-stable on the implicit half; the pure-implicit limit is backward
Euler. One evaluation + one solve per step.

## ARK3(2)4L[2]SA (`ARK324L2SA`)

Kennedy-Carpenter's additive RK pair: 4 stages, **implicit half order 3,
explicit half order 2, L[2]-stable**. The implicit half ($a_{\text{imp}}, b,
d$) is identical in $a/b/c$ to the registered
[ESDIRK3(2)4L\[2\]SA](dirk); the explicit half is the matching
$a_{\text{exp}}$ (full matrices in `ark.getARKTableau`), and both halves share
the single embedded $d$ vector. Stiffly accurate on the implicit side only —
the explicit half is **not** stiffly accurate, so `priorStep` is refused
(reuse would corrupt the explicit branch). Cost: one JFNK solve + $s$ explicit
evaluations per step.

## ARK4(3)6L[2]SA (`ARK436L2SA`)

Kennedy-Carpenter's 6-stage pair: implicit half order 4, explicit half order
3, L[2]-stable, carried in `ark.py` in exact rational form
($c = (0, 1/2, 83/250, 31/50, 17/20, 1)$). Note the implicit half is a
*different* 6-stage ESDIRK design from the standalone ESDIRK4(3)6L[2]SA
(different nodes: $83/250, 31/50, 17/20$ vs. $(2-\sqrt2)/4, 5/8, 26/25$), so
its coefficients are carried in full rather than by reference.

**Stability slices.** The additive pair's stability is two-parameter
(implicit eigenvalue scale × explicit eigenvalue scale); the slices are
measured in `tests/test_ark.py` (NOTES.md §3.6): the L[2] implicit half damps
its own stiff modes to $O(z^{-2})$ while the explicit half must stay inside
its own region — the practical CFL is set by the smooth part, which is the
point of the split.

## IMEX linear multistep — SBDF2, SBDF3, CNAB2

The multistep additive family (`imexmultistep.py`, NOTES.md §3.19): a BDF or
trapezoidal **implicit backbone** for the stiff part plus **explicit endpoint
extrapolation** for the smooth part.

**SBDF$p$** ($p = 2, 3$): the BDF$p$ backbone with the BDF derivative weight
$\beta$ scaling the *whole* endpoint derivative —

$$
y^{k+1} = \sum_j c_j\, y^{k-j} + \beta\, \mathrm{dt}\,
\big[ f_I(t^{k+1}, y^{k+1}) + E(f_E) \big],
$$

where $E$ is the $p$-point Lagrange polynomial through the known states,
extrapolated to $t^{k+1}$:

$$
\text{SBDF2: } E(f_E) = 2 f_E^k - f_E^{k-1}, \qquad
\text{SBDF3: } E(f_E) = 3 f_E^k - 3 f_E^{k-1} + f_E^{k-2}.
$$

Combined order $p$; order defects $O(h^3)$ (SBDF2), $O(h^4)$ (SBDF3), verified
by Taylor expansion in `tests/test_imexmultistep.py`. Dropping $\beta$ on the
explicit part amplifies it by $1/\beta$ and collapses the defect to $O(h)$ —
measured, and pinned as a regression guard. Pure-implicit limit: BDF$p$ (SBDF2
A- and L-stable, SBDF3 A($\alpha$) 86.03°); pure-explicit limit: the matching
zero-stable explicit $p$-step method.

**CNAB2** (order 2): trapezoidal on the implicit part + AB2 increment on the
explicit part:

$$
y^{k+1} = y^k + \tfrac{\mathrm{dt}}{2}\big[f_I(t^k, y^k) + f_I(t^{k+1}, y^{k+1})\big]
+ \mathrm{dt}\big[\tfrac{3}{2} f_E^k - \tfrac{1}{2} f_E^{k-1}\big].
$$

Pure-implicit limit: the trapezoidal rule (A-stable, not L); pure-explicit
limit: AB2.

**History and cold start** follow the BDF family exactly (state-snapshot
`StepHistory`, DP5 cold start, `priorStep` refused). Diagnostics: the step's
one solve plus the non-solve evaluations (the explicit-part evaluations, the
CNAB2 implicit start-point evaluation, the endpoint history evaluation) are
counted into `rhs_evaluations`, so the work-unit invariant covers the full
step cost. Per-step TVD: measured to CFL 1.5 with the rest of the multistep
family (NOTES.md §3.11).

## When to use

- **IMEX Euler**: the cheapest split; order 1 is often enough when the stiff
  part is a linear damping term.
- **ARK3/4**: the workhorse splits — L[2] damping of the stiff part at order
  3/4 with no cost on the smooth part.
- **SBDF2/3, CNAB2**: multistep splits when per-step cost must stay near one
  solve; SBDF2 is the A/L-stable one.

## References

- Ascher, Ruuth, Rapada, *Implicit-explicit Runge-Kutta methods for ordinary differential equations*, Proc. R. Soc. Lond. A 459 (2003) 427-443 — the ARK construction.
- Kennedy & Carpenter (2001/2002) — the L[2]SA designs (as shipped in SUNDIALS ARKODE v7.9.0).
- Hairer & Wanner, *Solving ODEs II* (1996) — IMEX and BDF theory.

**Tests:** `tests/test_imex.py` (IMEX Euler), `tests/test_ark.py` (ARK order,
stability slices, the pure-implicit limit), `tests/test_imexmultistep.py`
(SBDF/CNAB orders, defects, the $\beta$ regression guard), `tests/test_rhs.py`
(the capability-based split resolution).
