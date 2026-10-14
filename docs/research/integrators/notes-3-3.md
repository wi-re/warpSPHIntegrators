> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.3 What implicit *does* buy, even without stiffness

**Implicit midpoint is strictly better than Velocity Verlet for SPH.** It is
symplectic, A-stable, symmetric, second order — and unlike the four splitting schemes
flagged in §0, it keeps second order for a velocity-dependent force. [verified], on
`oscillator` at `dt=0.05`, maximum relative energy error over the whole run:

| scheme | T=20 | T=160 | order on `damped` |
|---|---|---|---|
| Implicit midpoint | 3.1e-15 | 6.9e-15 | **2.00** |
| RK4 | 5.5e-06 | 4.4e-05 | 4.00 |
| Velocity Verlet | 2.5e-03 | — | **1.0** |

Bounded to machine precision over an 8× longer run, versus RK4's secular growth. For
any SPH momentum equation with artificial viscosity or drag — i.e. all of them — this
is the symplectic scheme the library currently does not have, at one nonlinear solve
per step with a *single* stage. It is the single highest-value item in this section.
