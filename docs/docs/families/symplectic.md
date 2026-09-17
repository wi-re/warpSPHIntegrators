# Symplectic / geometric position-velocity methods (`Symplectic`) — 6 schemes

Integrators for the **second-order-ODE form** $x'' = f(t, x, v)$ written as the
position/velocity first-order system. They split each step into *drift*
(position) and *kick* (velocity) substeps. For a **separable** Hamiltonian —
force depending on position alone — the splits are exactly symplectic and the
methods conserve a shadow Hamiltonian: no secular energy drift, only bounded
oscillation around the true energy.

:::note
The separability caveat is load-bearing and measured, not folklore: with a
**velocity-dependent** force the order-2+ members drop to first order
(NOTES.md, "Still open"). The order-1 members are unaffected by definition.
:::

Two evaluations per step for the order-2 members, four for the order-4
Forest-Ruth members. Stability on the harmonic oscillator: bounded for
$\mathrm{dt}\,\omega \le 2$ (the Verlet-family bound). `priorStep` reuse is
accepted where the last stage is the next step's first force evaluation
(Leap Frog, Velocity Verlet, Symplectic Euler); PEFRL/VEFRL reject it.

## Explicit Euler (position/velocity form)

$$
x^{n+1} = x^n + \mathrm{dt}\, v^n, \qquad v^{n+1} = v^n + \mathrm{dt}\, a(t^n, x^n).
$$

Order 1. The baseline for the position/velocity form (it is in the `ExplicitRK`
family — see [explicit-rk](explicit-rk); listed here because it is the
order-1 limit of this family's update pattern).

## Semi-Implicit Euler (kick-then-drift)

$$
v^{n+1} = v^n + \mathrm{dt}\, a(t^n, x^n), \qquad x^{n+1} = x^n + \mathrm{dt}\, v^{n+1}.
$$

Order 1, symplectic. The lowest-cost geometric method: one evaluation, no
half-step bookkeeping. (`euler.integrateSemiImplicitEuler`; the `Symplectic`
family member is this one, registered as "Semi-Implicit Euler".)

## Leap Frog (synchronized form)

$$
x^{n+1} = x^n + \mathrm{dt}\, v^n + \tfrac{\mathrm{dt}^2}{2}\, a^n, \qquad
v^{n+1} = v^n + \tfrac{\mathrm{dt}}{2}\big(a^n + a^{n+1}\big),
\quad a^{n+1} = a(x^{n+1}).
$$

Order 2, symplectic. Two evaluations. (Also known as the synchronized
leapfrog; the staggered "kick-drift-kick" variant keeps $v$ at half-integer
times — this library's form advances $v$ to the full step.)

## Velocity Verlet (drift-kick-drift)

$$
v^{n+\frac12} = v^n + \tfrac{\mathrm{dt}}{2} a^n, \qquad
x^{n+1} = x^n + \mathrm{dt}\, v^{n+\frac12}, \qquad
v^{n+1} = v^{n+\frac12} + \tfrac{\mathrm{dt}}{2} a^{n+1}.
$$

Order 2, symplectic. Two evaluations. The standard molecular-dynamics
workhorse; the README's oscillator benchmark compares its energy drift against
RK4 (secular, linear in $T$) and PEFRL (flat).

## Symplectic Euler (kick-drift-kick, DualSPHysics form)

The SPH community's "symplectic Euler" (the DualSPHysics KDK form): drift to
the half step, take the *full* velocity kick with the force evaluated at the
half-step position, then finish the drift with the trapezoidal velocity:

$$
r^{n+\frac12} = r^n + \tfrac{\mathrm{dt}}{2} v^n,
\qquad
v^{n+1} = v^n + \mathrm{dt}\, a\big(t^n + \tfrac{\mathrm{dt}}{2},\, r^{n+\frac12}\big),
$$
$$
r^{n+1} = r^{n+\frac12} + \tfrac{\mathrm{dt}}{2}\big(v^n + v^{n+1}\big).
$$

Two evaluations per step; order 2; symplectic for separable forces (registered
`symplecticEuler`). It is distinct from the kick-then-drift "Semi-Implicit
Euler" above — both are in the family, and both are pinned by measurement in
`tests/test_hamiltonian.py`.

## PEFRL (order 4, Forest-Ruth)

The 4th-order Forest-Ruth *position*-first composition, Omelyan et al.'s
optimized coefficients (I.M. Omelyan, I.M. Mryglod, R. Folk, *Computer Physics
Communications* 146 (2002) 188, cond-mat/0110585). Drift substep fractions
$\xi,\ \chi,\ 1 - 2(\chi+\xi),\ \chi,\ \xi$ with kick fractions
$\frac{1-2\lambda}{2},\ \lambda,\ \lambda,\ \frac{1-2\lambda}{2}$:

$$
\lambda = -0.2123418310626054, \qquad \xi = +0.1786178958448091, \qquad \chi = -0.06626458266981849.
$$

Four force evaluations per step. Order 4, symplectic for separable Hamiltonians.
Time is treated as an extra coordinate advancing with the drift substeps (which
is what staggers the four force evaluations across the step). `priorStep`
rejected.

## VEFRL (order 4, Forest-Ruth)

The mirror image of PEFRL: kick-first, drift fractions
$\frac{1-2\lambda}{2},\ \lambda,\ \lambda,\ \frac{1-2\lambda}{2}$ and the same
$\lambda, \xi, \chi$ (the kick/drift roles of PEFRL's coefficients). Same order,
cost, and stability; the choice between the two is which split matches the
force structure of the problem.

## What symplecticity buys (measured)

On the harmonic oscillator over $T = 100$ at the same cost: RK4's energy drift
grows **linearly in $T$** (secular), while Velocity Verlet and PEFRL stay flat
(bounded oscillation) — `images/energy_drift.png`, and
`tests/test_hamiltonian.py` pins the shadow-Hamiltonian defect at
$\sim 10^{-13}$ (strict solve). On Kepler, the Verlet family's action-area
defect is $\sim 10^{-9}$ (default solve) vs. $\sim 10^{-13}$ strict
(`images/kepler_orbits.png`). The library's only symplectic method **above
order 2** is Gauss-Legendre 2 — see [coupled-rk](coupled-rk).

## References

- Verlet, L., *Computer "experiments" on classical fluids. I*, Phys. Rev. 159 (1967) 98.
- Ruth, R. D., *A canonical integration technique*, IEEE Trans. Nucl. Sci. 24 (1983) 2669-2671.
- Forest & Ruth, *Applications of implicit symplectic mapping integrators to particle dynamics*, Comput. Phys. Commun. 59 (1990) 123-130.
- Omelyan, Mryglod, Folk, *Optimized Forest-Ruth- and Suzuki-like algorithms for integration of motion in many-body systems*, CPC 146 (2002) 188.
- Leimkuhler & Matthews, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2018) — the general theory.

**Tests:** `tests/test_hamiltonian.py` (symplecticity, area defects, energy
drift classification), `tests/test_convergence.py` (orders, including the
velocity-dependent-force order drop).
