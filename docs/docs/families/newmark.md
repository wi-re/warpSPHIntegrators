---
sidebar_label: Newmark
---

# Newmark beta/gamma (`Newmark`) — 1 scheme

The classical **Newmark** time integrator for second-order dynamics
$x'' = f(t, x, v)$, driven by `newmark.py` — a repo-native version of the
standard form:

$$
x_{n+1} = x_n + \mathrm{dt}\, v_n + \mathrm{dt}^2\big( (1/2 - \beta)\, a_n + \beta\, a_{n+1} \big),
$$
$$
v_{n+1} = v_n + \mathrm{dt}\big( (1 - \gamma)\, a_n + \gamma\, a_{n+1} \big),
$$

with the implicit unknown $a_{n+1}$ closed by the same nonlinear-solve
machinery as the other implicit families (`JFNKSolver` by default; a
fixed-point solver is accepted through the usual `solver=` override).

## Parameters and variants

$\beta, \gamma \in [0, 1]$ (validated at construction). The two registered
variants:

| Scheme | $\beta$ | $\gamma$ | Note |
|---|---|---|---|
| Newmark (default = `newmark_average_acceleration`) | $1/4$ | $1/2$ | constant **average acceleration** — the standard structural-dynamics choice; unconditionally stable for the linear oscillator, no numerical damping |
| `newmark_linear_acceleration` | $1/6$ | $1/2$ | **linear acceleration** — the standard with numerical damping (energy dissipation) of the linear oscillator |

The default `newmark(state, dt, f, ..., beta=0.25, gamma=0.5)` accepts any
admissible pair. The RHS must expose a velocity derivative — an update with a
`dudt` field, or fields tagged `velocity_derivative` / `acceleration` — which
is what supplies $a_n$ and the implicit $a_{n+1}$ guess.

## Properties

Order 2. For the linear oscillator the average-acceleration variant is
unconditionally stable (its frequency response is bounded for all
$\mathrm{dt}\,\omega$); the linear-acceleration variant adds the standard
numerical damping. Newmark is **not** symplectic in general (measured
dissipative in the Hamiltonian test suite — it is absent from the symplectic
sets in `tests/test_hamiltonian.py`; the geometric alternatives for
conservative dynamics are the [symplectic family](symplectic) and
Gauss-Legendre 2, [coupled-rk](coupled-rk)).

The step's cost is one nonlinear solve (plus the endpoint evaluation recorded
for the history) and the standard `SolveDiagnostics` entry. `priorStep` is
refused; `history=` is threaded like the other one-step implicit schemes.

## When to use

- Structural/second-order dynamics where the position and velocity both need
  the implicit endpoint (contact, constrained systems): Newmark gives both in
  one solve, with the damping behaviour of the $\gamma$ choice.
- Not for conservative Hamiltonian dynamics where energy fidelity over long
  runs is the point — use a symplectic scheme instead.

## References

- Newmark, N. M., *Equations of motion for structural dynamics*, Journal of Engineering Mechanics, ASCE 77 (1959) 679-709 — the $\beta$-$\gamma$ family.
- Hughes, T. J. R., *The Finite Element Method* (1987), Ch. 15 — the unconditional-stability and damping analysis.

**Tests:** `tests/test_newmark.py` (order 2, the linear-oscillator stability
and damping of the two variants, the gradient through the solve).
