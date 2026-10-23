# Operator splitting design and application contracts

Reconciled 23 October 2026. Physical composition is implemented for declared
autonomous splits on known toy consumers. This reference preserves the structural
and application arguments from the retired design, with qualifications drawn
from the [implemented contracts](../docs/operator-splitting.md) and
[Phase 7 measurements](phase7-foundation-validation.md). Production SPH changes
require application validation; [the execution plan](../../REPOSITORY_PLAN.md)
owns scheduling.

## Flow decomposition and closure

For positions q and velocities v, the intended decomposition is

```text
H: q' = v,  v' = F_cons(q; frozen thermodynamic parameters)
D: q' = 0,  v' = F_visc(q, v)
```

The D flow freezes geometry. A neighbor topology built solely from that geometry
can remain fixed within the subflow; any changing supports, boundary state or
other topology inputs need their own invariance contract. A linear autonomous
action A at frozen configuration admits exp(hA)v. A nonlinear dissipative field
instead requires a qualified numerical substep such as implicit midpoint.
The current exact-flow factory uses a dense Torch matrix exponential for small
systems. It does not supply sparse SPH assembly or a generic matrix-free flow.

Velocity-independent force makes the kick exact for the declared reduced system.
Symplecticity additionally requires the force to be a potential gradient in
canonical variables. A kick with Jacobian blocks [I,0; h*dF/dq,I] has determinant
one, but preserves the canonical form only when dF/dq is symmetric. Volume
preservation alone does not certify a Hamiltonian or bounded energy error.

Thermodynamic closure is part of this contract:

| Closure | Consequence |
|---|---|
| Summation density rho(q), fixed entropy or entropic function | A variational discretization can have H=T(p)+V(q), with V=sum m*u(rho(q),s). Adaptive smoothing corrections must also depend only on the frozen parameters and q. |
| Independently integrated density | Density evolves with velocity; a position-only force expression is insufficient to justify a separable kick/drift map. |
| Summation density with integrated internal energy u | Even at fixed q, velocity-dependent thermal work can change pressure and force. Freezing u changes the subflow and requires a separate consistency argument. |
| Integrated total energy E | Reconstructed u=E/m-v²/2 can make pressure itself velocity-dependent. A published pressure-energy formulation is distinct from pressure-entropy SPH. |

The historical application assessment distinguishes variational compSPH,
non-variational CRKSPH and pressure-energy/pressure-entropy variants. It motivates
tests of actual force and thermal contracts; it does not certify today's frontend
implementation. Conservative differencing and pairwise momentum conservation do
not by themselves establish a potential. Integrated thermal fields cannot be
ignored just because the force's immediate arguments appear position-only.

For nonseparable conservative blocks, a symmetric implicit base may be useful
without an explicit Verlet factorization. A physical symplectic claim still
requires the appropriate canonical structure; arbitrary extra state variables
do not acquire that structure by selecting Gauss–Legendre.

## Order and negative substeps

Lie–Trotter is first order. Strang is second order when both part maps have
sufficient order; a backward-Euler dissipative half reduces the composite to
first order. Symmetry is an additional requirement for reversibility and the
usual symmetric higher-order composition. Explicit midpoint demonstrates that
second-order accuracy alone does not require a symmetric part integrator.

Replacing Verlet with a fourth-order conservative map does not raise Strang's
overall order: the splitting commutator error remains. Its error constant can
improve in a weak-dissipation regime, but the crossover is a measurement target.
For a symmetric second-order base, the Yoshida triple jump uses

```text
gamma1 = 1/(2 - 2^(1/3))
gamma0 = -2^(1/3)/(2 - 2^(1/3))
Y4(h) = S(gamma1*h) S(gamma0*h) S(gamma1*h)
```

A negative dissipative step amplifies decaying modes in intermediate states.
Commuting frozen exact exponentials can telescope analytically, while intermediate
overflow/cancellation and interleaved conservative motion still matter.
The implemented Yoshida4 restricts the conservative method, requires a static
exact linear D with nonpositive symmetric part, and bounds each negative step
by abs(tau)*norm(D,2) <= negative_step_limit. It is not an unconditional stiff
recipe or a generic nonlinear backward-dissipation method.

The old Suzuki five-stage proposal uses four coefficients
1/(4-4^(1/3)) and a middle coefficient 1-4/(4-4^(1/3)). Its largest negative
coefficient has smaller magnitude than Yoshida's, but that arithmetic alone is
not an established stability advantage for a complete application. Suzuki,
Blanes–Moan, higher orders, processing and complex-coefficient recipes remain
deferred until a consumer, state contract, order and intermediate-growth/cost
gates justify them. The original real-coefficient barrier concerns general
splitting constructions; it should not be read as a theorem about every possible
numerical method.

## Geometry and physical energy budgets

Dissipative composites generally are not symplectic. For constant scalar damping
v'=-gamma*v, an exact D flow composed with symplectic H maps scales the canonical
form by exp(-gamma*h). In one degree of freedom this also gives the determinant
factor exp(-gamma*h); for n degrees of freedom the determinant factor is
exp(-n*gamma*h). A reduced Jacobian or determinant test is not a general
symplectic-form proof for hidden-state or higher-dimensional maps.

Phase 7 measures the oscillator determinant and energy change plus integrated
viscous loss. These are qualified constant-damping/toy diagnostics. A symplectic
H submethod need not preserve the physical Hamiltonian exactly, so splitting
does not prove that every observed joule of numerical loss is physical. Keep
the energy budget, refinement and solve residuals as separate measurements.
The legacy boolean dissipation flag was not extended to a universal conformal
classification; such metadata needs a defined consumer and dimensional contract.

## Compatible energy as an application map

The historical compSPH/CRKSPH compatible update uses both endpoint velocities,
step size, pairwise accelerations, branching work partitions and stage/Butcher
data. It is an algebraic step map rather than an ordinary autonomous RHS.
An outer finalizer consuming a monolithic RK quadrature cannot silently finalize
a group of split substeps.

The retained alternative is **proposed substep-local work accounting**. For a
substep tau with v_i^+=v_i^-+tau*a_i and a_i=sum_j a_ij, define

```text
vbar_i = (v_i^- + v_i^+)/2
Delta u_i = tau * sum_j f_ij * (vbar_j - vbar_i) dot a_ij
m_i*a_ij = -m_j*a_ji,    f_ij + f_ji = 1
```

Pairwise antisymmetry and complementary partitions make the sum of kinetic and
thermal energy changes zero algebraically, including signed tau. Closing that
budget at each substep telescopes over a composition. This identity does not
prove positivity, thermodynamic admissibility, reversibility or the advertised
trajectory order of the complete coupled map.

The midpoint velocity requires the neighbor's total acceleration. One pass can
accumulate particle accelerations and a second can recompute pair contributions
at unchanged geometry to distribute work, trading persistent pair arrays for
per-particle storage and extra force arithmetic. Dropping the acceleration
correction changes the energy identity. A memory saving and its force cost must
be measured with the actual lifecycle, traversal and data ownership.

Frozen pair partitions avoid changing a discontinuous branch inside a nonlinear
solve. Freezing them alone does not establish reversibility: force evaluation,
thermodynamic refresh and the reverse step must also form a symmetric map.
A symmetric energy projection around a base step is another research design,
not an implemented package guarantee. Smooth/frozen branch choices need their
own derivative and physical validation. The proposed substep-local distribution
also differs from the original monolithic compatible discretization; equal
conservation does not establish equal local heating.

Before promotion, require per-substep signed-step conservation, endpoint order,
reverse-step checks where symmetry is claimed, gradients away from and across
partition switches as appropriate, measured memory/work, and comparison of the
thermal distribution with the retained application reference. Existing consumer
map cleanup does not complete this new physical method.

## Lifecycle and merging boundaries

The implemented composition owns one outer initialize/finalize, isolated child
states, inner preprocess/postprocess and an optional initialize_substep hook.
Custom finalizers explicitly opt into grouped component results. Failure leaves
no accepted outer side effects; runner rejection/transactions remain in charge.
Component clocks are reset for the declared autonomous split; a nonautonomous
extension must define clocks rather than add component elapsed times.

Adjacent exact fixed-operator D half-flows can be combined. Phase 7 demonstrates
that saving in a whole-interval benchmark, not transparently in the synchronized
runner. Two implicit-midpoint half-steps do not equal one full midpoint step.
Variable grids, staggered state, sampling, callbacks and restart require explicit
ownership before a merging optimization is enabled. Exact-flow merging does not
provide a free Strang-minus-Lie–Trotter adaptive estimate: that comparison and its
error exponent, extra work, rollback and controller still need qualification.

## References and activation conditions

The retained design draws on Hairer–Lubich–Wanner's composition, splitting and
geometric integration arguments; Yoshida's triple jump; Suzuki's five-stage
construction; and the SPH variational/compatible-energy discussions in Price,
Monaghan, Springel–Hernquist, Frontiere–Raskin–Owen and Hopkins. The local
[Hamiltonian research proposal](../research/integrators/hamiltonian-proposal.md)
retains the original context. Primary coefficient/source pins used by actual
implementations remain in their validation and method-ledger records.

Production closure audits, pairwise compatible work, sparse exponential assembly,
adaptive split control, general nonautonomous/multiway composition and broader
cost/stability studies remain conditional projects in the
[reconciliation record](plan-reconciliation-2026-10-23.md). Toy validation and
an attractive algebraic identity do not schedule application integration.
