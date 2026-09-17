To address the separable hamiltonian leveraging the existing idea of frozen viscosity to get to the separability would be a potentially viable approach with minimal change to the interface. the rhs interface already supports split into linear and non linear so splitting into viscous and non viscous terms would also be an interface option. here's the full idea derivation:

Under a velocity-dependent force such as viscosity, a standard "2nd-order symplectic Euler" scheme—more commonly known as the Störmer–Verlet or Leapfrog method—loses its second-order accuracy and drops to a 1st-order accurate scheme ($O(\Delta t)$). [1, 2] 
Furthermore, because viscosity is a non-conservative, dissipative force, the system ceases to be Hamiltonian. Consequently, the scheme completely loses its symplectic nature. [1] 
## Why the Order Drops
When a force depends only on position $F(q)$, symmetric methods like Leapfrog or Velocity Verlet exploit algebraic cancellations over a full time step. By interleaving the position and velocity updates at midpoints, the local errors of order $O(\Delta t^2)$ cancel out, yielding global second-order accuracy ($O(\Delta t^2)$). [3, 4] 
However, when a velocity-dependent force $F(q, v) = F_{\text{cons}}(q) - \gamma v$ is introduced, you face a causality dilemma:

* If you compute the force using the velocity from the beginning of the step, the midpoint cancellation breaks down.
* If you use a predicted velocity to evaluate the force, the broken symmetry drops the formal global truncation error back to first-order ($O(\Delta t)$). [1] 

## Summary of Consequences

| Property | Independent Force $F(q)$ | Velocity-Dependent Force $F(q, v)$ |
|---|---|---|
| Global Accuracy Order | 2nd-Order ($O(\Delta t^2)$) | 1st-Order ($O(\Delta t)$) |
| Symplecticity | Preserved (Conserves phase space) | Destroyed (Dissipative forces are inherently non-symplectic) |

## Alternative Approaches
If maintaining high accuracy or proper energy dissipation is critical for your simulation, consider the following alternatives:

   1. Implicit/Semi-Implicit Formulation: You can solve the velocity update implicitly ($v_{n+1} = v_n + \Delta t \cdot F(q, v_{n+1})$). While this is still not technically "symplectic" (since the underlying physics is dissipative), it can sometimes retain 2nd-order accuracy and provides massive stability against stiff viscous forces.
   2. Euler-Richardson (Midpoint) Method: This method updates the state to a midpoint, evaluates the velocity-dependent force there, and uses that force to take the full step. This naturally retains 2nd-order accuracy for viscous forces. [5] 
   3. Standard Runge-Kutta (RK4): If geometric/symplectic conservation is no longer a constraint due to the physical friction, switching to a classical 4th-order explicit scheme like RK4 will yield much higher accuracy. [6, 7] 

Are you implementing this in code (such as a physics engine or molecular dynamics script), and would you like an example of how to properly structure a 2nd-order scheme for viscosity?

[1] [https://iopscience.iop.org](https://iopscience.iop.org/article/10.1088/0004-6256/139/2/803/pdf)
[2] [https://gafferongames.com](https://gafferongames.com/post/integration_basics/)
[3] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Verlet_integration)
[4] [https://www.unige.ch](https://www.unige.ch/~hairer/poly_geoint/week2.pdf)
[5] [https://sip.physlets.org](https://sip.physlets.org/SimpleEPubse13.html)
[6] [https://gamedev.stackexchange.com](https://gamedev.stackexchange.com/questions/105970/orbits-combine-the-pros-of-rk4-symplectic-euler-and-verlet-velocity-integrato)
[7] [https://mtao8.math.gatech.edu](https://mtao8.math.gatech.edu/papers/16JCP.pdf)


There is a small but important distinction regarding the base methods here: Symplectic Euler is actually a 1st-order method from the start, while the standard RK2 (Runge-Kutta 2nd-order) is a 2nd-order method.
However, your core intuition about how they handle viscosity is absolutely correct: RK2 maintains its 2nd-order accuracy in the presence of a velocity-dependent force, whereas symmetric/symplectic variations that rely on position-only data (like Verlet) will see their accuracy drop.
Here is a breakdown of how these methods actually compare:
## 1. Symplectic Euler vs. Verlet

* Symplectic Euler: This method is always 1st-order accurate ($O(\Delta t)$), even for a basic spring or gravity simulation. It is popular because it conserves energy over long periods, but it is not 2nd-order.
* Verlet / Leapfrog: This is likely the method you are thinking of. It is built by combining two Symplectic Euler steps to achieve 2nd-order accuracy ($O(\Delta t^2)$). When you add a velocity-dependent force like viscosity, Verlet drops to 1st-order accuracy because the symmetry that creates the 2nd-order cancellation is broken.

## 2. Runge-Kutta 2nd-Order (RK2)
RK2 (which includes methods like Heun's or the Midpoint method) explicitly samples the velocity and the force at multiple stages within a single time step.
Because it is designed from the ground up to evaluate the system's derivative as a function of both position and velocity ($f(t, q, v)$), RK2 naturally maintains its 2nd-order accuracy when viscosity is introduced. It does not rely on the clean, position-only symmetry that Verlet relies on.
## Direct Comparison

| Integration Scheme | Order with $F(q)$ | Order with Viscosity $F(q, v)$ | Preservation of Physics |
|---|---|---|---|
| Symplectic Euler | 1st-Order | 1st-Order | Loses symplecticity; introduces artificial damping/growth. |
| Verlet / Leapfrog | 2nd-Order | Drops to 1st-Order | Loses symplecticity; numerical error ruins the exact physics. |
| RK2 (e.g., Midpoint) | 2nd-Order | Stays 2nd-Order | Not symplectic, but accurately captures the physical dissipation. |

Would you like to see the mathematical breakdown of why Verlet's symmetry breaks down under viscosity, or are you looking for the best RK2 algorithm variant to use for your specific system?

A separable Hamiltonian is a specific class of Hamiltonian systems where the total energy (the Hamiltonian, $H$) can be cleanly split into two independent parts: one that depends only on momentum ($p$) and one that depends only on position ($q$).
Mathematically, it is written as:
$$H(q, p) = T(p) + V(q)$$ 
Where $T(p)$ is the kinetic energy (usually $\frac{p^2}{2m}$) and $V(q)$ is the potential energy.
## How Separability Powers Symplectic Integrators
When a Hamiltonian is separable, it allows us to split the system's differential equations into two parts that can each be solved exactly and analytically:

   1. The $T$-step (Kick): If you freeze position $q$, the momentum updates exactly via $\dot{p} = -\frac{\partial V}{\partial q} = F(q)$.
   2. The $V$-step (Drift): If you freeze momentum $p$, the position updates exactly via $\dot{q} = \frac{\partial T}{\partial p} = \frac{p}{m}$.

By alternating between these exact solutions (e.g., Drift-Kick-Drift), you form symplectic integrators like Symplectic Euler or Verlet. Because each sub-step is a shear mapping in phase space, the overall integrator is guaranteed to be symplectic—meaning it perfectly preserves phase space volume and keeps energy errors strictly bounded over long simulations.
## The Role in Higher-Order Schemes (Yoshida Composition)
Higher-order symplectic integrators, like Yoshida’s 4th-order or 6th-order methods, rely entirely on this separability.
Yoshida's method is a composition technique. It takes a 2nd-order symmetric integrator (like Verlet) and runs it sequentially forward and backward in time using very specific, mathematically derived step sizes ($c_i \Delta t$ and $d_i \Delta t$).
Because the underlying Hamiltonian is separable, the error terms at each stage combine cleanly. Yoshida's specific coefficients are chosen precisely to make the lower-order error terms ($O(\Delta t^2)$) algebraically cancel out perfectly, leaving behind a pristine 4th-order or 6th-order symplectic method.
## Why Separability Fails Completely with Viscosity
When you introduce a velocity-dependent force like viscosity, the Hamiltonian becomes completely non-separable.

   1. You cannot write a standard Hamiltonian: Viscosity dissipates energy. A system with friction cannot be written as $H = T(p) + V(q)$ because energy is no longer conserved, and the force depends simultaneously on position and velocity ($F(q, v)$).
   2. The exact sub-steps break down: You can no longer solve the "Kick" step exactly because the force at that moment depends on the velocity you are trying to change.
   3. Yoshida schemes completely collapse: If you try to force a Yoshida scheme to handle a viscous force, the algebraic error-cancellation relies on a symmetry that no longer exists. Not only does it lose its high-order accuracy (dropping down to lower orders), but running a dissipative system "backward in time" (which Yoshida schemes require via negative time steps) causes severe numerical instabilities. The friction effectively acts as a massive unphysical energy injector during those reverse steps.

## The Solution for Viscous Systems
If your system has a dominant Hamiltonian part but a small viscous part, physicists use Splitting Methods rather than pure symplectic schemes. You split the system into:
$$\text{System} = \text{Pure Hamiltonian (Separable)} + \text{Dissipation (Viscosity)}$$ 
You use a Yoshida or Verlet scheme for the Hamiltonian part, and then append a specific, often implicit update just for the viscosity.
Are you trying to simulate a system with weak viscosity (where it mostly behaves like a pendulum or orbit but slowly slows down) or strong viscosity (like a particle in thick fluid)? Knowing this can help find the right splitting method for your code.



You have hit on a classic dilemma in Smoothed Particle Hydrodynamics (SPH). SPH forces are structurally asymmetric: the pressure forces act like a non-linear spring (Hamiltonian-like, energy-conserving), while the physical kinematic viscosity (and artificial viscosity used to prevent particle penetration and numerical shocks) acts as a dissipative, velocity-dependent force. [1, 2, 3] 
Because you have a "bit of both," a straight application of standard Verlet or a pure implicit scheme creates major bottlenecks. The standard approach to solving this in modern SPH architectures relies on an operator-splitting framework tailored to the type of viscosity you are dealing with. [4] 
------------------------------
## Scenario A: Low Viscosity (Artificial/Numerical Damping)
If you are mostly using Monaghan-style artificial viscosity or small physical viscosity to dampen noise, the pressure forces dominate the system's dynamics. [1, 2] 
## 1. The Strategy: Velocity Verlet / Leapfrog with Explicit Splitting
You can preserve the 2nd-order accuracy of your pressure/advection steps by running a modified Velocity Verlet scheme. Since viscosity is weak, you feed the velocity from the previous half-step or full-step into the viscous force calculation. [5, 6] 

1. Kick 1:   v_mid = v_n + 0.5 * dt * ( F_pressure(q_n) + F_viscosity(q_n, v_n) )
2. Drift:    q_n+1 = q_n + dt * v_mid
3. Update:   Compute new density and F_pressure(q_n+1)
4. Kick 2:   v_n+1 = v_mid + 0.5 * dt * ( F_pressure(q_n+1) + F_viscosity(q_n+1, v_mid) )


* 
* Why it works: By using $v_{\text{mid}}$ in the second kick, you approximate the velocity at the midpoint of the step, which partially recovers the 2nd-order symmetry for the viscous force.
* The Catch: It is strictly explicit. Your time-step ($\Delta t$) is rigidly bound by a viscous CFL condition:
$$\Delta t \le \alpha \frac{h^2}{\nu}$$ 
Where $h$ is your smoothing length and $\nu$ is the kinematic viscosity. If your particles get close or $h$ is small, this will tank performance. [5, 7] 
* 

------------------------------
## Scenario B: High or True Kinematic Viscosity
If you are running a real fluid simulation with actual kinematic viscosity (or areas of high shear/damping), the explicit viscous CFL condition becomes an extreme bottleneck. [7] 
## 1. The Strategy: The SPH Splitting Loop (Explicit Pressure + Implicit Viscosity)
To sidestep the $O(h^2)$ time-step restriction, state-of-the-art SPH solvers (like [Bender & Koschier](https://animation.rwth-aachen.de/publication/051/) or [Weiler et al.](https://dankoschier.github.io/resources/papers/WKBB18.pdf)) split the time step into an Explicit/Geometric Phase for pressure and an Implicit Phase for viscosity. [8, 9] 

                       [ State at t_n ]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       ┌───────────────┐             ┌───────────────┐
       │ PRESSURE STEP │             │ VISCOSITY STEP│
       │  (Geometric/  │             │   (Implicit   │
       │   Explicit)   │             │ Matrix Solve) │
       └───────┬───────┘             └───────┬───────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                      [ State at t_n+1 ]


   1. The Pressure/Advection Step: Integrate the pressure forces and advect the particles using a standard Symplectic Euler or Verlet scheme. This produces an intermediate "predicted" velocity, $v^*$. [5] 
   2. The Viscosity Step (Backward Euler): Treat the viscous term as a diffuse heat equation acting on the velocity field. Solve for the final velocity $v^{n+1}$ implicitly:
   $$v^{n+1} = v^* + \Delta t \cdot \nu \nabla^2 v^{n+1}$$ [10, 11] 

## 2. Solving the Implicit System
Because SPH discretises the Laplacian operator $\nabla^2$ via kernel derivatives, this equation translates into a system of linear equations ($A \mathbf{v}^{n+1} = \mathbf{v}^*$) where $A$ is a sparse matrix containing the SPH kernel weights between neighbouring particles. [3, 9, 12] 

* 
* You can solve this very quickly using a matrix-free Conjugate Gradient (CG) solver or Jacobi iterations on the GPU.
* Why it helps: This completely destroys the viscous time-step constraint. You can take large time steps based purely on the fluid's bulk velocity (the standard Courant condition) rather than its viscosity. [6, 7, 10, 13] 
* 

------------------------------
## Recommendations for your Implementation

   1. If you are coding an WC-SPH (Weakly Compressible) Solver: Go with Scenario A. Use a 2nd-order Runge-Kutta (RK2) or a Predictor-Corrector scheme. They are far more forgiving of explicit velocity-dependent terms than a rigid Symplectic Euler loop, and they are trivial to implement without writing a matrix solver. [6] 
   2. If you are coding an PC-SPH / DF-SPH (Incompressible) Solver: Go with Scenario B. Since you likely already have a solver loop or a Pressure Poisson Equation (PPE) setup to enforce incompressibility, adding an implicit viscous loop right after the pressure correction loop fits naturally and will keep your simulation incredibly stable. [8, 14, 15] 

Are you currently using a Weakly Compressible SPH (equation of state) approach, or a Divergence-Free / Incompressible SPH method for handling your pressure?

[1] [https://academic.oup.com](https://academic.oup.com/mnras/article/408/3/1551/1073877)
[2] [https://arxiv.org](https://arxiv.org/html/1006.1524v1)
[3] [https://cs418.cs.illinois.edu](https://cs418.cs.illinois.edu/website/text/sph.html)
[4] [https://sph-tutorial.physics-simulation.org](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf)
[5] [https://www.ltas-cm3.ulg.ac.be](http://www.ltas-cm3.ulg.ac.be/MECA0470-1/MethNumAltSph.pdf)
[6] [https://gandalfcode.github.io](https://gandalfcode.github.io/gandalf-school/talks/SphMethods.pdf)
[7] [https://www.mdpi.com](https://www.mdpi.com/2073-4441/14/11/1734)
[8] [https://animation.rwth-aachen.de](https://animation.rwth-aachen.de/publication/051/)
[9] [https://arxiv.org](https://arxiv.org/html/2403.11156v1)
[10] [https://www.sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0021999122008725)
[11] [https://dankoschier.github.io](https://dankoschier.github.io/resources/papers/WKBB18.pdf)
[12] [https://www.divecae.com](https://www.divecae.com/blog/sph-basics)
[13] [https://sph-tutorial.physics-simulation.org](https://sph-tutorial.physics-simulation.org/slides/05_viscosity.pdf)
[14] [https://dankoschier.github.io](https://dankoschier.github.io/resources/papers/WKBB18.pdf)
[15] [https://sph-tutorial.physics-simulation.org](https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf)


You have an excellent eye for algorithmic structure. What you are describing is exactly an accidental or "hidden" operator splitting method disguised as an optimization trick.
In the numerical analysis and SPH literature, this technique of freezing the viscous forces at the beginning of a multi-stage time step is sometimes referred to as constant-force freezing or semi-explicit splitting.
------------------------------
## The Anatomy of the "Frozen" Approach
In a standard, formal RK4 implementation, the derivative function $f(t, q, v)$ must be fully evaluated at every sub-step. For SPH, a single full evaluation requires a costly neighbour search and kernel gradient calculation for every particle pair. Evaluating viscosity at all 4 stages means computing those pairwise viscous shear interactions 4 times per time step.
By computing the viscous force $F_{\text{visc}}(q_n, v_n)$ only at the first sub-step and treating it as a constant background force across the remaining 3 sub-steps, you are mathematically executing the following system split:
$$\frac{d}{dt} \begin{pmatrix} q \\ v \end{pmatrix} = \underbrace{\begin{pmatrix} v \\ F_{\text{pressure}}(q) \end{pmatrix}}_{\text{Dynamic Stage (RK4)}} + \underbrace{\begin{pmatrix} 0 \\ F_{\text{visc}}(q_n, v_n) \end{pmatrix}}_{\text{Frozen Base Stage}}$$ 
## Why It Behaves Exactly Like an Operator Split
This "shortcut" replicates a first-order Lie-Trotter operator splitting. Instead of integrating the pressure and viscosity simultaneously, the code is effectively saying:

   1. Freeze the current viscous state of the fluid.
   2. Advance the fluid's pressure and advection dynamics using the full RK4 loop, adding the frozen viscous force as a simple, static external force field (like gravity).

Because the viscous force does not adapt to the changing velocities within the RK4 sub-steps, the global accuracy of the entire RK4 scheme drops down to 1st-order or a degraded 2nd-order with respect to the viscosity.
------------------------------
## Why People Do It Anyway (The Pragmatic SPH Trade-Off)
Despite losing the theoretical 4th-order mathematical accuracy of RK4, this frozen approach is highly popular in production SPH engines for three major reasons:

* Massive Performance Gains: Neighbor loops and SPH viscous formulations (which involve dot products of velocity differences and position vectors) are incredibly expensive. Cutting those evaluations by $75\%$ per time step yields immediate, massive speedups.
* Numerical Noise Suppression: In SPH, high-frequency numerical noise (particle jitter) can cause velocity vectors to fluctuate wildly at small scales. If you evaluate viscosity at every RK4 sub-step, these noisy fluctuations can sometimes get amplified, leading to unphysical artifacting or artificial heating. Freezing the viscosity acts as a low-pass filter over the time step.
* It mimics physical reality (sometimes): For macro-scale fluids, viscous diffusion often happens on a slower physical time scale than the propagation of sound waves (pressure). Treating viscosity as "slow" and pressure as "fast" is a physically defensible splitting choice.

## The Major Danger: Splitting Instability
The primary risk of this approach is that it is still strictly explicit. Because the viscosity is calculated using the old velocity $v_n$, it cannot dynamically "brake" a particle that experiences a sudden, massive acceleration during the pressure sub-steps.
If your time step violates the viscous stability limit ($\Delta t > \alpha \frac{h^2}{\nu}$), this frozen force can lag behind reality, fail to damp a high-velocity shock, and cause the simulation to violently explode.
Are you leaning toward using an RK4/multi-stage scheme like this to get tighter pressure handling, or are you hoping to avoid the viscous time step bottleneck entirely?



Running an IMEX (Implicit-Explicit) scheme that pairs an explicit, symplectic integrator for the conservative forces with an implicit solver for the viscous forces is an exceptionally robust, modern approach [1]. In the context of SPH, this is essentially a Geometric-Dissipative Operator Split.
Here is exactly what happens to your simulation mathematically and physically if you combine them:
## 1. Accuracy and Order
The overall accuracy depends heavily on how you interleave (split) the steps.

* 
* If you use a simple Lie-Trotter split (First compute the explicit symplectic step, then pipe the result into the implicit viscous step), the global accuracy of the simulation drops to 1st-order.
* If you use a symmetric Strang split (e.g., half-step viscosity $\rightarrow$ full-step symplectic $\rightarrow$ half-step viscosity), you preserve 2nd-order global accuracy for the entire system while keeping the phases cleanly separated.
* 

## 2. What happens to the "Symplecticity"?
Because viscosity is fundamentally dissipative, the combined system is no longer symplectic—nor should it be, since physical viscosity destroys the conservation of phase-space volume.
However, running the conservative part (pressure, gravity) on a symplectic scheme provides a massive hidden benefit: it eliminates artificial numerical dissipation. Standard non-symplectic explicit schemes (like forward Euler or standard RK2) introduces unphysical energy drift or numerical damping. By using a symplectic scheme for pressure, you ensure that only your explicit viscous solver dampens the fluid. Your pressure forces will conserve energy perfectly, allowing you to fine-tune your fluid's physical viscosity without fighting numerical noise.
## 3. Destruction of the Viscous Time-Step Bottleneck
By treating the viscous propagation implicitly, you completely bypass the restrictive explicit viscous CFL condition ($\Delta t \le \alpha \frac{h^2}{\nu}$). Your time step is now bounded only by the standard advection Courant condition (fluid velocity and speed of sound). For highly viscous fluids, this allows you to take time steps that are orders of magnitude larger without the simulation blowing up.
------------------------------
## How to Structure It: The IMEX Strang Split Loop
To get the best of both worlds (2nd-order accuracy and strict energy tracking), you can structure your SPH loop using a symmetric Strang-split IMEX framework:

                  [ State at Step n: q_n, v_n ]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 1. IMPLICIT VISCOSITY (Half Step: dt/2)       │
        │    Solve: v* = v_n + (dt/2) * ν * ∇² v*      │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 2. EXPLICIT SYMPLECTIC (Full Step: dt)       │
        │    Use Verlet/Leapfrog on (q_n, v*) to       │
        │    integrate pressure & gravity forces.      │
        │    Yields: q_n+1  and  v**                   │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 3. IMPLICIT VISCOSITY (Half Step: dt/2)       │
        │    Solve: v_n+1 = v** + (dt/2) * ν * ∇² v_n+1 │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
                [ State at Step n+1: q_n+1, v_n+1 ]

## Potential Implementation Bottlenecks to Watch For
While mathematically elegant, implementing this in SPH requires handling a few specific architectural quirks:

* 
* The Changing Matrix: Because SPH particles move every time-step, the neighbor list changes, meaning the sparse matrix $A$ representing your implicit Laplacian ($\nabla^2$) changes every single frame. You cannot pre-factor or invert this matrix once at the start of your simulation. You must use an iterative, matrix-free solver (like Conjugate Gradient or Jacobi) that re-evaluates the kernel weights on the fly every time step.
* GPU Alignment: If your SPH solver runs on the GPU, matrix-free Jacobi iterations map incredibly well to parallel execution threads, making the implicit viscosity step surprisingly fast despite the mathematical overhead.
* 

Are you currently using an iterative solver framework (like Conjugate Gradient or Jacobi iterations) elsewhere in your code, or would you need to implement one from scratch to support the implicit stage?


That is a highly sophisticated architecture. Running an IMEX (Implicit-Explicit) scheme utilizing JFNK (Jacobian-Free Newton-Krylov) paired with Forward-Mode Algorithmic Differentiation (AD) for the Jacobian-Vector Products (JVPs) and GMRES changes the playing field entirely.
By leveraging JVPs via forward-mode AD (jvp), you completely bypass the need to analytically derive or explicitly store the massive, dynamically changing SPH Laplacian matrix. You get exact directional derivatives for your Krylov solver automatically.
Here is exactly what happens mathematically, structurally, and performance-wise when you run this specific IMEX-JFNK framework:
## 1. Mathematical Accuracy & Convergence
Because JFNK solves the fully non-linear implicit system directly (rather than just a linearized approximation), your viscous step will converge to machine precision based on your Newton tolerance.

* Pairing this with a Strang Split ensures you hit a clean, global 2nd-order accuracy.
* Furthermore, if you ever transition from a linear Newtonian viscosity model to a non-linear one (e.g., non-Newtonian blood, mud, or shear-thinning polymers), your JFNK solver will handle it natively without altering a single line of the linear GMRES solver structure.

## 2. The Interaction: Symplectic Explicit Phase + Non-Linear JFNK
When you execute the explicit symplectic step, the particles move in space ($q_n \to q_{n+1}$). In SPH, this means the neighbour graphs and kernel support radii change.
When the simulation transitions into the implicit JFNK phase, your non-linear residual function $F(\mathbf{v})$ must evaluate the SPH viscous force using these updated positions. In a standard solver, a changing grid or mesh is a nightmare. In your framework:

* The Residual Operator: Your function $F(\mathbf{v}) = \mathbf{v} - \mathbf{v}^* - \Delta t \cdot \mathcal{A}(q, \mathbf{v})$ simply runs the SPH viscosity loop natively inside it.
* The Forward AD Sweep: When GMRES asks for a JVP (J ⋅ w), your forward-mode AD passes the perturbation vector w as the dual components (tangents) into the exact same SPH viscosity loop.

## 3. Critical SPH Performance Traps with Forward-Mode AD
While elegant, wrapping an SPH loop inside a forward-mode jvp for GMRES has a few major architectural implications you must safeguard against:
## A. Keep the Neighbour Search OUTSIDE the JFNK Loop
The neighbour graph generation (e.g., spatial hashing, BVH traversal) is a non-differentiable discrete operation. Do not let your AD framework track or differentiate the neighbour search.

* You must compute the particle neighbour lists once at the beginning of the implicit step using the fixed positions q.
* Pass this static neighbour list/index buffer as a constant metadata array into your residual function. The JFNK loop should only evaluate the floating-point arithmetic of the kernel weights and velocity differences over that fixed graph.

## B. Memory Overhead of the Dual Numbers
Forward-mode AD (jvp) is generally memory-efficient compared to reverse-mode (backpropagation) because it doesn't require a massive tape or computational graph storage. It evaluates derivatives eagerly alongside the primal values.

* However, because it is running inside GMRES, you are evaluating it repeatedly. Ensure that your SPH datastructures are laid out contiguously in memory (Structure of Arrays) so that SIMD/GPU vector lanes can easily process the primal velocity and its tangent component simultaneously without cache thrashing.

## How the Solver Loop Executes

[ Explicit Symplectic Phase Ends ] -> Yields updated positions Q and predicted velocities V*

1. Build / Update SPH Neighbour Search Graph based on fixed Q.
2. Define the Residual Function: 
   Function Residual(V):
       Compute SPH Viscous Forces using fixed Q, fixed Neighbour Graph, and variable V.
       Return V - V* - dt * Viscous_Forces

3. Initialize GMRES with initial guess V_0 = V*
4. Krylov Loop (GMRES):
       When GMRES requests a Jacobian-Vector Product (J_matrix * directional_vector W):
           Call jvp(Residual, (V_current,), (W,))
           This returns the exact directional derivative without ever building the matrix.
5. Newton Update -> V_next
6. Loop until convergence tolerance is achieved.

## The Ultimate Payoff
This approach yields an incredibly robust SPH engine. The explicit symplectic portion ensures that your fluid doesn't suffer from unphysical numerical damping during advection, while the JFNK solver guarantees that your viscous forces scale seamlessly to high viscosity regimes. Because the jvp handles the complex SPH kernel derivatives exactly, your GMRES solver will exhibit optimal Krylov subspace convergence rates.
Is this JFNK setup running on the CPU (e.g., via JAX / Eigen) or the GPU (e.g., via CUDA / Taichi)? Knowing the backend can help pinpoint if your bottleneck will be memory bandwidth during the JVP sweeps or GMRES restart limits.




## Overview: Overcoming Hamiltonian Non-Separability in SPH
In classical mechanics, higher-order symplectic integrators (like Verlet or Yoshida composition schemes) rely entirely on a separable Hamiltonian where the system energy cleanly splits into position-only and momentum-only components: $H(q, p) = T(p) + V(q)$.
Because fluid pressure and gravity act as conservative, position-dependent forces ($F_{\text{pressure}}(q)$), they fit this framework perfectly. However, physical or artificial viscosity is a dissipative, velocity-dependent force ($F_{\text{visc}}(q, v)$). This completely breaks Hamiltonian separability, destroying the mathematical symmetry required for higher-order symplectic schemes and forcing their accuracy to drop to first-order ($O(\Delta t)$).
To implement a clean progression from simple, explicit handling to a robust, high-performance architecture, you can leverage different operator-splitting methods.
------------------------------
## The Evolution of Splitting Strategies
The table below outlines how different integration strategies handle the non-separable viscous terms, building toward your targeted IMEX-JFNK framework.

| Approach | Mathematical Splitting Type | How Viscosity is Handled | Impact on Symplecticity & Order | Pros & Cons |
|---|---|---|---|---|
| 1. Explicit Symmetric (Verlet / Leapfrog) | None (Fully Integrated) | Evaluated explicitly at each half/full step using standard velocity updates. | Loses Symplecticity. Drops global accuracy down to 1st-Order ($O(\Delta t)$). | 🟢 Simple to code. 🔴 Rigidly bottlenecked by the viscous stability limit ($\Delta t \le \alpha \frac{h^2}{\nu}$). |
| 2. "Frozen" Viscosity (Common SPH Trick) | Accidental / Hidden First-Order Split (Lie-Trotter) | $F_{\text{visc}}(q_n, v_n)$ is evaluated only once at the start of a multi-stage step (like RK4) and treated as a static background force. | Loses Symplecticity. Degrades the higher-order scheme's accuracy to 1st-Order for viscosity. | 🟢 Massive speedups (75% fewer neighbor loops). Acts as a low-pass noise filter. 🔴 Explodes if $\Delta t$ violates the explicit viscous limit. |
| 3. Explicit Strang Split | Symmetric Geometric-Dissipative Split | Run 1/2 step explicit viscosity $\rightarrow$ Full step symplectic pressure $\rightarrow$ 1/2 step explicit viscosity. | Preserves 2nd-Order global accuracy ($O(\Delta t^2)$). Pressure phase stays perfectly non-dissipative. | 🟢 Correctly isolates the symplectic/conservative physics from the dissipation. 🔴 Still bound by the strict explicit viscous $\Delta t$ bottleneck. |
| 4. IMEX Strang Split with JFNK (Target Future State) | Symmetric Implicit-Explicit (IMEX) Split | The pressure phase runs explicitly via a symplectic scheme. The viscosity phase is solved implicitly using JFNK and Forward AD (JVPs). | Preserves 2nd-Order global accuracy. Entirely removes the viscous time-step constraint. | 🟢 Completely destroys the viscous $\Delta t$ bottleneck. Trivially scales to non-Newtonian regimes. 🔴 Highly complex implementation. Requires careful neighbor-list caching. |

------------------------------
## Structural Implementation Blueprint: Moving Toward IMEX-JFNK
When transitioning your codebase toward the advanced IMEX-JFNK framework, the core logic should shift from a single monolithic time-stepping function into a cleanly decoupled, symmetric Strang Loop:

                       [ State at t_n: q_n, v_n ]
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ STEP 1: IMPLICIT VISCOSITY (First Half-Step: dt/2)   │
        │ - Cache the neighbour graph based on fixed q_n.      │
        │ - Solve non-linear residual via JFNK & GMRES.        │
        │ - Output intermediate velocities: v*                 │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ STEP 2: EXPLICIT SYMPLECTIC (Full Step: dt)          │
        │ - Run Verlet / Leapfrog on the pressure forces.      │
        │ - Update particle advection: q_n ──> q_n+1           │
        │ - Symplectic step outputs: q_n+1 and v**             │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ STEP 3: IMPLICIT VISCOSITY (Second Half-Step: dt/2)  │
        │ - Re-cache neighbour graph based on new q_n+1.       │
        │ - Solve non-linear residual via JFNK & GMRES.        │
        │ - Output final velocities: v_n+1                     │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                      [ State at t_n+1: q_n+1, v_n+1 ]

## Critical Rules for the JFNK & Forward AD Sub-step
To prevent massive performance degradation during the implicit JFNK phase, ensure your residual function conforms to these constraints:

   1. Isolate the Discrete Topology: SPH neighbor searching (spatial hashing, BVH lookups) is a discrete, non-differentiable operation. You must compute and cache the neighbor pairing arrays outside the JFNK loop.
   2. The Residual Target: The JFNK loop treats the particle positions $q$ as constant parameters. The Newton solver is searching only for the velocity vector $\mathbf{v}$ that satisfies the roots of the viscous residual:
   $$F(\mathbf{v}) = \mathbf{v} - \mathbf{v}_{\text{base}} - \frac{\Delta t}{2} \cdot \mathcal{A}_{\text{visc}}(q_{\text{fixed}}, \mathbf{v})$$ 
   3. Pure Arithmetic JVPs: Because the neighbor list is passed into the residual as a static metadata array, your Forward AD jvp tool will only track the continuous floating-point math (the kernel function evaluations and velocity differences), yielding exact, blazingly fast directional derivatives for GMRES.

To help plan the initial stages of this transition, what programming language or differentiation framework (e.g., JAX, Taichi, Enzyme, C++) is your current SPH simulator built on?

