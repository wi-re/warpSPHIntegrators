---
title: Browser laboratory
---

The interactive browser laboratory compares the library's toy problems and
integrators with real WebGPU compute kernels and a float64 CPU fallback. It uses
the same panel-based design as the boundary-integral and remote tools.

[Open the integrator laboratory](https://fluids.dev/warpSPHIntegrators/laboratory/)

Pick a physical system, select up to six methods by family and order, and run an
experiment. The trajectory view shows phase, energy and state error; convergence
refines the timestep against analytic or checked numerical references. Separate
views measure SSP/TVD stage behavior, the canonical symplectic form, absolute RK
stability, and analytic forward/transpose derivative actions. Presets include
stiff decay, Kepler, structural damping and matrix-free Newton–GMRES. Each problem
opens with a tested default setup and a description of its dynamics. Playback
animates the state and error curves; hovering or focusing any scheme entry
highlights its paths. Stability sampling defaults to 512 × 512 and can be raised
to 2048 × 2048.

The port includes all 142 fixed registered schemes plus 131 parameterized
configurations. Source coefficients are exported directly, and browser drivers
are checked against Python fixtures. GPU execution uses float32; the CPU uses
float64. Auto selects float64 for convergence studies so high orders can be
resolved before rounding dominates. The GPU paths have explicit state/stage
budgets, and the interface reports CPU fallback reasons.

This is not a replacement for every Python runtime contract. Variable-order
Adams/NDF controllers, IRKN history, custom state/lifecycle hooks, general AD,
constraint wrappers, continuous dense output, event location and Python checkpoint
interchange remain outside the port. Fixed toy RHSs provide analytic Jv/Jᵀv;
the adjoint view does not claim differentiation through the selected integrator.
The in-app **Port coverage** report gives the detailed boundaries.

For standalone local serving, coefficient regeneration, validation and numerical
qualifications, read the [browser source README](https://github.com/wi-re/warpSPHIntegrators/blob/main/web/README.md).
