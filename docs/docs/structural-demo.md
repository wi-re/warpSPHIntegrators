---
title: Structural dynamics demo
---

# Two masses, two known modes

This consumer evaluates a linear generalized-alpha prototype against the actual
registered Newmark driver and a known solution. Run from the checkout root:

```bash
python examples/structural_modes.py
python benchmarks/structural_modes.py
```

The first command prints the slow/fast energy retained. The second writes a
measurement report under `.tmp/phase8/structural-demo` and the four-panel figure
under `docs/assets/structural-modes.png`.

[example:run structural_demo]: #
```python
import runpy

demo = runpy.run_path('examples/structural_modes.py')
ratios = demo['run_example']()
assert ratios[0] > .999
assert ratios[1] < 1e-15
print('Slow/fast modal energy retained:', ratios.tolist())
```

## What the problem means

Two unit masses each have a spring of stiffness 1 to ground and share a coupling
spring of stiffness 799.5. There is no physical damping or forcing:

$$
M=I,\qquad K=\begin{pmatrix}800.5&-799.5\\-799.5&800.5\end{pmatrix},
\qquad M\ddot q+Kq=0.
$$

The in-phase mode has frequency 1 rad/s; the out-of-phase mode has frequency
40 rad/s. Their initial displacement amplitudes are 1 and 0.05, with zero
velocity. In modal coordinates the exact solution is simply
`z_j(t)=A_j*cos(omega_j*t)`, and each exact modal energy is constant.

At `dt=0.1 s`, the slow mode has about 63 samples per period. The fast mode has
only 1.57 samples per period and is underresolved. This deliberately separates
the resolved motion from a fast component that the chosen timestep cannot
represent accurately. If the fast motion matters physically, refine the step;
numerical damping does not reconstruct it or supply physical viscosity.

## What generalized-alpha changes

The prototype keeps the Newmark position/velocity formulas and balances forces
at weighted states. It carries an **algorithmic acceleration** between steps;
that acceleration is generally different from a fresh physical `-M^-1*K*q`.
For old-state weights `alpha_m/alpha_f`, the classical parameters are

$$
\alpha_m=\frac{2\rho_\infty-1}{1+\rho_\infty},\quad
\alpha_f=\frac{\rho_\infty}{1+\rho_\infty},\quad
\gamma=\tfrac12-\alpha_m+\alpha_f,\quad
\beta=\tfrac14(1-\alpha_m+\alpha_f)^2.
$$

These agree with [PETSc's second-order generalized-alpha implementation](https://petsc.org/release/src/ts/impls/implicit/alpha/alpha2.c.html)
after converting its new-state alpha weights. `rho_inf=1` recovers undamped
average-acceleration Newmark; smaller values add high-frequency numerical damping.
The parameter is a limiting amplification factor, not a fraction of energy
removed on every finite-frequency step.

## Read the results

After 12 seconds at the same fixed step:

| Method | Slow energy retained | Fast energy retained | Slow phase lag |
|---|---:|---:|---:|
| Registered Newmark / prototype rho=1 | 100% | 100% | 0.0100 rad |
| Prototype rho=0.8 | 99.997% | 0.458% | 0.0105 rad |
| Prototype rho=0.5 | 99.947% | 1.53e-18% | 0.0149 rad |
| Prototype rho=0 | 98.778% | about 3e-29% | 0.0526 rad |

Stronger damping has a visible cost in slow-mode phase and amplitude. Newmark
preserves the fast-mode energy but gives it an incorrect discrete frequency at
this step. Energy retention alone therefore does not establish trajectory accuracy.
For average-acceleration Newmark the discrete frequency is
`omega_num = 2/dt * atan(omega*dt/2)`: the 40 rad/s mode becomes about
22.14 rad/s at this step, while the 1 rad/s mode becomes 0.99917 rad/s.
Values far below roundoff-scale modal energy should be read as suppression to
the numerical floor, not many-digit predictions of a tiny physical signal.

The gallery shows slow displacement, fast energy, spectral amplification and
refinement. Resolved refinement uses frequencies 1 and 3 rad/s on a separate
grid. Carrying acceleration measures order 1.987–2.000; resetting it measures
order 0.991 for the damped variants. At rho=1 the prototype agrees with registered
Newmark to about 5e-14 in displacement. The prototype and Newmark use different
dense/matrix-free solve paths, so their raw runtimes are not a backend speed comparison.

[Open the measured gallery](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/assets/structural-modes.png).

## Scope and source checks

This is a dense, constant-matrix, unforced **linear demo prototype**, outside the
139-method catalog. It does not add a production generalized-alpha driver,
adaptive estimator or restart format. Production work must retain algorithmic
acceleration in accepted/rejected state, gradients and checkpoints, define
initialization/timestep changes and preserve lifecycle/quantity semantics.

The pinned OrdinaryDiffEq source refreshes acceleration and uses a beta formula
that differs from the classical scheme. The demo includes separate negative
controls for both issues. At rho=1 its cached beta expression gives 1/16 rather
than 1/4; the scalar amplification radius at `omega*dt=4` becomes 5.83. This
isolates the expression, not a complete run of the upstream solver. A known
consumer is therefore already catching problems that a source-name audit cannot.
See the [source audit and validation](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/structural-demo-validation.md).
