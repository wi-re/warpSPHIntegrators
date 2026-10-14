# Benchmarks and analysis drivers

Install `python -m pip install -e '.[benchmark]'` from the checkout root.
Invoke drivers by their path, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/step_reuse_convergence.py --scheme RK4
python benchmarks/tvd_classifier.py --help
```

| Driver | Purpose |
|---|---|
| `adaptive_benchmark.py` | Caller-driven adaptive helpers and estimates |
| `blockrk_benchmark.py` | Historical two-stage coupled numerical comparisons |
| `implicit_scale.py` | Phase 6 isolated-process DOF, warm-start, preconditioner, timing and native-memory scale gate |
| `implicit_scale.py --spatial`, `spatial_preconditioners.py` | Sparse reference/ILU and shifted spatial solve follow-up |
| `implicit_scale.py --spatial --registered --selected-only` | Actual s=3 registration gate, direct reference and preferred shifted ILU with companion work |
| `firk_registration.py` | Known oscillator/cubic-decay accuracy and measured RHS/Krylov work for s=2/3 |
| `structural_modes.py` | Coupled-mass generalized-alpha prototype: modal energy/phase, Newmark parity, order, spectral radius and source controls |
| `generalized_alpha.py` | Package driver versus independent structural reference: accepted acceleration, modal energy, resolved order and directly counted RHS/GMRES work |
| `higher_collocation.py` | Parameterized Gauss/Radau spatial scale through nine stages, with real Schur and shifted sparse ILU |
| `collocation_accuracy.py` | Known oscillator/cubic-decay refinement, full RHS work and strict-solve precision floors for stage-parameterized collocation |
| `variable_adams.py` | Known vector reaction at fixed/alternating steps, adaptive accuracy/work and startup/steady estimator exponents |
| `stiff_startup.py` | BDF implicit/explicit startup and low-order control |
| `splitting_validation.py` | Known oscillator orders/contraction/energy, Burgers reference/CFL and qualified exact-flow merging |
| `jfnk_preconditioner_benchmark.py` | Krylov/preconditioning cost |
| `stiff_benchmark_suite.py` | Stiff/nonlinear accuracy and cost |
| `rkc_benchmark.py`, `rosenbrock_benchmark.py`, `exponential_benchmark.py` | Semilinear family comparisons |
| `stability_gallery.py`, `oscillator_stability_gallery.py` | Scalar, split and oscillator stability |
| `newmark_demo_gallery.py` | Newmark/implicit trajectory comparisons |
| `step_reuse_convergence.py` | Reuse order and measured errors |
| `tvd_classifier.py` | Registry-wide TVD/SSP measurements |

Gallery scripts locate `docs/assets` relative to their own file, so launching
outside the checkout does not redirect figures. `step_reuse_convergence.py --plot`
uses the caller's requested output path. Running gallery scripts can overwrite
tracked figures; review those changes. Record environment, hardware, size,
tolerances and command with new benchmark conclusions. Historical records live
under [numerical research](../docs/research/integrators/notes-index.md).
