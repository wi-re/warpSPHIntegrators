# Examples

Install the package from the checkout root with `python -m pip install -e '.[examples]'`.
This supplies matplotlib, tqdm and Jupyter in addition to the numerical package.

- [Harmonic oscillator](harmonic_oscillator.py): standalone float64 CPU example;
  run `python examples/harmonic_oscillator.py` from the checkout root.
- [Integrator comparisons](notebooks/integrators.ipynb): plotting notebook;
  CUDA architecture setup is conditional, so CPU use is supported.
- [Viscous Burgers](notebooks/viscous_burgers_demo.ipynb): semilinear/stiff family
  comparisons using the package's own problem helpers.
- [Wave equation / JFNK](integrations/jfnk_wave_equation.ipynb): downstream
  integration example requiring compatible editable installations of `warpSPH`
  and `warpSPHCore`, including their Warp/JVP dependencies. Install those sibling
  projects according to their own instructions before running this notebook.

Launch Jupyter from the checkout root or an examples subdirectory. Notebook setup
locates the checkout and writes durable figures to `docs/assets`, not the current
directory. Stored notebook outputs are dated evidence, not a fresh run. The wave
notebook selects CUDA when available and otherwise CPU; the historical benchmark
used float32 and an NVIDIA RTX PRO 6000 Blackwell. This package's CPU merge gate
does not validate downstream GPU performance or dependency compatibility.
