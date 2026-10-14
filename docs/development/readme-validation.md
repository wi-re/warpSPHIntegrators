# Front-page explanations and reproducible examples

The initial Phase 3 consolidation made the README an index rather than a useful
introduction. User review requested the explanatory material on the front page:
scheme examples, method-selection advice, TVD/SSP, JFNK, stability, geometric
properties, known oscillator behavior, benchmark results and the scheme list.
The revised README restores that material directly and retains deeper references.

The fixed scheme tables are generated from all 142 runtime records by
`scripts/generate_catalog.py`. They include orders, stored stages, estimates,
FSAL and applicable stability/structural qualifications. IMEX labels apply to
the implicit part, exponential labels to the linear test, and Newmark's claim
is explicitly about the default linear-oscillator model. Parameterized backends
are listed separately. The generator's existing check catches front-page drift.

`examples/damped_oscillator.py` is included verbatim and executed as a complete
script by the documentation validator. It integrates x'=v, v'=-x-.4v and
Q'=.4v², compares against the analytic underdamped trajectory and checks the
mechanical-plus-thermal budget. The front page explains that its adaptive code
and the fixed-grid comparison serve different purposes.

`benchmarks/readme_gallery.py` runs five methods at h=.2 to T=20, records directly
counted RHS entries and accepted GMRES iterations, writes analytic-reference
errors over all saved endpoints, and updates the README result table. Its JSON
records the CPU model, Python/Torch/NumPy versions, dtype, thread count and source
hashes. Wall timings are retained but are not used for an equal-error speed
ranking. It also generates six compact stability panels from implemented RK
coefficients, treating poles as unstable. The two new figures were visually
inspected for clipping, labels, legibility and consistency with the explanation.
Older BDF, oscillator and structural figures remain contextual illustrations
with their existing generators and linked numerical records.

Related guide corrections remove obsolete sole-high-order-symplectic claims,
clarify the special velocity-dependent behavior of Symplectic Euler, replace
blanket DP5/multistep-startup statements with current driver contracts, and
acknowledge qualified higher-stage collocation rather than the old deferred gate.
The detailed protocol reference also uses the algebraic RK symplectic condition
instead of treating matrix symmetry as a requirement.

Validation: **51 published Python blocks execute**, 13 sketches remain labeled;
**303 catalog/documentation tests pass, four skipped**. Generated catalog,
repository links/anchors and the production Docusaurus build pass. The latest
full backend result remains **8,139 passed, 639 skipped**; its source/test/config
fingerprint still matches because this revision changes documentation,
examples and generation/measurement tools, without changing the numerical drivers.
No new full-suite run or wheel run was needed for this documentation revision.

[Oscillator measurements](evidence/readme-oscillator.json),
[latest backend gate](evidence/expanded-rk-full.json).

```bash
python -m pip install -e '.[test,benchmark]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/readme_gallery.py
python scripts/generate_catalog.py --check
python scripts/validate_doc_examples.py
python scripts/validate_repository.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_doc_examples.py tests/test_catalog.py
cd docs
npm run build
```
