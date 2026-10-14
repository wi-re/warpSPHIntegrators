# Phase 8 — stabilized Legendre/Gegenbauer backends

`StabilizedRK` supplies source-backed RKL1/2 and RKG1/2 outside the fixed catalog.
It binds fixed stages or a supplied spectral bound, with a hard stage cap and
source odd-stage selection for Legendre. It never silently truncates the bound.
The catalog remains 142 registrations. Frontend availability is not a gate.

The pinned OrdinaryDiffEq 7.8.1 tree is checked before the Julia oracle runs
unchanged StabilizedRK 2.6.0 constant-cache step bodies. Thirty-two autonomous/
forced nonlinear steps cover all four families at s=3/5/9/21, including RHS
times, endpoint states and an independently formed Euler difference. The oracle
forces fixed stages and replaces the eigenvalue estimator with a supplied zero
bound; upstream controllers and spectral estimation are not reproduced.

Stage times follow the same first-order consistency recurrence as stage values.
Orders one/two are measured on nonlinear time-dependent forcing at multiple stage
counts. Legendre/Gegenbauer polynomial identities and conservative real-axis
intervals are checked independently. Source Gegenbauer sizing uses half the full
polynomial interval. Real negative eigenvalue bounds do not establish complex-
spectrum/nonnormal stability, TVD or positivity.

The public estimate is an unscaled main-minus-Euler difference, exponent two.
First-order methods compare two same-order formulas; second-order methods use a
lower-order companion. The source's norm/s scaling and FSAL policy are not
reproduced. Tests cover fixed/adaptive runs, rejection/refinement, portable restart,
parameter/initial gradients, full coupled components and quantities, one finalize
with paired weights, copied endpoint fields, nonfinite rollback, backward time,
float32/64 and configuration/stage-cap guards.

Endpoint refresh costs an extra RHS call, so an s-stage recurrence costs s+1.
Working states use a short recurrence, but returned derivatives and autograd
graphs retain storage proportional to stages. No low-storage allocation benefit
is claimed. Legacy fixed RKC/RKL entries retain their contracts and time policies;
the new RKL2 factory's autonomous main map agrees, while its time-dependent
stages, estimator and endpoint refresh differ.

The final full CPU gate passes **7,436 tests, 639 skipped**, zero failures/errors,
in 311.20 seconds of pytest time (312.4 seconds with orchestration). Eight
workers use one numerical thread each. Its stored fingerprint matches the final
source/tests; all 87 focused backend cases pass. The isolated installed 0.6.1
wheel passes all imports/exports, 135 tensor drivers and adaptive StabilizedRK.
Shared numerical dependencies are used with the project isolated outside the
checkout. All 42 published Python blocks execute; 13 sketches remain labeled.
Catalog, repository links and the production site build pass. Editable test
metadata remains historical 0.5.0; source/wheel version remains 0.6.1.

The 12 fresh-process diffusion cases use known low/high Dirichlet modes at
N=64/1024/4096, h=0.01, T=0.1. The declared bars are successful finite runs,
maximum error below 0.02/0.001 at orders one/two, and additional high-water RSS
at most 256 MiB. All pass. At N=4096, RKL2 uses 11,600 RHS calls with 2.11e-4
maximum error and 132.1 MiB additional peak RSS. RKG2 uses 20,070 calls, 1.56e-4
error and 213.5 MiB. Forward Euler's real-axis stability limit requires 3,357,082
steps over this interval. That is a stability-work comparison, not equal-accuracy
or wall-time parity. The large runs explicitly allow max_stages=4096; the public
default cap remains 1000. Three independent one-thread processes run concurrently.

Twelve nonlinear forced refinements and twelve adaptive runs complement the
scale cases. Tightening local rtol from 1e-3 to 1e-7 lowers RKG2's endpoint error
from 4.21e-5 to 4.17e-9. The first-order RKG1 errors fall from 4.35e-3 to 4.30e-5,
consistent with local O(h²) control and first-order global convergence. No claim
equates requested local tolerance with global error.

Consumer evidence: [scale, accuracy and adaptive measurements](evidence/stabilized-backend.json).
Final gates: [full CPU report](evidence/stabilized-backend-full.json) and
[installed wheel](evidence/stabilized-backend-wheel.json).

```bash
julia --startup-file=no scripts/coefficients/wave11/stabilized.jl /path/to/OrdinaryDiffEq/OnlvZ
python benchmarks/stabilized_backend.py
python -m pytest -q tests/test_stabilized_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/stabilized-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/stabilized-wheel
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [pinned source step oracle](../../scripts/coefficients/wave11/stabilized.json).
