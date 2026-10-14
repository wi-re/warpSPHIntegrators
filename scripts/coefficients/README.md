# Coefficient extraction and independent oracles

Promoted from preserved local scratch on 14 October 2026. The
[promotion manifest](promotion-manifest.json) records the original paths and
SHA-256 hashes, plus hashes of the maintained copies. Changes make checkout/data
paths portable and correct the original unrun wave-1 Julia argument/output format.
Text dumps have trailing whitespace normalized; their numerical values and
production coefficients were not altered. Original bytes remain in local scratch
and their hashes are retained in the promotion manifest.

## Pinned source

`julia/Project.toml` and `julia/Manifest.toml` preserve Julia 1.13.1 and
OrdinaryDiffEq 7.8.1, metapackage tree
`bae9b8e5f1d00e51c554648e5094e215164a19f8`, with subpackage git-tree pins.
On a machine with Julia and registry/network access, instantiate once:

```bash
julia --project=scripts/coefficients/julia -e 'using Pkg; Pkg.instantiate()'
```

The extracted source dumps and recorded one-step outputs are retained next to
their scripts. They permit Python checks without downloading Julia packages.
The full manifest is historical; it is not a requirement of the Python package.

## Reproduction

From the checkout root, after installing the Python package/test environment:

```bash
python scripts/coefficients/wave6/verify_tableaus.py
python scripts/coefficients/wave7/assemble_wave7.py
python scripts/coefficients/wave7/crosscheck_wave7.py
python scripts/coefficients/wave8/exact_values.py
```

Wave 6 reconstructs the source tableaus and checks their order/stability;
wave 7 assembles source data and compares with the recorded independent Julia
one-step outputs; wave 8 compares registered first-batch coefficients and emits
the later batch's literals. The exploratory `wave8/probe_wave8.py` intentionally
reports source-node and low-precision discrepancies; its failure status is not
the package validation gate. [Section 3.28](../../docs/research/integrators/notes-3-28.md)
records the accepted corrections and final numerical gate.

To refresh independent outputs with the pinned Julia environment:

```bash
julia --project=scripts/coefficients/julia scripts/coefficients/wave6/extract_tableaus.jl > scripts/coefficients/wave6/extract_out.txt
julia --project=scripts/coefficients/julia scripts/coefficients/wave6/onestep_check.jl
julia --project=scripts/coefficients/julia scripts/coefficients/wave7/extract_wave7.jl
julia --project=scripts/coefficients/julia scripts/coefficients/wave7/onestep_wave7.jl
julia --project=scripts/coefficients/julia scripts/coefficients/wave8/extract_wave8.jl
julia --project=scripts/coefficients/julia scripts/coefficients/wave8/onestep_wave8.jl
```

The wave-1 exploratory pair is also preserved (`wave1/ode_reference.jl`,
`wave1/wave1_tableaus.py`). Run the Julia oracle before the Python comparison;
absence of its output means the original probe cannot validate trajectory matching.
Original scratch files are retained locally and ignored. New measurements belong
in dated tracked records with commands, versions and numerical qualifications.

## Phase 8 / first wave-9 batch

`wave9/oracle.jl` checks the complete cached monorepo git-tree hash before
evaluating its FIRK constructors. It uses Julia stdlibs and Golub-Welsch
quadrature, so no FIRK package installation or network is required:

```bash
julia --startup-file=no scripts/coefficients/wave9/oracle.jl /path/to/OrdinaryDiffEq/OnlvZ > /tmp/wave9-oracle.json
python scripts/coefficients/phase8_inventory.py /path/to/OrdinaryDiffEq/OnlvZ --output /tmp/phase8-inventory.json
```

The tracked `wave9/oracle.json` contains source hashes, Gauss source quadrature,
Radau inverse-transform reconstruction and independent nonlinear dense-Newton
one-step states/errors. `tests/test_firk_wave9.py` compares the proposed production
coefficients and execution against these independent values. Gauss's companion is
from the source; Radau's solution companion is an explicitly recorded unsmoothed
transformation, not a transcription of upstream's filtered adaptive controller.
The inventory retains macro-generated types, aliases, package versions and order
expressions for waves 10–11; enumeration alone does not approve coefficients.

## Subsequent backend oracles

`wave9/higher_stages.jl` checks higher Gauss/Radau coefficients and nonlinear maps.
`wave10/variable_adams.jl` evaluates the source divided-difference helpers on
uniform/irregular history grids. `wave10/implicit_extensions.jl` executes source
MEBDF2 and CNLF2 constant-cache steps with independent dense Newton increments.
`wave11/stabilized.jl` executes RKL1/2 and RKG1/2 constant-cache steps at supplied
fixed stages, including nonautonomous stage times. Each checks the complete
monorepo tree and records its replaced scaffolding/controller qualifications.
`wave12/nystrom.jl` executes fifteen RKN constructors and VI/VD nonlinear step
bodies, including RKN4's screened limitation. `wave12/generate.py --check` verifies
the packaged fourteen-method coefficient module against that oracle. Runtime
needs the generated Python module only, not Julia or an external JSON data file.

`wave16/oracle.jl` executes 60 LowStorageRK constructors and 61 alternating/nonalternating branches, with nonlinear and scalar-refinement controls. `wave16/generate.py --check` verifies rooted-tree/time-node screening and the generated immutable full-stage RK coefficient module; see [the validation record](../../docs/development/expanded-rk-validation.md).
