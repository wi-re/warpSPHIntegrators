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
