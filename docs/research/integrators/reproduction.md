> Recorded commands and results; old timings are historical.
> [Current validation instructions](../../../CONTRIBUTING.md) · [Record index](notes-index.md)

## Reproducing the results

<a id="scriptsstep_reuse_convergencepy-committed"></a>

### `benchmarks/step_reuse_convergence.py` (committed)

A reusable tool for the first-stage reuse study behind §0's headline numbers. It
predicts the retained order from the tableau and measures it, so the two can be
compared:

```bash
conda activate warp

# one scheme in detail: error table, both orders, prediction vs measurement
python benchmarks/step_reuse_convergence.py --scheme RK4
python benchmarks/step_reuse_convergence.py --scheme 'SSP RK3' --problem forced

# every registered scheme, one line each
python benchmarks/step_reuse_convergence.py --all

# log-log convergence plot with reference slopes
python benchmarks/step_reuse_convergence.py --scheme RK4 --plot rk4_reuse.png
```

`--problem oscillator` is autonomous, `--problem forced` (`x'' = cos t`) is not — the
gap between them is what would expose a stage-time bug in a scheme's tableau or
hand-rolled logic, which is why `test_convergence.py` runs every scheme on both.

Sample output showing the trade concretely:

```
Scheme   : RK4 (registered order 4)
Reuse    : c_s = 1 but a[-1] != b; a[-1] is a method of order 2, so Y_s = y^{n+1} + O(dt^3)
           predicted order with reuse: 3 -- loses 1 order(s)

       dt    error (no reuse)      p     error (reuse)      p   reuse / no reuse
  0.10000          1.1612e-04     --        7.4343e-04     --               6.4x
  0.05000          7.0943e-06   4.03        1.0852e-04   2.78              15.3x
  0.02500          4.3710e-07   4.02        1.4367e-05   2.92              32.9x
  0.01250          2.7103e-08   4.01        1.8405e-06   2.96              67.9x

measured order   no reuse: 4.01
                    reuse: 2.96   (predicted 3: MATCH)
```

versus the midpoint scheme, where reuse is free apart from a fixed 2.5× constant:

```
Scheme   : Midpoint (registered order 2)
Reuse    : c_s = 0.5 != 1 ...; b[0] == 0, so the stale k0 only enters through the stage equations
           predicted order with reuse: 2 -- lossless
measured order   no reuse: 2.01
                    reuse: 2.00   (predicted 2: MATCH)
  >> Reuse is safe for this scheme: convergence order is retained.
```

<a id="scriptsadaptive_benchmarkpy-committed"></a>

### `benchmarks/adaptive_benchmark.py` (committed)

Phase 11's adaptive-control and dense-output benchmark (§3.17): the fixed-dt
ladder against a fine-dt reference plus the adaptive runs on van der Pol
μ = 10 (the roadmap gates — rejections trigger at the explicit wall, and the
accepted step count falls under control), the controller's dt / error norm in
flight, and DP5 dense output's interior interpolation rate on the linear
oscillator.

```bash
conda activate warp

OMP_NUM_THREADS=4 python benchmarks/adaptive_benchmark.py   # -> images/adaptive_benchmark.png
```

<a id="scriptsblockrk_benchmarkpy-committed"></a>

### `benchmarks/blockrk_benchmark.py` (committed)

Phase 6's block-implicit-RK benchmark (§3.18): the three panels behind
`docs/assets/blockrk_benchmark.png` — symplectic energy drift vs horizon (Gauss-Legendre
2 vs Velocity Verlet / PEFRL / RK4 on the oscillator), the stiff transient decay
at `z = −10` (Radau IIA, BE, BDF2-5, ESDIRK4(3)6 vs the A-stable-not-L pair), and
the order ladder on the linear oscillator.

```bash
conda activate warp

OMP_NUM_THREADS=4 python benchmarks/blockrk_benchmark.py   # -> images/blockrk_benchmark.png
```

### `tests/` (committed)

`pytest` from the repository root, inside the `warp` environment. Last run 2026-08-07
against `e327d8a`: **906 passed, 54 skipped in 119 s** out of 960 collected, no
failures. The skips are almost all per-scheme opt-outs from parametrised tests —
a scheme that does not implement reuse, or loses no order under it, or is unstable on
an oscillatory problem — plus the CUDA-gated cases in `test_backend_dispatch.py`.

| file | what it pins down |
|---|---|
| `test_convergence.py` | every scheme reaches its registered order on `oscillator` (autonomous), `forced` (time-dependent) and `kepler` (nonlinear); errors decrease monotonically; the four splitting schemes lose order on `damped` and nothing else does |
| `test_step_reuse.py` | measured reuse order matches `scheme.reuse_order` exactly — neither optimistic nor pessimistic; FSAL reuse is bit-for-bit free; Forward Euler refuses; the four non-reusing schemes warn, ignore, and do not leak `priorStep` into the RHS; the warning fires once and names the achieved order |
| `test_state.py` | `nograd()` on non-tensor fields; untagged-field agreement between the two clone paths; stage times translate by `dt` each step; `t` stays a Python `float`; the update helpers do not advance time; `get_scheme` accepts all three spellings; schemes do not mutate the caller |
| `test_embedded.py` | the error estimate reaches the caller, scales as `dt^p`, brackets the true error, and the *high-order* branch is the one propagated; tableau row sums and the FSAL property |
| `test_hamiltonian.py` | the `dissipation` flag predicts energy behaviour: symplectic schemes stay inside an `O(dt^p)` band over an 8× longer run, dissipative ones grow ~linearly |
| `test_copied_fields.py` | `copied()` fields arrive holding the *last stage's* value, `ephemeral()` ones do not survive at all |
| `test_backend_dispatch.py` | `wp.array`, lists, dicts and nested containers are cloned rather than aliased; `to(device)` reaches inside them; unknown types warn once; and no scheme hands the caller's own container back |
| `test_kwargs_passthrough.py` | caller kwargs survive every scheme — `verbose` in both states, alongside `priorStep`, and an unknown kwarg reaching the RHS intact |

The shared harness — three tagged-field reference systems, four problems with analytic
solutions, and the order/energy measurement helpers — lives in
[src/warpSPHIntegrators/testing.py](../../../src/warpSPHIntegrators/testing.py) so that the tests and
`benchmarks/step_reuse_convergence.py` share one definition. It doubles as the smallest
complete worked example of the protocol.
