# Explicit linear multistep — Adams-Bashforth & PECE (`MultistepExplicit`) — 7 schemes

The explicit linear multistep family, driven by `multistep.py`:

$$
y^{n+1} = y^n + \mathrm{dt}\, \sum_{j=0}^{p-1} \beta_j\, f^{n-j},
$$

where $f^{n-j} = f(t^{n-j}, y^{n-j})$ are **stored past right-hand-side
evaluations** carried in the scheme's `StepHistory` (derivative history,
threaded across calls with `history=`). No solves.

## Adams-Bashforth (pure predictor)

`AB2`–`AB5` evaluate the RHS **once** per step (at the new point) and apply the
order-$p$ Adams-Bashforth weights to it and the stored history. Weights
(newest first, $\beta_0$ weights $f^n$), from `getABCoefficients`:

| Order | $\beta$ weights (newest first) |
|---|---|
| 2 | $\tfrac{3}{2},\ -\tfrac{1}{2}$ |
| 3 | $\tfrac{23}{12},\ -\tfrac{16}{12},\ \tfrac{5}{12}$ |
| 4 | $\tfrac{55}{24},\ -\tfrac{59}{24},\ \tfrac{37}{24},\ -\tfrac{9}{24}$ |
| 5 | $\tfrac{1901}{720},\ -\tfrac{2774}{720},\ \tfrac{2616}{720},\ -\tfrac{1274}{720},\ \tfrac{251}{720}$ |

Cost: **1 evaluation per step** after startup — the cheapest order-2+ method in
the registry. The trade: zero-stable but not A-stable, with a bounded (and
order-dependent) stability region on the real axis; and a cold start.

## Adams-Bashforth-Moulton (PECE predictor-corrector)

`ABM2`–`ABM4` run a **predictor–corrector** pair per step:

1. **P**redict $y^{n+1}$ with the order-$p$ Adams-Bashforth weights (the table
   above) — no new evaluation;
2. **E**valuate $f^{n+1} = f(t^{n+1}, y^{n+1}_{\text{pred}})$ — one evaluation;
3. **C**orrect with the order-$p$ Adams-Moulton weights (newest first,
   predicted point first):

| Order | $\gamma$ corrector weights |
|---|---|
| 2 | $\tfrac{1}{2},\ \tfrac{1}{2}$ |
| 3 | $\tfrac{5}{12},\ \tfrac{8}{12},\ -\tfrac{1}{12}$ |
| 4 | $\tfrac{9}{24},\ \tfrac{19}{24},\ -\tfrac{5}{24},\ \tfrac{1}{24}$ |

4. **E**valuate again for the history (the corrected point's derivative is
   stored) — a second evaluation.

So ABM costs **2 evaluations per step** — the same cost as the *implicit*
Adams-Moulton of the same order ([multistep-implicit](multistep-implicit.md)),
but with a wider (explicit) stability region and no solve. The pairing of the
order-$p$ AB predictor with the order-$p$ AM corrector is what makes the PECE
scheme order $p$ (NOTES.md §3.6).

## History and cold start

The schemes carry derivative history in a `StepHistory` (NOTES.md S2): a cold
call (fewer than $p-1$ entries) runs **Dormand-Prince 5(4)** for that step and
records the history — a high-order, order-faithful startup, so the run is never
silently wrong, just at the starter's cost until the history fills (NOTES.md
§3.7). `priorStep` reuse is not the mechanism here — the history *is* the
reuse: each step's stored $f^{n}$ is exactly what the next step's formula
needs.

The measured cold-start TV jump ($+1.333\times10^{-2}\cdot\mathrm{TV}_0$, gone
after five steps) is shared by every self-starting multistep scheme and is not
a property of the step map (NOTES.md §3.11).

## Stability and TVD

Zero-stable, with bounded real-axis stability regions that shrink with order
(AB2: $[-1, 0]$; AB3: $\approx[-0.54, 0]$; AB4: $\approx[-0.32, 0]$;
AB5: $\approx[-0.17, 0]$) — nothing here is A-stable. Per-step TVD measured to
CFL 1.5 (tested to CFL 5), shared by the whole multistep family, explicit and
implicit (NOTES.md §3.11).

## When to use

- **AB2-AB5**: the cheapest way to buy order 2-5 on a non-stiff problem whose
  RHS is expensive to evaluate (one evaluation per step).
- **ABM2-4**: same cost as the implicit Adams-Moulton, no solve, wider
  stability region — the default choice when an implicit solve is not worth it.
- Not for stiff problems: the real-axis stability intervals shrink with order,
  and nothing here is A-stable.

## References

- Hairer, Nørsett, Wanner, *Solving ODEs I* (1993), Ch. 4 — Adams methods, PECE.
- Lambert, *Computational Methods in Ordinary Differential Equations* (1991).

**Tests:** `tests/test_multistep.py` (orders, the startup, the PECE cost
accounting, stability).
