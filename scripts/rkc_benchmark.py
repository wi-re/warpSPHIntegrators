"""Phase 12: RKC / RKL vs the implicit baselines on the semi-discrete diffusion
benchmark -> images/rkc_benchmark.png.

The roadmap gate: "Benchmark against BE / BDF2 / TR-BDF2 on ``diffusion_problem``
at several ``n``, reporting RHS evaluations to a fixed error."

Every configuration is fixed here (deterministic, no adaptive steps). The cost
metric is the total number of right-hand-side evaluations -- the one quantity a
matrix-free explicit method (RKC/RKL) and a nonlinearly implicit one (BE/BDF2/
TR-BDF2) share -- counted by a wrapper around the problem's ``rhs`` so that the
implicit methods' finite-difference Jacobian matvec evaluations (the expensive
part of the default JFNK solve) are included. Each scheme is run as shipped: the
implicit schemes use their registered default JFNK (finite-difference matvec, no
preconditioner); the RKC/RKL family uses the minimal stage count
``s = stage_count(dt * |lambda_max|)`` that is stable at each ``dt``.

Two findings are pinned in the figure:

* the order-2 super-timesteppers (RKC2 / RKL2) reach the fixed error with the
  fewest total RHS evaluations at every ``n`` -- the implicit methods' cost is
  dominated by their nonlinear solve, and RKC/RKL has none;
* the implicit baselines' *practical* stability is solver-limited, not just
  method-limited: BDF2 is A-stable, yet its default finite-difference JFNK solve
  fails on this stiff problem at coarse ``dt`` (and at ``n = 64`` at every
  ``dt`` tested), so its curve terminates early. That solver bottleneck is exactly
  the regime RKC/RKL is built to avoid.

Run:  OMP_NUM_THREADS=4 python scripts/rkc_benchmark.py
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators import (
    RKC1,
    RKC2,
    RKL2,
    getIntegrator,
    get_reference_state,
    stage_count,
    testing,
)

N_VALUES = (16, 32, 64)
T = 0.1
TARGET = 1e-2          # fixed max-norm error the schemes are driven to
BLOWUP = 10.0         # max |u - u_exact| above this is read as an unstable run
DT_GRID = [0.05 / 2 ** i for i in range(8)]   # 0.05 .. ~0.0004, ratio 2

IMPLICIT = [('Backward Euler (implicit)', 'BE'),
            ('BDF2', 'BDF2'),
            ('TR-BDF2', 'TR-BDF2')]
RKC_FAMILIES = [('RKC1', RKC1), ('RKC2', RKC2), ('RKL2', RKL2)]


def lam_max(n, D=1.0, L=1.0):
    """Magnitude of the stiffest Dirichlet-Laplacian eigenvalue (~4D/h^2)."""
    h = L / (n + 1)
    return (4.0 * D / h ** 2) * math.sin(n * math.pi * h / (2.0 * L)) ** 2


def run_count(scheme, problem, dt, s=None):
    """One fixed-dt run to t=T; returns (max-norm error vs exact, total RHS evals)."""
    count = {'n': 0}

    def f(system, dt_, **kw):
        count['n'] += 1
        return problem.rhs(system, dt_, **kw)

    system = problem.initial()
    for _ in range(int(round(T / dt))):
        kw = {'s': s} if s is not None else {}
        system = scheme(system, dt=dt, f=f, **kw).state
    x = get_reference_state(system).x.detach().cpu().numpy()
    exact, _ = problem.exact(T)
    return float(np.max(np.abs(x - np.asarray(exact)))), count['n']


def curve(name, scheme_fn, problem, n, family=None):
    """(total_rhs, max_error, stable) over the dt grid, coarsest to finest."""
    lm = lam_max(n)
    rows = []
    for dt in DT_GRID:
        s = stage_count(dt * lm, family) if family is not None else None
        err, rhs = run_count(scheme_fn, problem, dt, s=s)
        stable = math.isfinite(err) and err < BLOWUP
        rows.append((rhs, err, stable))
    return rows


def rhs_to_target(rows, target=TARGET):
    """Smallest total RHS count at which a *stable* run reaches the target error.

    Returns None when no stable point on the grid reaches the target (the
    order-1 schemes at the stiffest ``n``, and BDF2 once its solver fails)."""
    candidates = [(rhs, err) for rhs, err, stable in rows
                  if stable and err <= target]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p[0])[0]


def main():
    os.makedirs('images', exist_ok=True)
    table = {}   # n -> {label: (rhs_to_target or None, rows)}
    for n in N_VALUES:
        problem = testing.PROBLEMS['diffusion'](n=n)
        per_scheme = {}
        for full, label in IMPLICIT:
            rows = curve(full, getIntegrator(full), problem, n)
            per_scheme[label] = (rhs_to_target(rows), rows)
        for label, fn in RKC_FAMILIES:
            rows = curve(label, fn, problem, n, family=fn.family)
            per_scheme[label] = (rhs_to_target(rows), rows)
        table[n] = per_scheme

        # console table
        print(f'\nn = {n}   (|lambda_max| = {lam_max(n):.0f})   '
              f'RHS evaluations to reach max error <= {TARGET:g}')
        for label in [l for _, l in IMPLICIT] + [l for l, _ in RKC_FAMILIES]:
            rtt, rows = per_scheme[label]
            if rtt is None:
                print(f'  {label:>8}:  -- (no stable point reaches the target)')
            else:
                print(f'  {label:>8}:  {rtt:6d}')

    labels = [l for _, l in IMPLICIT] + [l for l, _ in RKC_FAMILIES]
    colors = {'BE': '#3478a6', 'BDF2': '#758c37', 'TR-BDF2': '#c44e52',
              'RKC1': '#9a5b36', 'RKC2': '#5b6ca6', 'RKL2': '#a04b8f'}
    n_colors = {16: '#c9d9ea', 32: '#5b8fc4', 64: '#20456e'}

    fig = plt.figure(figsize=(15, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Panel 1: RHS evaluations to reach the fixed error, grouped by n.
    width = 0.13
    x = np.arange(len(labels))
    for gi, n in enumerate(N_VALUES):
        vals, errs = [], []
        for i, label in enumerate(labels):
            rtt, _ = table[n][label]
            vals.append(rtt if rtt is not None else np.nan)
            errs.append(rtt is None)
        pos = x + (gi - 1) * width
        bars = ax1.bar(pos, np.nan_to_num(vals, nan=0.0), width=width,
                       color=n_colors[n], edgecolor='white', linewidth=0.5,
                       label=f'n = {n}')
        # mark schemes that could not reach the target with an 'x' at the top
        top = max([v for v in vals if v is not None and not math.isnan(v)] or [1]) * 1.2
        for i, label in enumerate(labels):
            if errs[i]:
                ax1.plot([pos[i]], [top], 'x', color='#333333',
                         markersize=7, mew=1.5)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('total RHS evaluations')
    ax1.set_title(f'RHS evaluations to reach max error <= {TARGET:g}')
    ax1.grid(axis='y', alpha=0.4, which='both')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.text(0.98, 0.02, "x = no stable dt on the grid reaches the target",
             transform=ax1.transAxes, ha='right', fontsize=8, color='#333333')

    # Panel 2: cost-accuracy curve at n = 32 (total RHS vs max error per dt).
    # Plot the stable curves first (they set the axis), then mark the unstable
    # points clipped to a fixed headroom above them.
    n2 = 32
    for label in labels:
        _, rows = table[n2][label]
        stable = sorted([(rhs, err) for rhs, err, ok in rows if ok and err > 0],
                        key=lambda p: p[1])
        if stable:
            ax2.loglog([p[0] for p in stable], [p[1] for p in stable], 'o-',
                       color=colors[label], label=label)
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo, hi * 40)
    top = hi * 10
    for label in labels:
        _, rows = table[n2][label]
        unstable = [p[0] for p in rows if not p[2] and math.isfinite(p[1])]
        if unstable:
            ax2.plot(unstable, [top] * len(unstable), 'x', color=colors[label],
                     markersize=7, mew=1.5)
    ax2.axhline(TARGET, color='#333333', linestyle=':', linewidth=1)
    ax2.text(1.1, TARGET * 1.15, f'target {TARGET:g}', fontsize=8, color='#333333')
    ax2.set_xlabel('total RHS evaluations')
    ax2.set_ylabel('max |u(T) - u_exact(T)|')
    ax2.set_title(f'Cost vs accuracy, diffusion n = {n2} (x = unstable)')
    ax2.grid(which='both', alpha=0.4)
    ax2.legend(fontsize=8)

    fig.suptitle('Phase 12: RKC / RKL vs implicit baselines on the diffusion benchmark '
                 '(cost = total RHS evaluations, implicit solves use the default JFNK)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('images/rkc_benchmark.png', dpi=200)
    plt.close(fig)
    print(f'\nWrote images/rkc_benchmark.png')


if __name__ == '__main__':
    main()
