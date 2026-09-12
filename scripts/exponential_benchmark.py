"""Phase 7: ETD2RK (exponential integrator) vs the implicit / IMEX / RW baselines
on the semi-discrete viscous Burgers benchmark -> images/exponential_benchmark.png.

ETD2RK is the first of Phase 7's exponential-integrator sub-phase: it integrates
the semilinear split ``f = L·y + N`` with the linear part ``L`` carried **exactly**
through ``exp(hL)`` and the entire ``phi`` functions -- applied **matrix-free** as
``phi_k(hL)v`` via a Krylov (Arnoldi) approximation, consuming the Phase 14
``linear`` accessor -- and quadratures the nonlinear remainder ``N`` (two
evaluations per step).

The roadmap gate for the exponential family is "**beat the same-order cost to a
fixed error**". For ETD2RK (order 2) only the **order** part is assessed here;
the **cost** part is deliberately deferred to the order-3 method (exprb32) before
the method set expands, because ETD2RK's per-step cost is dominated by three
Krylov builds (``3m`` ``L``-matvecs) and the work-unit metric below is *not* a
fair FLOP comparison (each Krylov ``L``-matvec is a cheap ``nu·u_xx`` application,
whereas each JFNK ``rhs_evaluation`` is a full ``N + L`` evaluation). See the
prose under Panel 2.

Two panels:

  1. **Convergence** (Gaussian IC): endpoint relative ``L2`` error at ``T = 0.4``
     against the fine-``dt`` RK4 reference, on ``dt`` that divide ``T`` exactly.
     ETD2RK sits on the order-2 line.
  2. **Cost** (sine IC, ``dt = 0.02``, 20 steps): total ``rhs_evaluations`` +
     ``gmres_iterations`` (the Phase 1 ``SolveDiagnostics``, summed over every
     stage). For ETD2RK the ``gmres_iterations`` field carries the total
     ``L``-matvec (Krylov) count. The order-2 bars (BDF2, TR-BDF2) are shown for
     reference; the cost gate is assessed at exprb32, not here.

Run:  OMP_NUM_THREADS=4 python scripts/exponential_benchmark.py
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from warpSPHIntegrators import IMEXRHS, getIntegrator, get_reference_state, resolve
from warpSPHIntegrators.history import StepHistory
from warpSPHIntegrators.testing import ParticleState, ParticleSystem, viscous_burgers_problem

N, L, NU = 64, 1.0, 0.01
T = 0.4
DT_STIFF = 0.02
h = L / N
X_GRID = torch.tensor([i * h for i in range(N)], dtype=torch.float64)


def make_system(u0):
    """A fresh Burgers system carrying ``u0`` (a length-N float64 tensor)."""
    return ParticleSystem(state=ParticleState(
        x=u0, u=torch.zeros(N, dtype=torch.float64),
        e=torch.zeros(N, dtype=torch.float64),
        m=torch.ones(N, dtype=torch.float64)), t=0.0)


def run_cost(scheme_name, dt, T, f, system_factory, history=False, **scheme_kwargs):
    """One fixed-``dt`` run of a registered scheme; returns
    ``(times, U, rhs_evals, gmres_iters)`` with ``U`` of shape ``(nSteps+1, N)``.

    Mirrors the notebook's ``runTrajectory`` (cost = the Phase 1
    ``SolveDiagnostics`` summed over every stage solve).
    """
    scheme = getIntegrator(scheme_name)
    system = system_factory()
    step_history = StepHistory(maxlen=4) if history else None
    times, us = [0.0], [get_reference_state(system).x.clone()]
    rhs_evals, gmres_iters = 0, 0
    for _ in range(int(round(T / dt))):
        kwargs = dict(scheme_kwargs)
        if step_history is not None:
            kwargs['history'] = step_history
        result = scheme(system, dt=dt, f=f, **kwargs)
        system = result.state
        times.append(system.t)
        us.append(get_reference_state(system).x.clone())
        if step_history is not None and result.history is not None:
            step_history = result.history
        diag = result.solver_diagnostics
        if diag is not None:
            for d in (diag if isinstance(diag, (list, tuple)) else [diag]):
                if d is not None:
                    rhs_evals += d.rhs_evaluations
                    gmres_iters += d.gmres_iterations
    return np.asarray(times, dtype=float), np.stack(us), rhs_evals, gmres_iters


def main():
    os.makedirs('images', exist_ok=True)
    problem = viscous_burgers_problem(n=N, L=L, nu=NU)
    make_gauss = lambda: problem.initial()
    sine0 = -torch.sin(2 * math.pi * X_GRID)
    make_sine = lambda: make_system(sine0)
    parts = resolve(problem.rhs, scheme_name='ETD2RK benchmark')
    imex_rhs = IMEXRHS(explicit=parts.nonlinear, implicit=parts.linear)

    # References: fine-dt RK4 (the notebook's canonical accuracy reference).
    _, u_gref, _, _ = run_cost('RK4', 2e-3, T, problem.rhs, make_gauss)
    _, u_sref, _, _ = run_cost('RK4', 2e-3, T, problem.rhs, make_sine)
    print(f'reference: RK4 @ dt=2e-3, {int(round(T / 2e-3))} steps '
          f'(Gaussian + sine IC)')

    # --------------------------------------------------------------------- #
    # Panel 1: convergence (Gaussian IC)                                     #
    # --------------------------------------------------------------------- #
    conv = [
        ('RK4', 'RK4', [0.01, 0.005, 0.0025], problem.rhs, False, {}),
        ('Backward Euler', 'Backward Euler (implicit)', [0.04, 0.02, 0.01], problem.rhs, False, {}),
        ('TR-BDF2', 'TR-BDF2', [0.04, 0.02, 0.01], problem.rhs, False, {}),
        ('ESDIRK3(2)4L[2]SA', 'ESDIRK3(2)4L[2]SA', [0.04, 0.02, 0.01], problem.rhs, False, {}),
        ('ESDIRK4(3)6L[2]SA', 'ESDIRK4(3)6L[2]SA', [0.04, 0.02, 0.01], problem.rhs, False, {}),
        ('BDF2', 'BDF2', [0.04, 0.02, 0.01], problem.rhs, True, {}),
        ('ARK3(2)4L[2]SA', 'ARK3(2)4L[2]SA', [0.04, 0.02, 0.01], imex_rhs, False, {}),
        ('ROS3P (jvp)', 'ROS3P', [0.08, 0.04, 0.02, 0.01, 0.005], problem.rhs, False, {'w': 'jvp'}),
        ('ETD2RK', 'ETD2RK', [0.08, 0.04, 0.02, 0.01, 0.005], problem.rhs, False, {}),
    ]
    print(f"\nConvergence, Gaussian IC, relative L2 error at T = {T} vs RK4 (dt = 2e-3):")
    print(f"  {'scheme':18s} {'errors':>30s}   measured orders")
    conv_data = {}
    for label, name, dts, f, hist, kw in conv:
        errs = []
        for dt in dts:
            _, U, _, _ = run_cost(name, dt, T, f, make_gauss, history=hist, **kw)
            errs.append(float(np.linalg.norm(U[-1] - u_gref[-1]) / np.linalg.norm(u_gref[-1])))
        orders = [math.log(errs[i] / errs[i + 1]) / math.log(dts[i] / dts[i + 1])
                  for i in range(len(errs) - 1)]
        conv_data[label] = (dts, errs)
        print(f"  {label:18s} {[f'{e:.2e}' for e in errs]}   {[f'{o:.2f}' for o in orders]}")

    # --------------------------------------------------------------------- #
    # Panel 2: cost (sine IC, dt = 0.02, 20 steps)                           #
    # --------------------------------------------------------------------- #
    cost = [
        ('explicit', 'Forward Euler', 'Forward Euler', 0.005, problem.rhs, False, {}),
        ('explicit', 'SSP RK3 / TVD RK3', 'TVD RK3', 0.005, problem.rhs, False, {}),
        ('explicit', 'RK4', 'RK4', 0.005, problem.rhs, False, {}),
        ('implicit', 'Backward Euler', 'Backward Euler (implicit)', DT_STIFF, problem.rhs, False, {}),
        ('implicit', 'TR-BDF2', 'TR-BDF2', DT_STIFF, problem.rhs, False, {}),
        ('implicit', 'ESDIRK3(2)4L[2]SA', 'ESDIRK3(2)4L[2]SA', DT_STIFF, problem.rhs, False, {}),
        ('implicit', 'ESDIRK4(3)6L[2]SA', 'ESDIRK4(3)6L[2]SA', DT_STIFF, problem.rhs, False, {}),
        ('implicit', 'BDF2', 'BDF2', DT_STIFF, problem.rhs, True, {}),
        ('IMEX', 'IMEX Euler', 'IMEX Euler', DT_STIFF, imex_rhs, False, {}),
        ('IMEX', 'ARK3(2)4L[2]SA', 'ARK3(2)4L[2]SA', DT_STIFF, imex_rhs, False, {}),
        ('IMEX', 'ARK4(3)6L[2]SA', 'ARK4(3)6L[2]SA', DT_STIFF, imex_rhs, False, {}),
        ('Rosenbrock-W', 'ROS3P (jvp)', 'ROS3P', DT_STIFF, problem.rhs, False, {'w': 'jvp'}),
        ('exponential', 'ETD2RK', 'ETD2RK', DT_STIFF, problem.rhs, False, {}),
    ]
    print(f"\nCost, sine IC, T = {T} (RHS evals + L-matvec Krylov iters, summed):")
    print(f"  {'family':13s} {'scheme':20s} {'dt':>6s} {'steps':>5s} "
          f"{'RHS evals':>9s} {'L-matvecs':>11s} {'total':>7s} {'rel L2 err':>11s}")
    cost_data = []
    for family, label, name, dt, f, hist, kw in cost:
        times, U, r, g = run_cost(name, dt, T, f, make_sine, history=hist, **kw)
        err = float(np.linalg.norm(U[-1] - u_sref[-1]) / np.linalg.norm(u_sref[-1]))
        cost_data.append((family, label, dt, len(times) - 1, r, g, err))
        print(f"  {family:13s} {label:20s} {dt:6g} {len(times) - 1:5d} "
              f"{r:9d} {g:11d} {r + g:7d} {err:11.3e}")

    # --------------------------------------------------------------------- #
    # Figure                                                                  #
    # --------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15, 5.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Panel 1: convergence.
    for name, (dts, errs) in conv_data.items():
        ax1.loglog(dts, errs, 'o-', label=name)
    x0 = 0.08
    for p in (1, 2, 3, 4):
        xs = np.array([x0 / 32, x0])
        ys = 2e-3 * (xs / x0) ** p
        ax1.loglog(xs, ys, '--', c='0.7')
        ax1.text(x0 * 1.12, ys[1] * 1.35, f'order {p}', fontsize=8, c='0.5')
    ax1.set_xlabel('dt')
    ax1.set_ylabel('relative L2 error at T = 0.4 (Gaussian IC)')
    ax1.set_title('Convergence vs the RK4 (dt = 2e-3) reference\n(ETD2RK sits on the order-2 line)')
    ax1.legend(fontsize=7)
    ax1.grid(True, which='both', alpha=0.3)

    # Panel 2: cost (bar chart of total RHS evals + L-matvec Krylov iters).
    labels = [c[1] for c in cost_data]
    rhs = [c[4] for c in cost_data]
    krylov = [c[5] for c in cost_data]
    fam_colors = {'explicit': '#9aa5b1', 'implicit': '#3478a6',
                  'IMEX': '#758c37', 'Rosenbrock-W': '#c44e52',
                  'exponential': '#b8860b'}
    x = np.arange(len(labels))
    width = 0.4
    ax2.bar(x - width / 2, rhs, width, color=[fam_colors[c[0]] for c in cost_data],
            edgecolor='white', linewidth=0.5, label='RHS (N) evals')
    ax2.bar(x + width / 2, krylov, width, color='0.75', edgecolor='white',
            linewidth=0.5, label='L-matvec / GMRES iters')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
    ax2.set_ylabel('count (log scale)')
    ax2.set_title(f'Cost to T = 0.4, sine IC (stiff families at dt = {DT_STIFF:g};\n'
                  f'ETD2RK "iters" are Krylov L-matvecs, a cheaper op than a full N+L eval)')
    ax2.grid(axis='y', alpha=0.4, which='both')
    ax2.legend(fontsize=8)

    fig.suptitle('Phase 7: ETD2RK (exponential integrator) on viscous Burgers '
                 '(cost = SolveDiagnostics summed over every stage)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('images/exponential_benchmark.png', dpi=200)
    plt.close(fig)
    print('\nWrote images/exponential_benchmark.png')


if __name__ == '__main__':
    main()
