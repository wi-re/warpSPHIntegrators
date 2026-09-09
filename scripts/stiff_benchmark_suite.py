"""Phase 8: broadened nonlinear/stiff benchmark suite -> images/stiff_benchmark_suite.png.

A 2x3 figure comparing accuracy and per-step nonlinear-solve cost on four stiff
benchmarks. Every parameter is fixed here (deterministic, no adaptive steps):

1. Prothero-Robinson, x' = -100(x - tanh t) + sech^2(t), T = 1: error vs dt for
   BE (1st order), BDF2 (2nd order) and DP5 (5th order). DP5's stability
   boundary on the negative real axis is |z| ~= 3.3, so its curve ends at
   dt = 0.02 (z = 2); the dt = 0.05 point (z = 5) diverges and is marked 'x'.
2. Semi-discrete diffusion, n = 64, D = 1, T = 0.1, initial data on Laplacian
   eigenvectors 1 and 5: error vs dt for BE and BDF2. Only the excited modes
   matter: the fastest (mode 5) sits at z ~= 1.23 at the largest dt, inside
   BDF2's A-stable interval (boundary 2.41).
3. Van der Pol, mu = 10, T = 50: max |x_coarse(t) - x_ref(t)| at the coarse
   output times, x_ref = ESDIRK6 at dt = 0.01 (5000 steps); ESDIRK6 at
   dt in {0.05, 0.1, 0.2, 0.4}; DP5 at dt = 0.5 diverges within a few steps
   (marked 'x').
4. Robertson kinetics, T = 10.01 (10 x dt = 0.001 bootstrap steps to cross the
   initial quasi-steady layer, then the main dt): BE error vs dt at
   dt in {1.0, 0.25, 0.05} against x_ref = ESDIRK6 at dt = 0.002.
5-6. Per-step cost on the PR problem (rate = 100, dt = 0.01, 100 steps):
   mean RHS evaluations and mean GMRES iterations per step for BE, BDF2,
   TR-BDF2 and ESDIRK6.

Implicit solves use the registered JFNK defaults (finite-difference
Jacobian-vector products, GMRES tolerance 1e-8; the DIRK driver's Newton
residual tolerance is 1e-3).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing


def _run(scheme_name, problem, dt, n_steps, history_len=0, record=True):
    """Fixed-step run; returns (x at every output time, per-step solver diagnostics)."""
    scheme = getIntegrator(scheme_name)
    system = problem.initial()
    history = StepHistory(maxlen=history_len) if history_len else None
    xs, diags = [], []
    for _ in range(n_steps):
        result = scheme(system, dt=dt, f=problem.rhs, history=history)
        system, history = result.state, result.history or history
        if record:
            xs.append(get_reference_state(system).x.detach().cpu().numpy().copy())
        diag = result.solver_diagnostics
        if diag is None:
            continue
        for d in (diag if isinstance(diag, list) else [diag]):
            if d is not None:
                diags.append(d)
    return xs, diags


def _pr_sweeps():
    """Panel 1: PR rate=100, T=1, dt in {0.0025, 0.005, 0.01, 0.02} + DP5 wall at 0.05."""
    dts = (0.0025, 0.005, 0.01, 0.02)
    exact = np.tanh(1.0)
    curves = {}
    for name, history_len in (('Backward Euler (implicit)', 0), ('BDF2', 1), ('Dormand-Prince 5(4)', 0)):
        errors = []
        for dt in dts:
            xs, _ = _run(name, testing.stiff_relaxation_problem(rate=100.0, forcing='tanh'),
                         dt, int(round(1.0 / dt)), history_len=history_len)
            errors.append(abs(float(xs[-1][0]) - exact))
        curves[name] = (dts, errors)
    # the explicit wall: z = 100*0.05 = 5, outside DP5's |R| boundary (~3.3)
    xs, _ = _run('Dormand-Prince 5(4)', testing.stiff_relaxation_problem(rate=100.0, forcing='tanh'),
                 0.05, 20)
    wall_error = abs(float(xs[-1][0]) - exact)
    return curves, (0.05, wall_error)


def _diffusion_sweeps():
    """Panel 2: diffusion n=64, T=0.1, dt in {0.005, 0.0025, 0.00125, 0.000625}."""
    dts = (0.005, 0.0025, 0.00125, 0.000625)
    curves = {}
    for name, history_len in (('Backward Euler (implicit)', 0), ('BDF2', 1)):
        errors = []
        for dt in dts:
            problem = testing.diffusion_problem(n=64)
            xs, _ = _run(name, problem, dt, int(round(0.1 / dt)), history_len=history_len)
            exact, _ = problem.exact(0.1)
            errors.append(float(np.max(np.abs(np.asarray(xs[-1]) - np.asarray(exact)))))
        curves[name] = (dts, errors)
    return curves


def _vdp_sweeps():
    """Panel 3: VdP mu=10, T=50; ref ESDIRK6 dt=0.01; coarse ESDIRK6 dt in
    {0.05, 0.1, 0.2, 0.4}; DP5 wall at dt=0.5."""
    problem = testing.van_der_pol_problem(mu=10.0)
    ref_xs, _ = _run('ESDIRK4(3)6L[2]SA', problem, 0.01, 5000)
    ref = np.asarray([x[0] for x in ref_xs])
    dts = (0.05, 0.1, 0.2, 0.4)
    errors, diverged = [], []
    for dt in dts:
        xs, _ = _run('ESDIRK4(3)6L[2]SA', problem, dt, int(round(50.0 / dt)))
        x = np.asarray([v[0] for v in xs])
        idx = np.arange(len(x)) * int(round(dt / 0.01))
        errors.append(float(np.nanmax(np.abs(x - ref[idx]))))
        diverged.append((not bool(np.isfinite(x).all()))
                        or float(np.max(np.abs(x[np.isfinite(x)]))) > 10.0)
    # DP5: dt=0.5, fast eigenvalue ~ mu*(1+x^2) = O(10-20), far outside its stability
    # region; the orbit overflows to NaN within 20 steps, so report the last finite |x|
    xs, _ = _run('Dormand-Prince 5(4)', problem, 0.5, 20)
    x_wall = np.abs(np.asarray([v[0] for v in xs]))
    x_wall = x_wall[np.isfinite(x_wall)]
    wall = float(np.max(x_wall)) if x_wall.size else float('inf')
    return dts, errors, diverged, (0.5, wall)


def _robertson_sweeps():
    """Panel 4: Robertson, bootstrap 10 x 0.001 then dt in {1.0, 0.25, 0.05} to
    T = 10.01; ref ESDIRK6 dt=0.002 (5000 main steps)."""
    def run_bootstrapped(scheme_name, dt, history_len=0):
        problem = testing.robertson_problem()
        scheme = getIntegrator(scheme_name)
        system = problem.initial()
        xs = []
        for _ in range(10):
            system = scheme(system, dt=0.001, f=problem.rhs).state
        n = int(round((10.01 - 0.01) / dt))
        for _ in range(n):
            system = scheme(system, dt=dt, f=problem.rhs).state
            xs.append(get_reference_state(system).x.detach().cpu().numpy().copy())
        return np.asarray(xs)

    ref = run_bootstrapped('ESDIRK4(3)6L[2]SA', 0.002)
    dts = (1.0, 0.25, 0.05)
    errors = []
    for dt in dts:
        x = run_bootstrapped('Backward Euler (implicit)', dt)
        idx = np.arange(len(x)) * int(round(dt / 0.002))
        errors.append(float(np.max(np.abs(x - ref[idx]))))
    return dts, errors


def _cost_panel():
    """Panels 5-6: per-step cost on PR rate=100, dt=0.01, 100 steps."""
    problem = testing.stiff_relaxation_problem(rate=100.0, forcing='tanh')
    names = ('Backward Euler (implicit)', 'BDF2', 'TR-BDF2', 'ESDIRK4(3)6L[2]SA')
    rhs_per_step, gmres_per_step = [], []
    for name in names:
        history_len = 1 if name == 'BDF2' else 0
        _, diags = _run(name, problem, 0.01, 100, history_len=history_len)
        n = max(1, len([d for d in diags]))
        rhs_per_step.append(sum(d.rhs_evaluations for d in diags) / 100.0)
        gmres_per_step.append(sum(d.gmres_iterations for d in diags) / 100.0)
    return names, rhs_per_step, gmres_per_step


def main():
    os.makedirs('images', exist_ok=True)
    pr_curves, pr_wall = _pr_sweeps()
    diff_curves = _diffusion_sweeps()
    vdp_dts, vdp_errors, vdp_diverged, vdp_wall = _vdp_sweeps()
    rob_dts, rob_errors = _robertson_sweeps()
    cost_names, cost_rhs, cost_gmres = _cost_panel()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1: PR (the DP5 wall marker sits at the top of the axis: the actual
    # error at dt = 0.05 is ~1e17, which would squash the converging curves)
    axis = axes[0, 0]
    for name, (dts, errors) in pr_curves.items():
        axis.loglog(dts, errors, 'o-', label=name.replace(' (implicit)', ' (BE)').replace('Dormand-Prince 5(4)', 'DP5'))
    axis.set_ylim(bottom=1e-12)
    axis.plot([pr_wall[0]], [0.9 * axis.get_ylim()[1]], 'rx', markersize=10, mew=2,
              label=f'DP5 diverges (z = 5, err = {pr_wall[1]:.1e})')
    axis.set_xlabel('dt')
    axis.set_ylabel('|x(1) - tanh(1)|')
    axis.set_title('Prothero-Robinson, rate = 100, T = 1')
    axis.grid(which='both', alpha=0.4)
    axis.legend(fontsize=8)

    # Panel 2: diffusion
    axis = axes[0, 1]
    for name, (dts, errors) in diff_curves.items():
        axis.loglog(dts, errors, 'o-', label=name.replace(' (implicit)', ' (BE)'))
    axis.set_xlabel('dt')
    axis.set_ylabel('max |u - u_exact|')
    axis.set_title('Semi-discrete diffusion, n = 64, D = 1, T = 0.1')
    axis.grid(which='both', alpha=0.4)
    axis.legend(fontsize=8)

    # Panel 3: van der Pol (diverged runs are marked at the top of the axis)
    axis = axes[0, 2]
    stable_dts = [d for d, div in zip(vdp_dts, vdp_diverged) if not div]
    stable_errs = [e for e, div in zip(vdp_errors, vdp_diverged) if not div]
    axis.loglog(stable_dts, stable_errs, 'o-', label='ESDIRK6')
    axis.set_ylim(bottom=1e-6)
    top = 0.9 * axis.get_ylim()[1]
    for d in (d for d, div in zip(vdp_dts, vdp_diverged) if div):
        axis.plot([d], [top], 'rx', markersize=10, mew=2)
    axis.plot([vdp_wall[0]], [top], 'rx', markersize=10, mew=2, label='DP5 diverges (dt = 0.5)')
    axis.set_xlabel('dt')
    axis.set_ylabel('max |x - x_ref|')
    axis.set_title('Van der Pol, mu = 10, T = 50 (ref: ESDIRK6 dt = 0.01)')
    axis.grid(which='both', alpha=0.4)
    axis.legend(fontsize=8)

    # Panel 4: Robertson
    axis = axes[1, 0]
    axis.loglog(rob_dts, rob_errors, 'o-', label='Backward Euler')
    axis.set_xlabel('dt (after 10 x dt = 0.001 bootstrap)')
    axis.set_ylabel('max |y - y_ref|')
    axis.set_title('Robertson kinetics, T = 10.01 (ref: ESDIRK6 dt = 0.002)')
    axis.grid(which='both', alpha=0.4)
    axis.legend(fontsize=8)

    # Panel 5: cost -- RHS evaluations per step
    axis = axes[1, 1]
    labels = [n.replace('Backward Euler (implicit)', 'BE').replace('ESDIRK4(3)6L[2]SA', 'ESDIRK6') for n in cost_names]
    axis.bar(labels, cost_rhs, color='#3478a6')
    for x, y in zip(range(len(cost_rhs)), cost_rhs):
        axis.text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=9)
    axis.set_ylabel('RHS evaluations / step')
    axis.set_title('PR rate = 100, dt = 0.01, 100 steps (JFNK defaults)')
    axis.grid(axis='y', alpha=0.4)

    # Panel 6: cost -- GMRES iterations per step
    axis = axes[1, 2]
    axis.bar(labels, cost_gmres, color='#758c37')
    for x, y in zip(range(len(cost_gmres)), cost_gmres):
        axis.text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=9)
    axis.set_ylabel('GMRES iterations / step')
    axis.set_title('PR rate = 100, dt = 0.01, 100 steps (JFNK defaults)')
    axis.grid(axis='y', alpha=0.4)

    fig.suptitle('Phase 8 stiff benchmark suite: accuracy and per-step nonlinear-solve cost', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('images/stiff_benchmark_suite.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
