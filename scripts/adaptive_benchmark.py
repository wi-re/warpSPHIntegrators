"""Phase 11: adaptive step control and dense output ->
images/adaptive_benchmark.png.

The controller is a *helper, not a solver* (NOTES.md S3.17): the library exposes
``estimate_error_norm`` (the embedded-pair difference as a dimensionless number,
1.0 = at the tolerance) and ``propose_dt`` (the classic predictive controller,
``dt * clamp(safety * (target/error)**(1/order))``), and the caller drives the
accept/reject loop. ``run_adaptive`` below is that loop, the demo the Phase 11
contract leaves to the caller.

The gate problem is van der Pol at mu = 10, where the Phase 8 work measured a
hard explicit wall (fast eigenvalue ~ mu*(1 + x^2) = O(10-20)): the stiffness
varies along the trajectory (slow drift on the branches, fast jump at the fold),
so a fixed dt must be set by the stiffest instant while an adaptive run can use
large steps where the solution is smooth.

  1. **Fixed vs. adaptive at T = 5**: fixed-dt DP5 endpoint error on a dt ladder
     against a fine-dt reference (its own floor, measured at two resolutions),
     with the adaptive runs (rtol = 1e-4 / 1e-5 / 1e-6) marked by their endpoint
     error and accepted-step count.
  2. **The controller in flight**: the accepted step size and the error norm per
     accepted step over the adaptive run (rtol = 1e-5, dt0 = 0.1 -- deliberately
     far outside the explicit wall, so the run opens with rejections).
  3. **Dense output**: the quartic continuous extension of one DP5 step (Shampine
     1986) queried at theta = 0.4 on the linear oscillator, against the analytic
     solution: the local interpolation error is O(dt**5) -- a slope-5 line. The
     step endpoints are exact by construction (theta = 1 re-applies the
     propagated weights and is bit-for-bit the step's returned state).

Run:  OMP_NUM_THREADS=4 python scripts/adaptive_benchmark.py
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from warpSPHIntegrators import (
    dormand_prince_dense_output,
    estimate_error_norm,
    getIntegrator,
    get_reference_state,
    propose_dt,
    testing,
)

MU, T = 10.0, 5.0
DT0 = 0.1  # deliberately far outside the explicit wall (fast eigenvalue ~ O(10-20))


def estimate_order(scheme) -> int:
    """The power q the estimate scales as h^q: the scheme's order, except
    TR-BDF2, whose published SUNDIALS pair makes the estimate O(h**(order+1))."""
    return scheme.order + 1 if scheme.name == 'TR-BDF2' else scheme.order


def run_adaptive(scheme, problem, T, dt0, *, rtol=1e-5, atol=1e-8,
                 safety=0.9, growth_min=0.2, growth_max=5.0):
    """The caller-driven accept/reject loop Phase 11's helper-only contract
    leaves to the caller: step, measure, accept/reject, propose.

    Returns ``(final_state, accepted, rejected, trajectory)``; ``trajectory`` is
    one ``(t, dt, error_norm)`` per accepted step.
    """
    q = estimate_order(scheme)
    state = problem.initial()
    t = float(state.t)
    dt = dt0
    accepted = rejected = 0
    trajectory = []
    while t < T - 1e-12:
        step = min(dt, T - t)
        result = scheme(state, dt=step, f=problem.rhs)
        norm = estimate_error_norm(result, rtol=rtol, atol=atol)
        if norm <= 1.0:
            trajectory.append((t, step, norm))
            state = result.state
            t = float(state.t)
            accepted += 1
        else:
            rejected += 1
        proposed = propose_dt(norm, step, q, safety=safety,
                              growth_min=growth_min, growth_max=growth_max)
        if norm > 1.0:
            proposed = min(proposed, step)  # never grow on a rejection
        dt = min(proposed, T - t)
    return state, accepted, rejected, trajectory


def endpoint_error(problem, T, system, reference):
    s, r = get_reference_state(system), get_reference_state(reference)
    return max(abs(float(s.x[0]) - float(r.x[0])),
               abs(float(s.u[0]) - float(r.u[0])))


def fixed_run(scheme, problem, dt, T):
    system, t = problem.initial(), 0.0
    while t < T - 1e-12:
        system = scheme(system, dt=min(dt, T - t), f=problem.rhs).state
        t = float(system.t)
    return system


def main():
    prob = testing.van_der_pol_problem(mu=MU)
    scheme = getIntegrator('Dormand-Prince 5(4)')

    # ---- reference: fine fixed-dt DP5, floor at two resolutions ------------ #
    ref = fixed_run(scheme, prob, dt=5e-4, T=T)
    ref2 = fixed_run(scheme, prob, dt=2.5e-4, T=T)
    ref_floor = endpoint_error(prob, T, ref, ref2)
    print(f'reference: DP5 dt=5e-4 vs dt=2.5e-4 (T={T}): floor {ref_floor:.3e}')

    # ---- Panel 1: fixed dt ladder + adaptive runs --------------------------- #
    ladder = [0.016, 0.008, 0.004, 0.002, 0.001]
    fixed = []
    for dt in ladder:
        err = endpoint_error(prob, T, fixed_run(scheme, prob, dt, T), ref)
        if math.isfinite(err):
            fixed.append((dt, err))
            print(f'fixed dt={dt}: error {err:.3e} ({T / dt:.0f} steps)')
        else:
            print(f'fixed dt={dt}: diverged (outside DP5 stability interval)')

    adaptive = {}
    for rtol in (1e-4, 1e-5, 1e-6):
        state, acc, rej, _ = run_adaptive(scheme, prob, T, DT0, rtol=rtol, atol=1e-8)
        adaptive[rtol] = (endpoint_error(prob, T, state, ref), acc, rej)
        print(f'adaptive rtol={rtol:.0e}: error {adaptive[rtol][0]:.3e}, '
              f'{acc} accepted, {rej} rejected')

    # ---- Panel 2: controller in flight -------------------------------------- #
    _, acc2, rej2, traj = run_adaptive(scheme, prob, T, DT0, rtol=1e-5, atol=1e-8)
    ts = [t for t, _, _ in traj]
    dts = [d for _, d, _ in traj]
    norms = [n for _, _, n in traj]

    # ---- Panel 3: dense output on the linear oscillator --------------------- #
    osc = testing.PROBLEMS['oscillator']()
    dense_dts = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    theta = 0.4
    dense_errs = []
    for dt in dense_dts:
        state0 = osc.initial()
        result = scheme(state0, dt=dt, f=osc.rhs)
        y = dormand_prince_dense_output(state0, result.stages, dt, theta)
        ex, eu = osc.exact(theta * dt)
        sy, su = get_reference_state(y).x, get_reference_state(y).u
        dense_errs.append(max(abs(float(sy[0]) - ex[0]), abs(float(su[0]) - eu[0])))
        # endpoint sanity: theta=1 must be bit-for-bit the propagated state
        y1 = dormand_prince_dense_output(state0, result.stages, dt, 1.0)
        assert torch.equal(get_reference_state(y1).x, get_reference_state(result.state).x)
    rates = [math.log(a / b) / math.log(2) for a, b in zip(dense_errs, dense_errs[1:])]
    print(f'dense output interior rates at theta={theta}: '
          + '  '.join(f'{r:.3f}' for r in rates))

    # ---- figure -------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.loglog([T / dt for dt, _ in fixed], [e for _, e in fixed], 'o-',
              label='fixed dt (DP5)')
    for rtol, (err, acc, rej) in adaptive.items():
        ax.loglog([acc], [err], '*', ms=13,
                  label=f'adaptive rtol={rtol:.0e} ({acc} steps, {rej} rej)')
    ax.axhline(ref_floor, color='gray', ls='--', lw=1)
    ax.text(1.2, ref_floor * 1.5, f'reference floor {ref_floor:.0e}', fontsize=7,
            color='gray')
    ax.set_xlabel('accepted steps to T = 5')
    ax.set_ylabel('endpoint error (vs. fine-dt reference)')
    ax.set_title(f'van der Pol, mu = {MU:g}: error vs. step count')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.semilogy(range(len(dts)), dts, 'o-', ms=3, label='accepted dt')
    ax.axhline(2 / 20, color='gray', ls=':', lw=1)
    ax.text(0.02 * len(dts), 2 / 20, 'explicit wall ~ dt < 2/20', fontsize=7,
            va='bottom')
    ax.set_xlabel('accepted step')
    ax.set_ylabel('dt')
    ax2 = ax.twinx()
    ax2.semilogy(range(len(norms)), norms, 's-', ms=3, color='tab:red',
                 label='error norm')
    ax2.axhline(1.0, color='tab:red', ls=':', lw=1)
    ax2.text(0.02 * len(norms), 1.2, 'target = 1', fontsize=7, color='tab:red')
    ax2.set_ylabel('error norm (1 = at tolerance)', color='tab:red')
    ax.set_title(f'adaptive in flight: {acc2} accepted, {rej2} rejected (rtol = 1e-5)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.loglog(dense_dts, dense_errs, 'o-', label=f'DP5 dense output, theta = {theta}')
    ax.loglog(dense_dts, [dense_errs[-1] * (d / dense_dts[-1]) ** 5 for d in dense_dts],
              'k:', label='slope 5 (O(dt$^5$) locally)')
    ax.set_xlabel('step size dt (one step, query at theta*dt)')
    ax.set_ylabel('interpolation error vs. analytic')
    ax.set_title('dense output: quartic continuous extension (Shampine 1986)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'images',
                       'adaptive_benchmark.png')
    fig.savefig(out, dpi=140)
    print(f'wrote {os.path.abspath(out)}')


if __name__ == '__main__':
    main()
