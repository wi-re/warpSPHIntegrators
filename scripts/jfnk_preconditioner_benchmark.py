"""Phase 2: JFNK preconditioner benchmark (IMPLICIT_ROADMAP.md Phase 2).

Solves a variable-coefficient stiff relaxation of growing size through
``JFNKSolver``, with and without the per-DOF diagonal preconditioner, and
records the Krylov iterations and residual evaluations the solver reports.

The stage operator is the matrix-free  A = I + dt*(diag(K) + eps*L)  with a
per-DOF rate K in [1, kmax] and the 1D Laplacian L -- the stand-in for a stiff
SPH stage whose stiffness varies by particle, where a diagonal is a strong
approximation to A^{-1}. The problem is linear, so each JFNK step is one Newton
correction and the reported ``gmres_iterations`` is exactly the cost of the
stage linear solve.

Saves ``images/jfnk_preconditioner_benchmark.png``.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from warpSPHIntegrators import JFNKSolver, get_reference_state
from warpSPHIntegrators.testing import ParticleState, ParticleSystem

DTYPE = torch.float64
SIZES = [64, 128, 256, 512, 1024]


def build_problem(n: int, dt: float = 10.0, eps: float = 0.1, kmax: float = 100.0):
    """Length-``n`` stiff relaxation on the ``x`` block (``u``/``e`` carried):
    backward-Euler fixed point of ``x' = -K x + eps L x``. Returns the initial
    system, the fixed-point step, and the per-DOF diagonal inverse for the full
    flattened state."""
    K = torch.linspace(1.0, kmax, n, dtype=DTYPE)
    x0 = torch.rand(n, dtype=DTYPE)

    def laplacian(x):
        Lx = torch.zeros_like(x)
        Lx[0] = 2 * x[0] - x[1]
        Lx[-1] = 2 * x[-1] - x[-2]
        Lx[1:-1] = -x[:-2] + 2 * x[1:-1] - x[2:]
        return Lx

    def step(Y):
        s_in = get_reference_state(Y)
        out = Y.initializeNewState()
        s_out = get_reference_state(out)
        s_out.x = x0 + dt * (-K * s_in.x + eps * laplacian(s_in.x))
        s_out.u, s_out.e = s_in.u.clone(), s_in.e.clone()
        return out

    system = ParticleSystem(state=ParticleState(
        x=x0.clone(), u=torch.zeros(n, dtype=DTYPE), e=torch.zeros(n, dtype=DTYPE),
        m=torch.ones(n, dtype=DTYPE)))
    inv_full = torch.cat([1.0 / (1.0 + dt * K),
                          torch.ones(n, dtype=DTYPE),
                          torch.ones(n, dtype=DTYPE)])
    return system, step, inv_full


def main():
    os.makedirs('images', exist_ok=True)
    ref_iters, prec_iters = [], []
    ref_total, prec_total = [], []
    for n in SIZES:
        system, step, inv_full = build_problem(n)
        ref = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10).solve(step, system)
        prec = JFNKSolver(matvec='fd', tol=1e-10, max_iterations=10,
                          preconditioner=lambda v, state, context: v * context['inv'],
                          preconditioning='right',
                          preconditioner_context={'inv': inv_full}).solve(step, system)
        assert ref.converged and prec.converged, f'n={n}: convergence failure'
        ref_iters.append(ref.diagnostics.gmres_iterations)
        prec_iters.append(prec.diagnostics.gmres_iterations)
        # Documented cost model (NOTES.md S3.4): total stage-map evaluations =
        # rhs_evaluations + gmres_iterations (one FD matvec is one step eval).
        # This linear problem needs one Newton correction, so rhs_evaluations is
        # a constant 3 and the Krylov count is essentially the whole cost.
        ref_total.append(ref.diagnostics.rhs_evaluations + ref.diagnostics.gmres_iterations)
        prec_total.append(prec.diagnostics.rhs_evaluations + prec.diagnostics.gmres_iterations)
        print(f'n={n:5d}  gmres_iters: no prec {ref_iters[-1]:4d} -> diagonal {prec_iters[-1]:3d}   '
              f'total stage-map evals: {ref_total[-1]:5d} -> {prec_total[-1]:4d} '
              f'({ref_total[-1] / prec_total[-1]:.1f}x)')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(SIZES, ref_iters, 'o-', color='#9a5b36', label='no preconditioner')
    ax1.plot(SIZES, prec_iters, 's-', color='#3478a6', label='diagonal (right)')
    ax1.set_xlabel('problem size n')
    ax1.set_ylabel('GMRES iterations')
    ax1.set_title('JFNK stage solve: Krylov iterations vs size')
    ax1.grid()
    ax1.legend()
    speedup = ref_iters[-1] / prec_iters[-1]
    ax1.annotate(f'{speedup:.1f}x at n={SIZES[-1]}',
                 xy=(SIZES[-1], ref_iters[-1]),
                 xytext=(SIZES[-1] * 0.55, ref_iters[-1] * 0.9),
                 fontsize=9)

    ax2.plot(SIZES, ref_total, 'o-', color='#9a5b36', label='no preconditioner')
    ax2.plot(SIZES, prec_total, 's-', color='#3478a6', label='diagonal (right)')
    ax2.set_xlabel('problem size n')
    ax2.set_ylabel('stage-map evaluations')
    ax2.set_title('JFNK stage solve: total cost vs size\n(rhs_evaluations + gmres_iterations)')
    ax2.grid()
    ax2.legend()

    fig.tight_layout()
    fig.savefig('images/jfnk_preconditioner_benchmark.png', dpi=200)
    print('saved images/jfnk_preconditioner_benchmark.png')


if __name__ == '__main__':
    main()
