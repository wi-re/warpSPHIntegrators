"""Generate linear-oscillator stability plots for Newmark and Verlet-family methods.

Two figures:

- ``images/oscillator_stability_newmark_verlet.png``: spectral radius vs ``h*omega``
  for the undamped oscillator (the Phase 6 results).
- ``images/oscillator_stability_damped_newmark_verlet.png``: the damping-ratio
  picture -- ``log10(rho)`` over the ``(h*omega, zeta)`` plane, with the
  ``rho = 1`` boundary contoured. All parameters are fixed below (``h*omega`` in
  [0, 8], ``zeta`` in [0, 1]) and stated in the figure caption.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators.stability import oscillator_spectral_radius


def _damped_rho_grid(method: str, h_omega: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Vectorized spectral radius of the damped-oscillator amplification matrix.

    Same closed forms as ``stability.damped_oscillator_amplification_matrix``, but
    evaluated on a meshgrid with the 2x2 eigenvalue formula
    (``rho = |tr|/2 + sqrt(disc)`` for real eigenvalues, ``sqrt(det)`` for a complex
    pair) instead of a per-point ``np.linalg.eigvals`` call.
    """
    Q, Z = np.meshgrid(h_omega, zeta, indexing='xy')
    Q2 = Q ** 2
    C = 2.0 * Z * Q
    if method in ('leapfrog', 'velocity_verlet'):
        a00 = 1.0 - 0.5 * Q2
        a01 = 1.0 - 0.5 * C
        if method == 'leapfrog':
            a10 = -0.5 * Q2 * (1.0 + a00)
            a11 = 1.0 - C - 0.5 * Q2 + 0.25 * C * Q2
        else:
            a10 = -0.5 * Q2 * (1.0 + a00 - 0.5 * C)
            a11 = (1.0 - 0.5 * C) * (1.0 - 0.5 * C - 0.5 * Q2)
    elif method == 'symplectic_euler':
        a00 = 1.0 - 0.25 * Q2 * (2.0 - C)
        a01 = 1.0 - 0.25 * Q2 - 0.5 * C + 0.25 * C ** 2
        a10 = 0.5 * Q2 * (C - 2.0)
        a11 = 1.0 - 0.5 * Q2 - C + 0.5 * C ** 2
    elif method in ('newmark_average_acceleration', 'newmark_linear_acceleration'):
        beta = 0.25 if method == 'newmark_average_acceleration' else 1.0 / 6.0
        gamma = 0.5
        l00, l01 = 1.0 + beta * Q2, beta * C
        l10, l11 = gamma * Q2, 1.0 + gamma * C
        r00, r01 = 1.0 - (0.5 - beta) * Q2, 1.0 - (0.5 - beta) * C
        r10, r11 = -(1.0 - gamma) * Q2, 1.0 - (1.0 - gamma) * C
        det_l = l00 * l11 - l01 * l10
        # (L^{-1} R) with the 2x2 inverse, elementwise
        a00 = (l11 * r00 - l01 * r10) / det_l
        a01 = (l11 * r01 - l01 * r11) / det_l
        a10 = (l00 * r10 - l10 * r00) / det_l
        a11 = (l00 * r11 - l10 * r01) / det_l
    else:
        raise ValueError(f'Unknown damped oscillator stability method {method!r}')

    trace = a00 + a11
    det = a00 * a11 - a01 * a10
    disc = (trace / 2.0) ** 2 - det
    real = disc >= 0.0
    rho = np.empty_like(disc)
    rho[real] = np.abs(trace[real]) / 2.0 + np.sqrt(disc[real])
    rho[~real] = np.sqrt(np.clip(det[~real], 0.0, None))
    return rho


def main():
    os.makedirs('images', exist_ok=True)
    h_omega = np.linspace(0.0, 5.0, 1001)
    methods = {
        'Leap Frog': ('leapfrog', '#3478a6'),
        'Velocity Verlet': ('velocity_verlet', '#3478a6'),
        'Symplectic Euler': ('symplectic_euler', '#758c37'),
        'Newmark average acceleration': ('newmark_average_acceleration', '#9a5b36'),
        'Newmark linear acceleration': ('newmark_linear_acceleration', '#9a5b36'),
    }

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for label, (method, color) in methods.items():
        radii = [oscillator_spectral_radius(method, value) for value in h_omega]
        linestyle = '--' if label in ('Velocity Verlet', 'Newmark linear acceleration') else '-'
        axis.plot(h_omega, radii, label=label, color=color, linestyle=linestyle)
    axis.axhline(1.0, color='black', linewidth=0.8)
    axis.axvline(2.0, color='gray', linewidth=0.8, linestyle=':')
    axis.axvline(np.sqrt(12.0), color='gray', linewidth=0.8, linestyle=':')
    axis.set_ylim(0.0, 3.0)
    axis.set_xlabel('h omega')
    axis.set_ylabel('Spectral radius of amplification matrix')
    axis.set_title('Undamped oscillator stability: Newmark and Verlet-family methods')
    axis.grid()
    axis.legend()
    fig.tight_layout()
    fig.savefig('images/oscillator_stability_newmark_verlet.png', dpi=200)
    plt.close(fig)


def damped_figure():
    """Damping-ratio stability picture: log10(rho) over (h*omega, zeta) for the five
    methods, with the rho = 1 boundary contoured. Shows where damping shrinks the
    stable region (leapfrog's explicit old-velocity term), where it extends it
    (velocity Verlet's stable window at h*omega = 2.5), and how Newmark's neutral
    undamped line becomes asymptotically stable for any zeta > 0."""
    h_omega = np.linspace(0.0, 8.0, 401)
    zeta = np.linspace(0.0, 1.0, 201)
    methods = {
        'Leap Frog (old-velocity damping)': 'leapfrog',
        'Velocity Verlet (half-step damping)': 'velocity_verlet',
        'Symplectic Euler (KDK damping)': 'symplectic_euler',
        'Newmark average acceleration (beta=1/4)': 'newmark_average_acceleration',
        'Newmark linear acceleration (beta=1/6)': 'newmark_linear_acceleration',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (label, method) in zip(axes.flat, methods.items()):
        rho = _damped_rho_grid(method, h_omega, zeta)
        log_rho = np.log10(np.clip(rho, 1e-3, None))
        pcol = axis.pcolormesh(h_omega, zeta, log_rho, shading='auto',
                               cmap='RdYlGn_r', vmin=-1.5, vmax=1.5)
        axis.contour(h_omega, zeta, rho, levels=[1.0], colors='black', linewidths=1.5)
        axis.set_xlabel('h omega')
        axis.set_ylabel('zeta')
        axis.set_title(label)
        axis.set_xlim(0.0, 8.0)
        axis.set_ylim(0.0, 1.0)
        fig.colorbar(pcol, ax=axis, label='log10(rho)')
    axes.flat[-1].axis('off')
    fig.suptitle(
        'Damped oscillator stability, x\'\' + 2 zeta omega x\' + omega^2 x = 0: '
        'rho(h omega, zeta) over h omega in [0, 8], zeta in [0, 1] '
        '(black contour: rho = 1)',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('images/oscillator_stability_damped_newmark_verlet.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
    damped_figure()