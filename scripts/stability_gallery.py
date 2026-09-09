"""Generate Dahlquist stability-region figures for implemented tableau and BDF methods.

The regions apply to y' = lambda*y with z = dt*lambda. Newmark/Verlet use a
second-order oscillator stability analysis, and IMEX has a two-parameter split
region, so the additive (IMEX) methods get a dedicated two-parameter figure
(``imex_stability_slices``) rather than this scalar one-parameter diagnostic.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators.stability import bdf_is_stable, imex_stability_function, rk_is_stable
from warpSPHIntegrators.integration import IntegrationSchemes
from warpSPHIntegrators.ark import getARKTableau


def _imex_euler_stability(z_explicit: complex, z_implicit: complex) -> complex:
    """Forward-Euler-on-explicit + backward-Euler-on-implicit: R = (1+ze)/(1-zi)."""
    return (1.0 + z_explicit) / (1.0 - z_implicit)


def imex_stability_slices(grid_n: int = 201):
    """Two-parameter IMEX stability slices: stable region in the z_explicit plane with
    z_implicit fixed.

    Each row is an additive method (IMEX Euler and the two Kennedy-Carpenter ARK
    pairs); each column fixes the implicit part at an increasingly stiff value, so the
    progression shows how the explicit stability region changes as the implicit half
    damps harder. Column 0 (z_implicit = 0) is the pure-explicit component region; the
    rightmost column is the strong-damping (L) limit.
    """
    real = np.linspace(-6.0, 2.0, grid_n)
    imag = np.linspace(-5.0, 5.0, grid_n)
    ze_grid = real[None, :] + 1j * imag[:, None]

    methods = [
        ('IMEX Euler', None),
        ('ARK3(2)4L[2]SA', getARKTableau('ARK324L2SA')),
        ('ARK4(3)6L[2]SA', getARKTableau('ARK436L2SA')),
    ]
    implicit_slices = [0.0, -1.0, -5.0, -20.0, -100.0]

    rows, cols = len(methods), len(implicit_slices)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    for row, (label, tableau) in enumerate(methods):
        for col, zi in enumerate(implicit_slices):
            axis = axes[row, col]
            stable = np.empty(ze_grid.shape, dtype=bool)
            for index, ze in np.ndenumerate(ze_grid):
                if tableau is None:
                    r = _imex_euler_stability(ze, zi)
                else:
                    r = imex_stability_function(tableau, ze, zi)
                stable[index] = abs(r) <= 1.0
            axis.contourf(real, imag, stable, levels=[-0.5, 0.5, 1.5], colors=['white', '#5b8f5a'])
            axis.contour(real, imag, stable, levels=[0.5], colors='#1d4d2d', linewidths=0.8)
            axis.axhline(0, color='black', linewidth=0.4)
            axis.axvline(0, color='black', linewidth=0.4)
            if row == 0:
                axis.set_title(f'z_imp = {zi:g}', fontsize=9)
            if col == 0:
                axis.set_ylabel(label, fontsize=9)
            axis.set_aspect('equal')
            axis.set_xlabel('Re(z_exp)', fontsize=8)
            axis.set_ylabel('Im(z_exp)', fontsize=8)
    fig.suptitle('Two-parameter IMEX stability: |R(z_exp, z_imp)| <= 1, z_imp fixed per column')
    fig.tight_layout()
    fig.savefig('images/imex_stability_slices.png', dpi=200)
    plt.close(fig)


def main():
    os.makedirs('images', exist_ok=True)
    real = np.linspace(-8.0, 3.0, 401)
    imag = np.linspace(-6.0, 6.0, 401)
    grid = real[None, :] + 1j * imag[:, None]

    tableau_schemes = [
        scheme for scheme in IntegrationSchemes
        if hasattr(scheme.function, 'butcherTableau') or hasattr(scheme.function, 'dirkTableau')
    ]
    columns = 5
    rows = int(np.ceil(len(tableau_schemes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.8 * rows), squeeze=False)
    for axis, scheme in zip(axes.flat, tableau_schemes):
        tableau = getattr(scheme.function, 'butcherTableau', getattr(scheme.function, 'dirkTableau', None))
        stable = np.empty(grid.shape, dtype=bool)
        for index, z in np.ndenumerate(grid):
            stable[index] = rk_is_stable(tableau, z)
        axis.contourf(real, imag, stable, levels=[-0.5, 0.5, 1.5], colors=['white', '#5b8f5a'])
        axis.contour(real, imag, stable, levels=[0.5], colors='#1d4d2d', linewidths=0.8)
        axis.axhline(0, color='black', linewidth=0.4)
        axis.axvline(0, color='black', linewidth=0.4)
        axis.set_title(scheme.name, fontsize=9)
        axis.set_aspect('equal')
    for axis in axes.flat[len(tableau_schemes):]:
        axis.set_visible(False)
    fig.suptitle("Dahlquist stability regions: tableau methods (green: |R(z)| <= 1)")
    fig.tight_layout()
    fig.savefig('images/dahlquist_stability_tableau_methods.png', dpi=200)
    plt.close(fig)

    # BDF4/5 regions extend far along the positive real axis (to ~10.7 and ~17.1),
    # so each order gets a window wide enough for its own lobe; BDF1-3 keep the
    # original window.
    bdf_windows = {
        1: (-8.0, 3.0, -6.0, 6.0),
        2: (-8.0, 3.0, -6.0, 6.0),
        3: (-8.0, 3.0, -6.0, 6.0),
        4: (-8.0, 11.5, -7.5, 7.5),
        5: (-8.0, 18.0, -9.0, 9.0),
    }
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for axis, order in zip(axes, (1, 2, 3, 4, 5)):
        lo, hi, ilo, ihi = bdf_windows[order]
        panel_real = np.linspace(lo, hi, 401)
        panel_imag = np.linspace(ilo, ihi, 401)
        panel_grid = panel_real[None, :] + 1j * panel_imag[:, None]
        stable = np.vectorize(lambda z: bdf_is_stable(order, z))(panel_grid)
        axis.contourf(panel_real, panel_imag, stable, levels=[-0.5, 0.5, 1.5],
                      colors=['white', '#c68142'])
        axis.contour(panel_real, panel_imag, stable, levels=[0.5], colors='#7a3c09',
                     linewidths=1.0)
        axis.axhline(0, color='black', linewidth=0.4)
        axis.axvline(0, color='black', linewidth=0.4)
        axis.set_title(f'BDF{order}')
        axis.set_xlabel('Re(z)')
        axis.set_ylabel('Im(z)')
        axis.set_aspect('equal')
    fig.suptitle('Dahlquist stability regions: BDF methods')
    fig.tight_layout()
    fig.savefig('images/dahlquist_stability_bdf_methods.png', dpi=200)
    plt.close(fig)

    imex_stability_slices()


if __name__ == '__main__':
    main()