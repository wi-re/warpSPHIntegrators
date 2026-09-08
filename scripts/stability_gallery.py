"""Generate Dahlquist stability-region figures for implemented tableau and BDF methods.

The regions apply to y' = lambda*y with z = dt*lambda. Newmark/Verlet use a
second-order oscillator stability analysis, and IMEX has a two-parameter split
region, so neither is represented by this scalar one-parameter diagnostic.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators.stability import bdf_is_stable, rk_is_stable
from warpSPHIntegrators.integration import IntegrationSchemes


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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, order in zip(axes, (1, 2, 3)):
        stable = np.vectorize(lambda z: bdf_is_stable(order, z))(grid)
        axis.contourf(real, imag, stable, levels=[-0.5, 0.5, 1.5], colors=['white', '#c68142'])
        axis.contour(real, imag, stable, levels=[0.5], colors='#7a3c09', linewidths=1.0)
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


if __name__ == '__main__':
    main()