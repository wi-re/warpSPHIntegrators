"""Generate linear-oscillator stability plots for Newmark and Verlet-family methods."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators.stability import oscillator_spectral_radius


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


if __name__ == '__main__':
    main()