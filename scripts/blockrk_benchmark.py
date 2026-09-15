"""Phase 6: the coupled fully implicit block pair vs the baselines
-> images/blockrk_benchmark.png.

Two schemes are registered through the generic block driver
(``fullyimplicit.py``): Gauss-Legendre 2 (order 4, A-stable, symplectic for
separable Hamiltonians, NOT L-stable) and Radau IIA s=2 (order 3, L-stable,
stiffly accurate). Every configuration is fixed here (deterministic, no
adaptive steps), and the three panels pin the three properties that separate
the pair from the rest of the registry:

* Panel 1 (symplecticity, max energy drift vs horizon on the linear
  oscillator): the symplectic schemes stay inside their O(dt^p) band no matter
  how long the run (Velocity Verlet order 2, PEFRL order 4, and -- measured,
  not assumed -- Gauss-Legendre 2), while RK4 (also order 4) accumulates
  secularly. GL2's band sits at JFNK-solve noise (~1e-10): for a *linear*
  problem its per-mode amplification |R(iy)| = 1 is exact, so the only energy
  error is the nonlinear solve's.
* Panel 2 (L-damping, Prothero-Robinson stiff relaxation, rate 100,
  dt 0.1, z = -10): the L-stable schemes (Radau IIA, BE, BDF2-5, ESDIRK)
  damp the stiff transient to their method-order floor in a few steps -- the
  BDF3-5 curves carry the shared DP5 startup spike documented in
  NOTES.md S3.7 -- while the A-stable but NOT L-stable pair (Gauss-Legendre 2,
  |R(-10)| ~= 0.30; Trapezoidal, |R(-10)| = 2/3 by the (1-z/2)/(1+z/2) closed
  form) bounds the transient but damps it slowly, one |R(-10)| factor per step.
* Panel 3 (order ladder, oscillator convergence): both block schemes land on
  their designed orders, and their error constants at equal order are the
  smallest in the registry (GL2 vs RK4 / ESDIRK4(3)6; Radau vs ESDIRK3(2)4 /
  TR-BDF2).

Run:  OMP_NUM_THREADS=4 python scripts/blockrk_benchmark.py
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.stability import bdf_characteristic_roots, rk_stability_function

DT_ENERGY = 0.1
T_HORIZONS = [10.0, 20.0, 40.0, 80.0, 160.0]

#: Panel 1: symplectic (bounded band) vs secular (linear growth in T).
ENERGY_SCHEMES = [
    ('Velocity Verlet', 'vv'),
    ('PEFRL', 'pefrl'),
    ('Gauss-Legendre 2', 'gl2'),
    ('RK4', 'rk4'),
]

#: Panel 2: stiff Prothero-Robinson relaxation, x' = -rate*(x - tanh(t)) + sech^2(t),
#: exact x(t) = tanh(t) + (x(0) - tanh(0)) * exp(-rate*t); with x(0) = 1 the
#: stiff transient starts at 1 and decays exactly as |R(-rate*dt)|^n per step.
STIFF_RATE = 100.0
STIFF_DT = 0.1
STIFF_STEPS = 100   # t in (0, 10]
STIFF_SCHEMES = [
    ('Radau IIA s=2', 'radau'),
    ('Backward Euler (implicit)', 'be'),
    ('BDF2', 'bdf2'),
    ('BDF3', 'bdf3'),
    ('BDF4', 'bdf4'),
    ('BDF5', 'bdf5'),
    ('ESDIRK4(3)6L[2]SA', 'esdirk6'),
    ('Gauss-Legendre 2', 'gl2'),
    ('Trapezoidal (Crank-Nicolson)', 'trap'),
]

#: Panel 3: order ladder on the linear oscillator.
ORDER_SCHEMES = [
    ('Gauss-Legendre 2', 'gl2'),
    ('Radau IIA s=2', 'radau'),
    ('RK4', 'rk4'),
    ('ESDIRK3(2)4L[2]SA', 'esdirk4'),
    ('ESDIRK4(3)6L[2]SA', 'esdirk6'),
    ('TR-BDF2', 'trbdf2'),
    ('Trapezoidal (Crank-Nicolson)', 'trap'),
    ('BDF5', 'bdf5'),
]
ORDER_DTS = [0.2, 0.1, 0.05, 0.025, 0.0125]
ORDER_T = 2.0

COLORS = {
    'vv': '#3478a6', 'pefrl': '#758c37', 'gl2': '#c44e52', 'rk4': '#9a5b36',
    'radau': '#c44e52', 'be': '#5b6ca6', 'bdf2': '#758c37', 'bdf3': '#9a5b36',
    'bdf4': '#a04b8f', 'bdf5': '#8c8c8c', 'esdirk6': '#5b8fc4', 'esdirk4': '#4c9f70',
    'trbdf2': '#c98a2d', 'trap': '#20456e',
}
MARKERS = {
    'vv': 'o', 'pefrl': 's', 'gl2': '*', 'rk4': '^',
    'radau': '*', 'be': 'o', 'bdf2': 's', 'bdf3': 'D', 'bdf4': 'v', 'bdf5': 'P',
    'esdirk6': 'o', 'esdirk4': 's', 'trbdf2': 'D', 'trap': 'v', 'bdf5': 'P',
}


def stiff_rhs(rate):
    """The nonlinear Prothero-Robinson form with exact solution x(t) = tanh(t).

    Affine-linear in the state (the force is -rate*(x - tanh(t))), so the stiff
    transient x - tanh(t) decays exactly as |R(-rate*dt)|^n under any one-step
    method -- the panel measures the methods' L-damping, not a nonlinearity.
    """
    def rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        time = float(system.t)
        target = math.tanh(time)
        target_derivative = 1.0 / math.cosh(time) ** 2
        return testing.ParticleUpdate(
            dxdt=-rate * (state.x - target) + target_derivative,
            dudt=state.u * 0.0,
            dedt=state.e * 0.0,
        ), None
    return rhs


def panel1_energy():
    """Max relative energy drift vs horizon (symplectic = flat, secular = grows)."""
    prob = testing.PROBLEMS['oscillator']()
    data = {}
    for name, key in ENERGY_SCHEMES:
        scheme = getIntegrator(name)
        data[key] = (name,
                     [testing.max_energy_drift(scheme, prob, DT_ENERGY, T)
                      for T in T_HORIZONS])
    print('\nPanel 1: max relative energy drift on the linear oscillator '
          f'(dt = {DT_ENERGY})')
    print(f'  {"scheme":>28}  ' + '  '.join(f'T={T:g}' for T in T_HORIZONS))
    for name, key in ENERGY_SCHEMES:
        _, drifts = data[key]
        print(f'  {name:>28}  ' + '  '.join(f'{d:10.2e}' for d in drifts))
    return data


def panel2_stiff():
    """Stiff transient decay vs time on the Prothero-Robinson relaxation."""
    print(f'\nPanel 2: |x(t) - tanh(t)| on the stiff relaxation '
          f'(rate = {STIFF_RATE:g}, dt = {STIFF_DT}, z = -{STIFF_RATE * STIFF_DT:g})')
    print('  per-step stiff-mode amplification |R(z)| / largest BDF root:')
    data = {}
    for name, key in STIFF_SCHEMES:
        scheme = getIntegrator(name)
        tableau = getattr(scheme.function, 'blockTableau',
                          getattr(scheme.function, 'dirkTableau',
                                  getattr(scheme.function, 'butcherTableau', None)))
        if tableau is not None:
            amp = abs(rk_stability_function(tableau, -STIFF_RATE * STIFF_DT))
            amp_label = f'|R(-10)| = {amp:.4g}'
        else:  # BDF family: the dominant characteristic root at z
            amp = float(np.max(np.abs(bdf_characteristic_roots(
                scheme.order, -STIFF_RATE * STIFF_DT))))
            amp_label = f'|rho_max(-10)| = {amp:.4g}'
        print(f'  {name:>28}:  {amp_label}')

        system = testing.PROBLEMS['oscillator']().initial()
        ref = get_reference_state(system)
        ref.x = ref.x.clone()
        history = StepHistory(maxlen=max(1, scheme.steps))
        ts, errs = [], []
        for _ in range(STIFF_STEPS):
            result = scheme(system, dt=STIFF_DT, f=stiff_rhs(STIFF_RATE),
                            history=history)
            system, history = result.state, result.history or history
            time = float(system.t)
            ts.append(time)
            errs.append(abs(float(get_reference_state(system).x[0])
                            - math.tanh(time)))
        data[key] = (name, ts, errs)
    print(f'  {"scheme":>28}  {"t=1":>10}  {"t=5":>10}  {"t=10":>10}')
    for name, key in STIFF_SCHEMES:
        _, ts, errs = data[key]
        row = {round(t, 1): e for t, e in zip(ts, errs)}
        print(f'  {name:>28}  {row[1.0]:10.2e}  {row[5.0]:10.2e}  {row[10.0]:10.2e}')
    return data


def panel3_order():
    """Convergence order ladder on the linear oscillator."""
    prob = testing.PROBLEMS['oscillator']()
    data = {}
    print(f'\nPanel 3: convergence on the linear oscillator (T = {ORDER_T})')
    print(f'  {"scheme":>28}  {"measured order":>14}  {"error @ dt=0.0125":>18}')
    for name, key in ORDER_SCHEMES:
        scheme = getIntegrator(name)
        order, errors = testing.convergence(scheme, prob, ORDER_DTS, ORDER_T)
        data[key] = (name, list(ORDER_DTS), list(errors), order)
        o = f'{order:.2f}' if order is not None else 'n/a'
        print(f'  {name:>28}  {o:>14}  {errors[-1]:18.3e}')
    return data


def main():
    os.makedirs('images', exist_ok=True)
    e_data = panel1_energy()
    s_data = panel2_stiff()
    o_data = panel3_order()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # Panel 1: energy drift vs horizon.
    for name, key in ENERGY_SCHEMES:
        _, drifts = e_data[key]
        ax1.loglog(T_HORIZONS, drifts, marker=MARKERS[key], color=COLORS[key],
                   label=name, markersize=6, linewidth=1.4)
    ax1.axhline(1e-2, color='#dddddd', linestyle=':', linewidth=1)
    ax1.set_xlabel('horizon T')
    ax1.set_ylabel('max relative energy drift')
    ax1.set_title('Symplectic = bounded band; RK4 = secular growth')
    ax1.grid(which='both', alpha=0.4)
    ax1.legend(fontsize=8)

    # Panel 2: stiff transient decay.
    for name, key in STIFF_SCHEMES:
        _, ts, errs = s_data[key]
        ax2.semilogy(ts, errs, marker=MARKERS[key], color=COLORS[key],
                     label=name, markersize=4, linewidth=1.2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('|x(t) - tanh(t)|')
    ax2.set_title('L-stable: transient to the order floor in a few steps;\n'
                  'A-stable, not L: bounded but slow (|R(-10)|/step); '
                  'BDF3-5: DP5 startup spike (S3.7)')
    ax2.set_ylim(1e-13, 1e13)
    ax2.grid(which='both', alpha=0.4)
    ax2.legend(fontsize=6.5, loc='center right')

    # Panel 3: order ladder (the measured order of every curve is in its label).
    for name, key in ORDER_SCHEMES:
        _, dts, errs, order = o_data[key]
        ax3.loglog(dts, errs, marker=MARKERS[key], color=COLORS[key],
                   label=f'{name} (p={order:.0f})' if order else name,
                   markersize=5, linewidth=1.2)
    ax3.set_xlabel('dt')
    ax3.set_ylabel('error in (x, u) at T = 2')
    ax3.set_title('Order ladder: the block pair lands on its designed order\n'
                  'with the smallest constants at equal order')
    ax3.grid(which='both', alpha=0.4)
    ax3.legend(fontsize=7, loc='lower left')

    fig.suptitle('Phase 6: coupled fully implicit RK (Gauss-Legendre 2, Radau IIA s=2) '
                 'vs the baselines (default JFNK, fixed dt)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('images/blockrk_benchmark.png', dpi=200)
    plt.close(fig)
    print('\nWrote images/blockrk_benchmark.png')


if __name__ == '__main__':
    main()
