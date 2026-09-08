"""Generate the Newmark demo gallery using the repo's default modified oscillator.

This mirrors the notebook-based gallery used for the other schemes: one phase/energy
summary for the order-2 family and one difference plot against the reference 5th-order
Nystrom solver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from warpSPHIntegrators import getIntegrator
from warpSPHIntegrators.fields import BaseState, constant, get_reference_state, integrated, reference_state, tagged, update_component, update_position
from warpSPHIntegrators.integration import BaseIntegrationSystem, ComponentUpdateSpec, PositionUpdateSpec


m = 1.0
x_0 = 1.0
u_0 = 0.0
k_0 = 5.0
alpha = -5.0
beta = 0.5
gamma = 0.25
c = 0.1


def build_initial_state():
    e_0 = m * u_0**2 / 2 + k_0 * x_0**2 / 2
    return HarmonicOscillatorState(
        x=torch.tensor([x_0]),
        u=torch.tensor([u_0]),
        e=torch.tensor([e_0]) * 0,
        k=torch.tensor([k_0]),
        m=torch.tensor([m]),
        k_0=k_0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        c=c,
    )


@dataclass
class HarmonicOscillatorState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    u: torch.Tensor = integrated('dudt', tags=('velocity',))
    e: torch.Tensor = integrated('dedt', tags=('hidden_energy', 'quantity'))
    k: torch.Tensor = constant(tags=('spring_constant',))
    m: torch.Tensor = constant(tags=('mass',))
    k_0: float = constant(tags=('base_spring_constant',))
    alpha: float = constant(tags=('alpha',))
    beta: float = constant(tags=('beta',))
    gamma: float = constant(tags=('gamma',))
    c: float = constant(tags=('damping',))


@dataclass
class OscillatorUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dudt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dedt: torch.Tensor = tagged(tags=('hidden_energy_derivative',))


@dataclass
class HarmonicOscillatorSystem(BaseIntegrationSystem):
    state: HarmonicOscillatorState = reference_state(tags=('oscillator_state',))
    t: float = 0.0

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        return HarmonicOscillatorSystem(state=state.initializeNewState(), t=self.t)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'hidden_energy', 'hidden_energy_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self


def f_modified_harmonic_system(system: HarmonicOscillatorSystem, dt: float, verbose: bool = False):
    state = get_reference_state(system)
    x = state.x
    u = state.u
    e = state.e
    m = state.m
    alpha = state.alpha
    beta = state.beta
    gamma = state.gamma
    c = state.c
    k_0 = state.k_0
    k_current = k_0 + alpha * e
    dxdt = u
    dudt = (-k_current * x - c * u) / m
    dedt = beta * torch.sign(u) * u**2 - gamma * e
    kinetic_energy = m * u**2 / 2
    potential_energy = k_current * x**2 / 2
    total_energy = kinetic_energy + potential_energy
    return OscillatorUpdate(dxdt, dudt, dedt), (k_current, kinetic_energy, potential_energy, total_energy)


def system_value(system, tag):
    return get_reference_state(system).__getattribute__(tag)


def run_integrator(initial_system, integration_scheme_name, dt, time_limit):
    integrator = getIntegrator(integration_scheme_name)
    num_steps = int(time_limit / dt)
    initial_state = get_reference_state(initial_system)
    kinetic_energy = torch.tensor([initial_state.m * initial_state.u**2 / 2])
    potential_energy = torch.tensor([k_0 * initial_state.x**2 / 2])
    total_energy = kinetic_energy + potential_energy
    states = [(initial_system, kinetic_energy, potential_energy, total_energy)]
    for _ in range(num_steps):
        result = integrator.function(
            states[-1][0],
            dt=dt,
            f=f_modified_harmonic_system,
            verbose=False,
            priorStep=None,
        )
        next_system = result.state
        aux_tuple = result.stages[-1].aux
        if aux_tuple is None:
            raise ValueError('Integrator produced no auxiliary energy payload for the demo plot.')
        if isinstance(aux_tuple, tuple) and len(aux_tuple) == 1:
            aux_tuple = aux_tuple[0]
        if isinstance(aux_tuple, tuple) and len(aux_tuple) == 4:
            _, kinetic_energy, potential_energy, total_energy = aux_tuple
        elif isinstance(aux_tuple, list) and len(aux_tuple) == 4:
            _, kinetic_energy, potential_energy, total_energy = aux_tuple
        else:
            raise ValueError(f'Unexpected aux format: {aux_tuple!r}')
        states.append((next_system, kinetic_energy, potential_energy, total_energy))
    return states, integrator


def plot_trace(axis, trace, label, color):
    axis['A'].plot(
        [system_value(state[0], 'x').item() for state in trace],
        [system_value(state[0], 'u').item() for state in trace],
        label=label, color=color,
    )
    axis['B'].plot([state[0].t for state in trace], [state[1].item() for state in trace], ls='--', color=color)
    axis['B'].plot([state[0].t for state in trace], [state[2].item() for state in trace], ls=':', color=color)
    axis['B'].plot([state[0].t for state in trace], [state[3].item() for state in trace], label=label, color=color)
    axis['C'].plot([state[0].t for state in trace], [system_value(state[0], 'x').item() for state in trace], label=label, color=color)
    axis['D'].plot([state[0].t for state in trace], [system_value(state[0], 'u').item() for state in trace], label=label, color=color)
    axis['E'].plot([state[0].t for state in trace], [system_value(state[0], 'e').item() for state in trace], label=label, color=color)


def plot_difference(axis, trace1, trace2, label, color):
    diff_x = np.array([system_value(state[0], 'x').item() for state in trace1]) - np.array([system_value(state[0], 'x').item() for state in trace2])
    diff_u = np.array([system_value(state[0], 'u').item() for state in trace1]) - np.array([system_value(state[0], 'u').item() for state in trace2])
    diff_e = np.array([system_value(state[0], 'e').item() for state in trace1]) - np.array([system_value(state[0], 'e').item() for state in trace2])
    diff_total_energy = np.array([state[3].item() for state in trace1]) - np.array([state[3].item() for state in trace2])
    axis['A'].plot([state[0].t for state in trace1], diff_x, label=label, color=color)
    axis['B'].plot([state[0].t for state in trace1], diff_u, label=label, color=color)
    axis['C'].plot([state[0].t for state in trace1], diff_e, label=label, color=color)
    axis['D'].plot([state[0].t for state in trace1], diff_total_energy, label=label, color=color)


def main():
    os.makedirs('images', exist_ok=True)
    initial_state = build_initial_state()
    initial_system = HarmonicOscillatorSystem(state=initial_state)
    dt = 0.01
    time_limit = 10.0

    second_order_names = [
        'Midpoint',
        "Heun's Method (2nd order)",
        "Ralston's Method (2nd order)",
        'Leap Frog',
        'Symplectic Euler',
        'Velocity Verlet',
        'Newmark',
    ]
    implicit_scheme_names = [
        'Backward Euler (implicit)',
        'Implicit Midpoint',
        'Trapezoidal (Crank-Nicolson)',
        'SDIRK2',
        'Newmark',
    ]
    reference_name = 'Nystrom 5th order'

    reference_trace, reference_scheme = run_integrator(initial_system, reference_name, dt, time_limit)
    traces = []
    for name in second_order_names:
        trace, scheme = run_integrator(initial_system, name, dt, time_limit)
        traces.append((trace, scheme))

    fig, axis = plt.subplot_mosaic([
        ['A', 'A', 'A', 'B', 'B', 'C'],
        ['A', 'A', 'A', 'E', 'E', 'D'],
    ], figsize=(15, 6))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plot_trace(axis, reference_trace, label=reference_scheme.name, color=color_cycle[0])
    for i, (trace, scheme) in enumerate(traces):
        plot_trace(axis, trace, label=scheme.name, color=color_cycle[(i + 1) % len(color_cycle)])

    axis['A'].set_xlabel('Position (x)')
    axis['A'].set_ylabel('Velocity (u)')
    axis['A'].set_title('Phase Plot of the System')
    axis['A'].legend()
    axis['A'].grid()
    axis['B'].set_title('Energy')
    axis['B'].grid()
    axis['C'].set_title('Position')
    axis['C'].grid()
    axis['D'].set_xlabel('Time')
    axis['D'].set_title('Velocity')
    axis['D'].grid()
    axis['E'].set_xlabel('Time')
    axis['E'].set_title('Hidden Energy')
    axis['E'].grid()

    fig.suptitle(f'Integration of Modified Harmonic Oscillator System (dt={dt}, time_limit={time_limit}) with Second-Order Integrators Including Newmark')
    fig.tight_layout()
    fig.savefig('images/newmark_modified_harmonic_oscillator_order_2_integrators.png', dpi=200)
    plt.close(fig)

    fig, axis = plt.subplot_mosaic('AB\nCD', figsize=(10, 8))
    for i, (trace, scheme) in enumerate(traces):
        plot_difference(axis, trace, reference_trace, label=scheme.name, color=color_cycle[(i + 1) % len(color_cycle)])

    axis['A'].set_title('Position Difference')
    axis['B'].set_title('Velocity Difference')
    axis['C'].set_title('Hidden Energy Difference')
    axis['D'].set_title('Total Energy Difference')

    for ax in axis.values():
        ylim = ax.get_ylim()
        if ylim:
            ax.set_ylim(max(abs(ylim[0]), abs(ylim[1])) * 1.1 * np.array([-1, 1]))
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.grid(which='major', axis='both')
        ax.grid(which='minor', axis='y', ls=':', color='gray')
        ax.axhline(0, ls='--', color='black', lw=1, alpha=0.5)

    fig.suptitle('Differences Compared to Nystrom 5th Order Integrator for the Default Demo Case')
    handles, labels = axis['A'].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=True)
    fig.savefig('images/newmark_integrator_comparison_order_2.png', dpi=250)
    plt.close(fig)

    implicit_traces = []
    for name in implicit_scheme_names:
        trace, scheme = run_integrator(initial_system, name, dt, time_limit)
        implicit_traces.append((trace, scheme))

    fig, axis = plt.subplots(2, 2, figsize=(13, 9))
    for i, (trace, scheme) in enumerate(implicit_traces):
        color = color_cycle[i % len(color_cycle)]
        times = [state[0].t for state in trace]
        positions = [system_value(state[0], 'x').item() for state in trace]
        velocities = [system_value(state[0], 'u').item() for state in trace]
        hidden_energy = [system_value(state[0], 'e').item() for state in trace]
        energy_difference = np.array([state[3].item() for state in trace]) - np.array([state[3].item() for state in reference_trace])
        axis[0, 0].plot(positions, velocities, label=scheme.name, color=color)
        axis[0, 1].plot(times, positions, label=scheme.name, color=color)
        axis[1, 0].plot(times, hidden_energy, label=scheme.name, color=color)
        axis[1, 1].plot(times, energy_difference, label=scheme.name, color=color)

    axis[0, 0].set_title('Phase Plot')
    axis[0, 0].set_xlabel('Position (x)')
    axis[0, 0].set_ylabel('Velocity (u)')
    axis[0, 1].set_title('Position')
    axis[0, 1].set_xlabel('Time')
    axis[1, 0].set_title('Hidden Energy')
    axis[1, 0].set_xlabel('Time')
    axis[1, 1].set_title('Total Energy Difference from Nystrom 5th')
    axis[1, 1].set_xlabel('Time')
    for ax in axis.flat:
        ax.grid()
    handles, labels = axis[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.suptitle('All Implicit Schemes on the Default Modified Harmonic Oscillator Demo')
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig('images/all_implicit_schemes_default_demo.png', dpi=250)
    plt.close(fig)

    print('Generated:')
    print('  images/newmark_modified_harmonic_oscillator_order_2_integrators.png')
    print('  images/newmark_integrator_comparison_order_2.png')
    print('  images/all_implicit_schemes_default_demo.png')


if __name__ == '__main__':
    main()
