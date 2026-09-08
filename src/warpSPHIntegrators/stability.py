"""Numerical absolute-stability checks for scalar Dahlquist test equations."""

from __future__ import annotations

import numpy as np

from .bdf import getBDFCoefficients


def rk_stability_function(tableau, z: complex) -> complex:
    """The propagated Runge-Kutta stability function ``R(z)``."""
    identity = np.eye(len(tableau.c), dtype=complex)
    ones = np.ones(len(tableau.c), dtype=complex)
    weights = tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b
    return 1.0 + z * np.asarray(weights, dtype=complex) @ np.linalg.solve(
        identity - z * tableau.a, ones
    )


def rk_is_stable(tableau, z: complex, tolerance: float = 1e-12) -> bool:
    return abs(rk_stability_function(tableau, z)) <= 1.0 + tolerance


def bdf_characteristic_roots(order: int, z: complex) -> np.ndarray:
    """Roots of the BDF characteristic polynomial at ``z = h * lambda``."""
    state_weights, derivative_weight = getBDFCoefficients(order)
    return np.roots([1.0 - derivative_weight * z, *(-np.asarray(state_weights))])


def bdf_is_stable(order: int, z: complex, tolerance: float = 1e-12) -> bool:
    return np.max(np.abs(bdf_characteristic_roots(order, z))) <= 1.0 + tolerance


def oscillator_amplification_matrix(method: str, h_omega: float) -> np.ndarray:
    """Amplification matrix for ``x'' + omega^2*x = 0`` in ``(x, h*v)`` variables."""
    q2 = h_omega ** 2
    if method in ('leapfrog', 'velocity_verlet'):
        position_x = 1.0 - 0.5 * q2
        return np.array([
            [position_x, 1.0],
            [-0.5 * q2 * (1.0 + position_x), position_x],
        ])
    if method == 'symplectic_euler':
        return np.array([
            [1.0 - 0.5 * q2, 1.0 - 0.25 * q2],
            [-q2, 1.0 - 0.5 * q2],
        ])
    if method in ('newmark_average_acceleration', 'newmark_linear_acceleration'):
        beta = 0.25 if method == 'newmark_average_acceleration' else 1.0 / 6.0
        gamma = 0.5
        denominator = 1.0 + beta * q2
        position_x = (1.0 - (0.5 - beta) * q2) / denominator
        position_v = 1.0 / denominator
        velocity_x = -q2 * ((1.0 - gamma) + gamma * position_x)
        velocity_v = 1.0 - gamma * q2 * position_v
        return np.array([[position_x, position_v], [velocity_x, velocity_v]])
    raise ValueError(f'Unknown oscillator stability method {method!r}')


def oscillator_spectral_radius(method: str, h_omega: float) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(oscillator_amplification_matrix(method, h_omega)))))


def oscillator_is_stable(method: str, h_omega: float, tolerance: float = 1e-7) -> bool:
    return oscillator_spectral_radius(method, h_omega) <= 1.0 + tolerance