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


def imex_stability_function(tableau, z_explicit: complex, z_implicit: complex) -> complex:
    """Additive (IMEX) Runge-Kutta stability function ``R(z_e, z_i)``.

    Applied to the split test equation ``y' = f_e(y) + f_i(y)`` with
    ``f_e = lambda_e y`` and ``f_i = lambda_i y`` (so ``z_e = dt*lambda_e``,
    ``z_i = dt*lambda_i``). ``tableau`` is an ``ark.AdditiveTableau`` (``.c``,
    ``.a_explicit``, ``.a_implicit``, ``.b_explicit``, ``.b_implicit``).

    The stages ``z_1 .. z_{s-1}`` (``z_0 = y^n``) satisfy a lower-triangular linear
    system ``M K = r`` that couples the two halves; ``R`` is then read off from the
    b-weighted update. The pure limits recover the component stability functions:
    ``R(0, z)`` is the implicit-half (DIRK) stability function and ``R(z, 0)`` the
    explicit-half (ERK) one, both in the standard ``1 + z b^T (I - z A)^{-1} 1`` form.
    """
    s = len(tableau.c)
    a_e = np.asarray(tableau.a_explicit, dtype=complex)
    a_i = np.asarray(tableau.a_implicit, dtype=complex)
    b_e = np.asarray(tableau.b_explicit, dtype=complex)
    b_i = np.asarray(tableau.b_implicit, dtype=complex)
    ze, zi = z_explicit, z_implicit
    if s == 1:
        return 1.0 + ze * b_e[0] + zi * b_i[0]

    # Lower-triangular stage system for the internal stages z_1 .. z_{s-1}.
    m = s - 1
    matrix = np.zeros((m, m), dtype=complex)
    rhs = np.empty(m, dtype=complex)
    for i in range(1, s):
        ii = i - 1
        matrix[ii, ii] = 1.0 - zi * a_i[i, i]
        for j in range(1, i):
            matrix[ii, j - 1] = -(ze * a_e[i, j] + zi * a_i[i, j])
        rhs[ii] = 1.0 + ze * a_e[i, 0] + zi * a_i[i, 0]
    stages = np.linalg.solve(matrix, rhs)
    return 1.0 + ze * b_e[0] + zi * b_i[0] + (ze * b_e[1:] + zi * b_i[1:]) @ stages


def imex_is_stable(tableau, z_explicit: complex, z_implicit: complex,
                   tolerance: float = 1e-12) -> bool:
    return abs(imex_stability_function(tableau, z_explicit, z_implicit)) <= 1.0 + tolerance


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


def damped_oscillator_amplification_matrix(method: str, h_omega: float, zeta: float) -> np.ndarray:
    """Amplification matrix for ``x'' + 2*zeta*omega*x' + omega^2*x = 0`` in ``(x, h*v)``.

    ``zeta`` is the damping ratio (critical damping at ``zeta = 1``); ``c = 2*zeta*h*omega``
    is the dimensionless damping below. Each formula uses the same convention as the
    registered scheme it describes -- the force is evaluated with whichever velocity is
    available at that evaluation point:

    - ``leapfrog``: synchronized form; the second force evaluation sits at the new
      position with the *old* velocity, so its damping is an explicit old-velocity term
      that is itself unstable for ``c > 2``.
    - ``velocity_verlet``: kick-drift-kick; the second kick uses the half-step velocity.
    - ``symplectic_euler``: the registered KDK form, force evaluated at the
      half-drifted state ``(x + h*v/2, v + h*a/2)``.
    - ``newmark_average_acceleration`` / ``newmark_linear_acceleration``: the library's
      fixed-point Newmark (``newmark.py``), where the unknown is the endpoint state and
      the endpoint acceleration is the force evaluated at that endpoint,
      ``a_{n+1} = f(x_{n+1}, v_{n+1})``. For position-only forces this is the textbook
      Newmark update; with velocity-dependent forces the damping enters through the
      endpoint force rather than the textbook acceleration relation.

    Every formula reduces to :func:`oscillator_amplification_matrix` at ``zeta = 0``.
    """
    q2 = h_omega ** 2
    c = 2.0 * zeta * h_omega
    if method in ('leapfrog', 'velocity_verlet'):
        position_x = 1.0 - 0.5 * q2
        if method == 'leapfrog':
            return np.array([
                [position_x, 1.0 - 0.5 * c],
                [-0.5 * q2 * (1.0 + position_x), 1.0 - c - 0.5 * q2 + 0.25 * c * q2],
            ])
        return np.array([
            [position_x, 1.0 - 0.5 * c],
            [-0.5 * q2 * (1.0 + position_x - 0.5 * c),
             (1.0 - 0.5 * c) * (1.0 - 0.5 * c - 0.5 * q2)],
        ])
    if method == 'symplectic_euler':
        return np.array([
            [1.0 - 0.25 * q2 * (2.0 - c), 1.0 - 0.25 * q2 - 0.5 * c + 0.25 * c * c],
            [0.5 * q2 * (c - 2.0), 1.0 - 0.5 * q2 - c + 0.5 * c * c],
        ])
    if method in ('newmark_average_acceleration', 'newmark_linear_acceleration'):
        beta = 0.25 if method == 'newmark_average_acceleration' else 1.0 / 6.0
        gamma = 0.5
        left = np.array([
            [1.0 + beta * q2, beta * c],
            [gamma * q2, 1.0 + gamma * c],
        ])
        right = np.array([
            [1.0 - (0.5 - beta) * q2, 1.0 - (0.5 - beta) * c],
            [-(1.0 - gamma) * q2, 1.0 - (1.0 - gamma) * c],
        ])
        return np.linalg.solve(left, right)
    raise ValueError(f'Unknown damped oscillator stability method {method!r}')


def damped_oscillator_spectral_radius(method: str, h_omega: float, zeta: float) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(
        damped_oscillator_amplification_matrix(method, h_omega, zeta)))))


def damped_oscillator_is_stable(method: str, h_omega: float, zeta: float,
                                tolerance: float = 1e-7) -> bool:
    return damped_oscillator_spectral_radius(method, h_omega, zeta) <= 1.0 + tolerance