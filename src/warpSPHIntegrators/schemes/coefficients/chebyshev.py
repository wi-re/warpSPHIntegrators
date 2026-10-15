"""Fixed RKC/RKL recurrence coefficients, stage sizing and clock weights."""
def _rkc1_coeffs(s):
    """RKC1 (order 1, K = 2 s^2): pure Chebyshev, no Y_0 correction."""
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = 1.0 / (s * s)
    for j in range(2, s + 1):
        mu[j] = 2.0
        nu[j] = -1.0
        mtilde[j] = 2.0 / (s * s)
    return mtilde[1], mu, nu, mtilde, gtilde


def _rkc2_coeffs(s):
    """RKC2 (order 2, K = 2(s^2-1)/3): relaxed Chebyshev.

    ``R_s = a_s + b_s T_s(1 + w_1 z)`` with ``w_1 = 3/(s^2-1)``,
    ``b_s = (s^2-1)/(3 s^2)``, ``a_s = 1 - b_s``. The stage weights follow from the
    Chebyshev three-term recurrence; the per-stage amplitude ``b_j`` runs geometrically
    from ``b_0 = b_1 = b_2 = 1/3`` to the final ``b_s``, which keeps every intermediate
    stage ``R_j = a_j + b_j T_j(1 + w_1 z)`` bounded by 1 on the stability interval.
    """
    w1 = 3.0 / (s * s - 1)
    b_s = (s * s - 1) / (3.0 * s * s)
    b = [0.0] * (s + 1)
    b[0] = b[1] = b[2] = 1.0 / 3
    if s >= 3:
        ratio = (b_s / (1.0 / 3)) ** (1.0 / (s - 2))
        for j in range(3, s + 1):
            b[j] = (1.0 / 3) * ratio ** (j - 2)
    b[s] = b_s  # pin the final stage exactly
    a = [1.0 - bj for bj in b]
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = b[1] * w1
    for j in range(2, s + 1):
        mu[j] = 2.0 * (b[j] / b[j - 1])
        nu[j] = -(b[j] / b[j - 2])
        mtilde[j] = mu[j] * w1
        gtilde[j] = -a[j - 1] * mtilde[j]
    return mtilde[1], mu, nu, mtilde, gtilde


def _rkl2_coeffs(s):
    """RKL2 (order 2, K = (s^2+s-2)/2): relaxed Lobatto (Meyer et al. 2014, Eqs 15-19).

    ``R_s = a_s + b_s P_s(1 + w_1 z)`` with ``w_1 = 4/(s^2+s-2)``,
    ``b_s = (s^2+s-2)/(2 s (s+1))``, ``a_s = 1 - b_s``. The stage weights follow from
    the Legendre three-term recurrence with ``b_j/b_{j-1} = (j+2)(j-1)^2 /
    ((j-2)(j+1)^2)`` for ``j >= 3`` and ``b_0 = b_1 = b_2 = 1/3``.
    """
    w1 = 4.0 / (s * s + s - 2)
    b = [0.0] * (s + 1)
    b[0] = b[1] = b[2] = 1.0 / 3
    for j in range(3, s + 1):
        b[j] = b[j - 1] * (j + 2) * (j - 1) ** 2 / ((j - 2) * (j + 1) ** 2)
    a = [1.0 - bj for bj in b]
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = b[1] * w1
    for j in range(2, s + 1):
        mu[j] = (2 * j - 1) / j * (b[j] / b[j - 1])
        nu[j] = -(j - 1) / j * (b[j] / b[j - 2])
        mtilde[j] = mu[j] * w1
        gtilde[j] = -a[j - 1] * mtilde[j]
    return mtilde[1], mu, nu, mtilde, gtilde


_COEFFS = {'rkc1': _rkc1_coeffs, 'rkc2': _rkc2_coeffs, 'rkl2': _rkl2_coeffs}


_NAMES = {'rkc1': 'RKC1', 'rkc2': 'RKC2', 'rkl2': 'RKL2'}


_K = {
    'rkc1': lambda s: 2.0 * s * s,
    'rkc2': lambda s: 2.0 * (s * s - 1) / 3.0,
    'rkl2': lambda s: (s * s + s - 2) / 2.0,
}


_SMIN = {'rkc1': 1, 'rkc2': 3, 'rkl2': 3}


def stage_count(dt_lambda_max, family):
    """Smallest stage count ``s`` with ``K(s) >= dt_lambda_max``.

    ``dt_lambda_max`` is ``dt * |lambda_max| >= 0``. Clamped to the method's minimum
    stage count. Does not enforce odd parity (see the module docstring).
    """
    K = _K[family]
    s = _SMIN[family]
    while K(s) < dt_lambda_max:
        s += 1
    return s


def _recurrence_clock_weights(s, mtilde1, mu, nu, mtilde, gtilde):
    """Consistent first-order stage nodes and final derivative quadrature."""
    c = [0.0, mtilde1] + [0.0] * (s - 1)
    for j in range(2, s + 1):
        c[j] = mu[j]*c[j-1] + nu[j]*c[j-2] + mtilde[j] + gtilde[j]
    adjoint = [0.0] * (s + 1)
    adjoint[s] = 1.0
    weights = [0.0] * s
    for j in range(s, 1, -1):
        weights[j-1] += adjoint[j]*mtilde[j]
        weights[0] += adjoint[j]*gtilde[j]
        adjoint[j-1] += adjoint[j]*mu[j]
        adjoint[j-2] += adjoint[j]*nu[j]
    weights[0] += adjoint[1]*mtilde1
    return c, tuple(weights)
