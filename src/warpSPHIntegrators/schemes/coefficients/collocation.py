"""schemes.coefficients.collocation — extracted numerical definitions with unchanged formulas."""


import numpy as np


from warpSPHIntegrators.schemes.coefficients.coupled import BlockTableau

def collocation_tableau(family, stages):
    """Build float64 collocation coefficients, with stages in [2, 9]."""
    if family not in ('gauss', 'radau'):
        raise ValueError("family must be 'gauss' or 'radau'")
    if type(stages) is not int or not 2 <= stages <= 9:
        raise ValueError('stages must be an integer from 2 through 9')
    leg = np.polynomial.legendre
    if family == 'gauss':
        x, quadrature = leg.leggauss(stages)
    else:
        # Right Radau nodes are the roots of P_s - P_{s-1}, including +1.
        polynomial = np.zeros(stages+1); polynomial[-1] = 1; polynomial[-2] = -1
        roots = leg.legroots(polynomial)
        if np.max(np.abs(roots.imag)) > 1e-12:
            raise ValueError('Radau node calculation produced nonreal roots')
        x = np.sort(roots.real); x[-1] = 1.
    c = (x+1)/2
    v = leg.legvander(x, stages-1)
    cardinal = np.linalg.solve(v, np.eye(stages))
    integrals = np.empty((stages, stages))
    for k in range(stages):
        polynomial = np.zeros(stages); polynomial[k] = 1
        primitive = leg.legint(polynomial)
        integrals[:, k] = (leg.legval(x, primitive)-leg.legval(-1., primitive))/2
    a = integrals@cardinal
    b = quadrature/2 if family == 'gauss' else a[-1].copy()
    if family == 'gauss':
        # Source sets the s-th monomial moment to zero. Express the resulting
        # moment functional in a Legendre basis to avoid a Vandermonde inverse.
        moments = []
        for k in range(stages):
            p = leg.Legendre.basis(k, domain=[0., 1.]).convert(kind=np.polynomial.Polynomial)
            moments.append(sum(p.coef[j]/(j+1) for j in range(min(k+1, stages-1))))
        embedded = np.linalg.solve(v.T, moments)
        companion = np.concatenate(([0.], embedded))
    else:
        # Interpolate f(0) from the s stage nodes. The defect is O(h^s),
        # so adding h*(sum L_j(0)*f_j-f(0)) gives a companion of order s.
        at_zero = np.array([(-1.)**k for k in range(stages)])@cardinal
        companion = np.concatenate(([-1.], b+at_zero))
    for array in (a, b, c, companion):array.setflags(write=False)
    return BlockTableau(a, b, c, companion)
