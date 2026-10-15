"""schemes.coefficients.rosenbrock — extracted numerical definitions with unchanged formulas."""
from dataclasses import dataclass

from functools import lru_cache

import math


from warpSPHIntegrators.schemes.coefficients.rosenbrock_data import W6S4OS


@dataclass(frozen=True)
class RosenbrockTableau:
    """Rodas-form increments: M*K_i=h*gamma*f_i+gamma*sum(C_ij*K_j)+h²*gamma*d_i*f_t."""
    A: tuple
    C: tuple
    gamma: float
    c: tuple
    d: tuple
    b: tuple
    btilde: tuple|None
    order: int

@lru_cache(maxsize=3)
def rosenbrock_tableau(method):
    if method=='RosenbrockW6S4OS':
        return RosenbrockTableau(**W6S4OS,order=4)
    if method not in ('Rosenbrock23','Rosenbrock32'):raise ValueError('Unknown Rosenbrock method')
    g=1/(2+math.sqrt(2));c32=6+math.sqrt(2)
    # Algebraic change of source slopes: K1=h*g*k1, K2=h*g*(k2-k1),
    # K3=h*g*k3-2*K1-c32*K2. This preserves both main maps and raw defects.
    A=((0.,0.,0.),(1/(2*g),0.,0.),(1/g,1/g,0.))
    C=((0.,0.,0.),(-1/g,0.,0.),(-2/g,-c32/g,0.))
    low=(1/g,1/g,0.);high=(7/(6*g),(4+c32)/(6*g),1/(6*g))
    # Third-stage partial-time coefficient is corrected by exact autonomous
    # augmentation t'=1: source dt*f_t must be dt*g*f_t. In these variables
    # d3=-g (the uncorrected source would give 1-2*g and an O(h²) defect).
    return RosenbrockTableau(A,C,g,(0.,.5,1.),(g,0.,-g),
        low if method=='Rosenbrock23' else high,(1/(6*g),(c32-2)/(6*g),1/(6*g)),
        2 if method=='Rosenbrock23' else 3)
