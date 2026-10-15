"""schemes.coefficients.stabilized — extracted numerical definitions with unchanged formulas."""
from dataclasses import dataclass

from functools import lru_cache


from warpSPHIntegrators.schemes.coefficients.chebyshev import _rkl2_coeffs

_MINIMUM={'rkl1':3,'rkl2':3,'rkg1':2,'rkg2':3}

def stabilized_interval(family,stages):
    """Conservative negative-real sizing interval used by the pinned source.

    Gegenbauer sizing uses half its full polynomial interval. This is a supplied
    spectral-radius bound, not an internal eigenvalue or pseudospectrum estimate.
    """
    if family not in _MINIMUM:raise ValueError('unknown stabilized family')
    if type(stages) is not int or stages<_MINIMUM[family]:raise ValueError('invalid stage count')
    if family=='rkl1':return float(stages*(stages+1))
    if family=='rkl2':return float(stages*stages+stages-2)/2
    if family=='rkg1':return float(stages*(stages+3))/4
    return float((stages+4)*(stages-1))/6

@dataclass(frozen=True)
class StabilizedCoefficients:
    mu: tuple
    nu: tuple
    mtilde: tuple
    gtilde: tuple
    c: tuple
    b: tuple

@lru_cache(maxsize=64)
def stabilized_coefficients(family,stages):
    """Immutable three-term recurrence, internally consistent times and final weights."""
    stabilized_interval(family,stages)
    s=stages
    if family=='rkl2':
        _,mu,nu,mtilde,gtilde=_rkl2_coeffs(s)
    else:
        mu=[0.]*(s+1);nu=[0.]*(s+1);mtilde=[0.]*(s+1);gtilde=[0.]*(s+1)
        if family=='rkl1':
            w1=2/(s*(s+1));mtilde[1]=w1
            for j in range(2,s+1):
                mu[j]=(2*j-1)/j;nu[j]=-(j-1)/j;mtilde[j]=mu[j]*w1
        else:
            if family=='rkg1':
                w1=4/(s*(s+3));b=[2/((j+1)*(j+2)) for j in range(s+1)]
            else:
                w1=6/((s+4)*(s-1));b=[1.,1/3]+[
                    4*(j-1)*(j+4)/(3*j*(j+1)*(j+2)*(j+3)) for j in range(2,s+1)]
            mtilde[1]=w1
            for j in range(2,s+1):
                mu[j]=(2*j+1)/j*b[j]/b[j-1]
                nu[j]=-(j+1)/j*b[j]/b[j-2]
                mtilde[j]=mu[j]*w1
                if family=='rkg2':gtilde[j]=-mtilde[j]*(1-j*(j+1)/2*b[j-1])
    c=[0.,mtilde[1]]+[0.]*(s-1)
    for j in range(2,s+1):c[j]=mu[j]*c[j-1]+nu[j]*c[j-2]+mtilde[j]+gtilde[j]
    # Reverse propagation through the recurrence obtains all final derivative
    # weights in O(s), without expanding a dense Butcher tableau.
    adjoint=[0.]*(s+1);adjoint[s]=1.;weights=[0.]*s
    for j in range(s,1,-1):
        weights[j-1]+=adjoint[j]*mtilde[j];weights[0]+=adjoint[j]*gtilde[j]
        adjoint[j-1]+=adjoint[j]*mu[j];adjoint[j-2]+=adjoint[j]*nu[j]
    weights[0]+=adjoint[1]*mtilde[1]
    return StabilizedCoefficients(*(tuple(x) for x in (mu,nu,mtilde,gtilde,c,weights)))
