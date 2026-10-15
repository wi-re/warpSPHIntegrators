"""schemes.coefficients.ndf — extracted numerical definitions with unchanged formulas."""


from functools import lru_cache

import math

from numbers import Real


import numpy as np


_KAPPA=(-37/200,-1/9,-823/10000,-83/2000,0.)

_HARMONIC=tuple(sum(1/j for j in range(1,k+1)) for k in range(1,7))

@lru_cache(maxsize=128)
def ndf_difference_transform(order,ratio):
    """Immutable R(ratio)*U from the pinned quasi-constant history resampling.

    New difference j = sum_i old difference i * transform[i,j].
    This transforms a polynomial history; it is not variable-order selection.
    """
    if type(order) is not int or not 1<=order<=5:raise ValueError('NDF order must be from 1 through 5')
    if isinstance(ratio,bool) or not isinstance(ratio,Real) or not math.isfinite(ratio) or ratio<=0:
        raise ValueError('positive finite timestep ratio required')
    def matrix(scale):
        out=np.zeros((order,order))
        for r in range(1,order+1):
            out[0,r-1]=-r*scale
            for j in range(2,order+1):out[j-1,r-1]=out[j-2,r-1]*((j-1)-r*scale)/j
        return out
    return tuple(tuple(row) for row in matrix(ratio)@matrix(1.))
