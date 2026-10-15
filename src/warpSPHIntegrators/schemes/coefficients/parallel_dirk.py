"""schemes.coefficients.parallel_dirk — extracted numerical definitions with unchanged formulas."""


import numpy as np

from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau


def parallel_dirk_tableau():
    """PDIRK 4-stage, order-four tableau; stages (0,1) and (2,3) are independent."""
    return butcherTableau(a=np.array([[.5,0.,0.,0.],[0.,2/3,0.,0.],
        [-5/2,5/2,.5,0.],[-5/3,4/3,0.,2/3]]),
        b=np.array([-1.,1.5,-1.,1.5]),c=np.array([.5,2/3,.5,1/3]))
