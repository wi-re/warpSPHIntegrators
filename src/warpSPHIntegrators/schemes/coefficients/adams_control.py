"""schemes.coefficients.adams_control — extracted numerical definitions with unchanged formulas."""


import numpy as np


from warpSPHIntegrators.schemes.coefficients.adams import adams_weights

def _defect_weights(nodes, order):
    """Source AM(q+1)-AM(q) difference on one shared endpoint derivative."""
    points=np.r_[1.,nodes];result=np.zeros(len(points))
    result[:order+1]=adams_weights(points[:order+1])
    result[:order]-=adams_weights(points[:order])
    return result

def _upper_weights(nodes, order):
    """Source dt*gamma_star[p+2]*phi_endpoint[p+2], normalized by outer dt."""
    points=np.r_[1.,np.asarray(nodes)[:order+1]]
    if len(points)!=order+2:raise ValueError('one extra history derivative required')
    gamma=(1.,-.5,-1/12,-1/24,-19/720,-3/160,-863/60480)
    scale=gamma[order+1]*np.prod(1-points[1:])
    return np.array([scale/np.prod(point-np.delete(points,i)) for i,point in enumerate(points)])
