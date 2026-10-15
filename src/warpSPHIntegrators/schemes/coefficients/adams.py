"""schemes.coefficients.adams — extracted numerical definitions with unchanged formulas."""


import numpy as np


def adams_weights(nodes):
    """Integrate cardinal polynomials at distinct normalized nodes over [0,1]."""
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=1 or not len(nodes) or not np.isfinite(nodes).all() or len(np.unique(nodes))!=len(nodes):
        raise ValueError('Finite distinct one-dimensional Adams nodes required')
    result=[]
    for j,node in enumerate(nodes):
        polynomial=np.polynomial.Polynomial([1.])
        for k,other in enumerate(nodes):
            if k!=j:polynomial*=np.polynomial.Polynomial([-other,1.])/(node-other)
        primitive=polynomial.integ()
        result.append(primitive(1)-primitive(0))
    return np.asarray(result)
