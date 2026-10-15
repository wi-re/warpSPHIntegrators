"""Unregistered three-stage collocation candidates for the Phase 6 scale gate.

Coefficients integrate the Lagrange basis at shifted Gauss/Radau nodes. See
Hairer & Wanner's collocation derivation: https://www.unige.ch/~hairer/preprints/coimbra.pdf
Names here encode stage count. Upstream RadauIIA5 means order five, s=3;
RadauIIA3 is the already registered order-three, s=2 method. No companion
estimator is claimed by these prototypes.
"""
import numpy as np
from warpSPHIntegrators.drivers.implicit.coupled import BlockTableau, fullyImplicitBlock


def prototype_tableau(family):
    if family == 'gauss3':
        c = np.array([.5-np.sqrt(15)/10,.5,.5+np.sqrt(15)/10])
    elif family == 'radau3':
        c = np.array([(4-np.sqrt(6))/10,(4+np.sqrt(6))/10,1.])
    else:
        raise ValueError(f'Unknown collocation prototype {family}')
    a = np.empty((3,3)); b = np.empty(3)
    for j in range(3):
        polynomial = np.polynomial.Polynomial([1.])
        for k in range(3):
            if k != j:
                polynomial *= np.polynomial.Polynomial([-c[k],1.])/(c[j]-c[k])
        primitive = polynomial.integ()
        a[:,j] = primitive(c)-primitive(0)
        b[j] = primitive(1)-primitive(0)
    return BlockTableau(a,b,c,None)


def prototype_scheme(family):
    tableau = prototype_tableau(family)
    def step(state,dt,f,*args,**kwargs):
        return fullyImplicitBlock(state,dt,f,tableau,*args,name=family,**kwargs)
    step.blockTableau = tableau
    return step
