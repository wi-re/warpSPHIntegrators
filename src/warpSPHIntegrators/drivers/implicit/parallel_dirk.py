"""Numerical stepping for parallel_dirk; configuration and coefficients are supplied separately."""
from dataclasses import replace

import math


import numpy as np


from warpSPHIntegrators.drivers.implicit.dirk import DIRK

from warpSPHIntegrators.state.specs import StageResult

from warpSPHIntegrators.state.lifecycle import finalizeSystem, updateStep


from warpSPHIntegrators.schemes.coefficients.parallel_dirk import parallel_dirk_tableau

def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state, assess_step
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    base=snapshot_state(state)
    if not math.isfinite(float(base.t)):raise ValueError('finite initial time required')
    tableau=parallel_dirk_tableau()
    result=DIRK(base,dt,f,tableau,*args,name=self.name,_finalize=False,**kwargs)
    weights=np.r_[tableau.b,0.]
    if not assess_step(result).successful:return replace(result,quadrature=tableau.b.copy())
    endpoint=result.state.initializeNewState(*args,**kwargs);endpoint.t=float(base.t+dt)
    update,aux=updateStep(base,endpoint,dt,f,*args,**kwargs)
    stages=[*result.stages,StageResult(aux,update)]
    result=replace(result,stages=stages,quadrature=weights)
    if assess_step(result).successful:
        finalizeSystem(result.state,base,dt,[s.aux for s in stages],[s.update for s in stages],weights,
            *args,lastStageSystem=endpoint,**kwargs)
    return result
