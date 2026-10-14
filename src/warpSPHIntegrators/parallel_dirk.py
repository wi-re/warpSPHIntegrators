"""Pinned parallel DIRK map executed by the serial full-state DIRK solver.

PDIRK44 is an ordinary first-order RK method, not a partitioned RKN scheme.
Two pairs of independent stages are available to a future scheduling backend;
this implementation makes no stage-threading or parallel-speed claim.
"""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import numpy as np

from .butcher import butcherTableau
from .catalog import SchemeMetadata
from .dirk import DIRK
from .specs import StageResult
from .util import finalizeSystem, updateStep


def parallel_dirk_tableau():
    """PDIRK 4-stage, order-four tableau; stages (0,1) and (2,3) are independent."""
    return butcherTableau(a=np.array([[.5,0.,0.,0.],[0.,2/3,0.,0.],
        [-5/2,5/2,.5,0.],[-5/3,4/3,0.,2/3]]),
        b=np.array([-1.,1.5,-1.,1.5]),c=np.array([.5,2/3,.5,1/3]))


@dataclass(frozen=True)
class ParallelDIRK:
    """Fixed fourth-order L-stable PDIRK44 map, with serial stage execution.

    JFNK options/preconditioners follow DIRK. No embedded estimate, FSAL or
    source nonlinear-cache extrapolation. Physical endpoint refresh gives copied
    fields their endpoint values. Full-state quantities use the same RK weights.
    """
    @property
    def name(self):return 'Parallel DIRK (4 stages, order 4)'
    @property
    def identifier(self):return SimpleNamespace(name='pdirk44_serial')
    @property
    def order(self):return 4
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def metadata(self):return SchemeMetadata(stage_count=5)
    @property
    def stage_groups(self):return ((0,1),(2,3))

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state, assess_step
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
