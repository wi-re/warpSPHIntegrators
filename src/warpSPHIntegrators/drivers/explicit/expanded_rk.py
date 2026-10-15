"""Numerical stepping for expanded_rk; configuration and coefficients are supplied separately."""
from dataclasses import replace


import math


from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB


from warpSPHIntegrators.state.lifecycle import reject_prior_step


def step(self,state,dt,f,*args,**kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state
    if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
    reject_prior_step(self.name,kwargs.pop('priorStep',None))
    options=dict(kwargs)
    if options.get('history') is not None:options['history']=snapshot_state(options['history'])
    result=RungeKuttaB(snapshot_state(state),dt,f,self.tableau,*args,**options)
    return replace(result,estimate_order=self.estimate_order)
