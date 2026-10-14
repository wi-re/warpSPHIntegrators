"""Qualified source recurrences expanded into ordinary full-stage RK maps.

These methods come from a low-storage source family. This backend deliberately
uses the existing full-stage driver and does not claim a low-storage execution.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
import difflib
import math
from types import SimpleNamespace
import numpy as np
from ._expanded_rk_data import _METHODS
from .butcher import butcherTableau,RungeKuttaB
from .catalog import SchemeMetadata
from .util import reject_prior_step


def expanded_rk_methods():
    """Qualified source method names, sorted; unsupported source pins are omitted."""
    return tuple(sorted(_METHODS))


def _array(values):
    array=np.asarray(values,dtype=np.float64)
    return np.frombuffer(array.tobytes(),dtype=np.float64).reshape(array.shape)


@lru_cache(maxsize=64)
def expanded_rk_tableau(method):
    """Immutable full-stage coefficients, including physical endpoint refresh.

    Nineteen source companions pass the independently checked order gates. The
    pinned CKLLSRK85_4FM_4R main map passes, but its advertised companion does not;
    that configuration is deliberately fixed-step only in this backend.
    """
    if not isinstance(method,str) or method not in _METHODS:
        hint=difflib.get_close_matches(method,expanded_rk_methods(),n=3) if isinstance(method,str) else ()
        raise ValueError(f'Unknown qualified expanded RK method {method!r}; closest: {tuple(hint)}')
    data=_METHODS[method];main=_array(data['b']);error=data['error_weights']
    weights=main if error is None else (main,_array(np.asarray(data['b'])-np.asarray(error)))
    return butcherTableau(_array(data['a']),weights,_array(data['c']))


@dataclass(frozen=True)
class ExpandedRK:
    """Owned standard-RK execution of 54 screened low-storage source main maps.

    The source recurrences are expanded, duplicate initial stages merged and a
    physical endpoint reconstructed. Coefficients remain immutable. Source
    cache/FSAL/fused-storage/controller behavior is separate. Finalization uses
    the standard RK stageWeights, stageIndex and compatible-energy rerun path.
    """
    method: str='CarpenterKennedy2N54'
    def __post_init__(self):expanded_rk_tableau(self.method)
    @property
    def name(self):return f'Expanded RK {self.method}'
    @property
    def identifier(self):return SimpleNamespace(name='expanded_rk_'+self.method)
    @property
    def order(self):return _METHODS[self.method]['order']
    @property
    def tableau(self):return expanded_rk_tableau(self.method)
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def estimate_order(self):return _METHODS[self.method]['estimate_order']
    @property
    def has_error_estimate(self):return self.estimate_order is not None
    @property
    def embedded_order(self):return self.estimate_order-1 if self.has_error_estimate else None
    @property
    def metadata(self):return SchemeMetadata(stage_count=len(self.tableau.c),
        embedded_order=self.embedded_order,estimate_order=self.estimate_order)

    def __call__(self,state,dt,f,*args,**kwargs):
        from .run import snapshot_state
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        reject_prior_step(self.name,kwargs.pop('priorStep',None))
        options=dict(kwargs)
        if options.get('history') is not None:options['history']=snapshot_state(options['history'])
        result=RungeKuttaB(snapshot_state(state),dt,f,self.tableau,*args,**options)
        return replace(result,estimate_order=self.estimate_order)
