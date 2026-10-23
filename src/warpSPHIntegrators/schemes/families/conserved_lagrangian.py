"""Conserved moving-quadrature stepping with system-owned compatibility hooks."""
from dataclasses import dataclass
from types import SimpleNamespace
from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class ConservedLagrangian:
    """Hancock, SSPRK2 or KDK with the historical moving-quadrature drift.

    A system supplies tagged position/conserved fields, conserved_velocity(Q)
    and wrap_positions(x). Normal initialize/preprocess/postprocess/finalize
    hooks prepare geometry, derive primitives and apply compatible corrections.
    Hancock's RHS predicts over h; SSPRK/KDK evaluate unpredicted rates (h=0).
    These position predictors differ from applying ordinary Heun to (x,Q).
    No spatial scheme or conserved-vector layout is assumed by this package.
    Order depends on the supplied prediction/kick contract, so no universal
    ODE order is advertised for these consumer-specific maps.
    """
    temporal: str = 'hancock'

    def __post_init__(self):
        if self.temporal not in ('hancock', 'ssprk', 'kdk'):
            raise ValueError('temporal must be hancock, ssprk or kdk')

    @property
    def name(self): return f'Conserved Lagrangian {self.temporal}'
    @property
    def identifier(self): return SimpleNamespace(name=f'conserved_lagrangian_{self.temporal}_v1')
    @property
    def order(self): return None
    @property
    def implicit(self): return False
    @property
    def fsal(self): return False
    @property
    def has_error_estimate(self): return False
    @property
    def estimate_order(self): return None
    @property
    def metadata(self): return SchemeMetadata(stage_count=1 if self.temporal == 'hancock' else 2, state_kind='conserved_lagrangian')
    @property
    def function(self): return self.__call__

    def __call__(self, state, dt, f, *args, **kwargs):
        from warpSPHIntegrators.drivers.mechanics.conserved_lagrangian import step
        return step(self, state, dt, f, *args, **kwargs)
