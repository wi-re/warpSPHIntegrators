"""Qualified source recurrences expanded into ordinary full-stage RK maps.

These methods come from a low-storage source family. This backend deliberately
uses the existing full-stage driver and does not claim a low-storage execution.
"""

from warpSPHIntegrators.schemes.coefficients.expanded import expanded_rk_tableau
from dataclasses import dataclass


from types import SimpleNamespace

from warpSPHIntegrators.schemes.coefficients.expanded_data import _METHODS

from warpSPHIntegrators.schemes.metadata import SchemeMetadata


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

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.explicit.expanded_rk import step
        return step(self,*args,**kwargs)
