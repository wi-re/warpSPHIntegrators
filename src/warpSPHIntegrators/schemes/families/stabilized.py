"""Stage-bound Legendre/Gegenbauer stabilized explicit RK backends.

Recurrence stage times match their first-order consistency relation. The Euler
comparison is unscaled (the source controller divides its norm by stage count).
Every step refreshes the physical endpoint, costing s+1 RHS evaluations. Stored
derivatives and autograd graphs mean this is not a low-storage execution path.
"""

from warpSPHIntegrators.schemes.coefficients.stabilized import _MINIMUM, stabilized_interval
from dataclasses import dataclass

import math
from numbers import Real
from types import SimpleNamespace

from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class StabilizedRK:
    """Bind RKL1/2 or RKG1/2 and fixed stages or a supplied spectral bound.

    Default: five stages. ``lambda_max=...`` selects stages per attempted h;
    exceeding max_stages raises instead of silently truncating the stability
    requirement. Fixed Legendre counts must be odd, matching source selection.
    No source spectral estimator, FSAL, dense output or low-storage claim.
    """
    family: str='rkg2'
    stages: int|None=None
    lambda_max: float|None=None
    max_stages: int=1000

    def __post_init__(self):
        if self.family not in _MINIMUM:raise ValueError('family must be rkl1, rkl2, rkg1 or rkg2')
        if type(self.max_stages) is not int or self.max_stages<_MINIMUM[self.family]:
            raise ValueError('max_stages must be an integer above the method minimum')
        if self.stages is None and self.lambda_max is None:object.__setattr__(self,'stages',5)
        if self.stages is not None:
            stabilized_interval(self.family,self.stages)
            if self.stages>self.max_stages:raise ValueError('stages exceeds max_stages')
            if self.family.startswith('rkl') and self.stages%2==0:raise ValueError('Legendre stages must be odd')
            if self.lambda_max is not None:raise ValueError('supply stages or lambda_max, not both')
        else:
            if isinstance(self.lambda_max,bool) or not isinstance(self.lambda_max,Real) or not math.isfinite(self.lambda_max) or self.lambda_max<0:
                raise ValueError('lambda_max must be finite and nonnegative')
            object.__setattr__(self,'lambda_max',float(self.lambda_max))

    def stage_count(self,dt):
        if isinstance(dt,bool) or not isinstance(dt,Real) or not math.isfinite(dt):
            raise ValueError('finite real dt required')
        if self.stages is not None:return self.stages
        magnitude=abs(dt)*self.lambda_max
        s=_MINIMUM[self.family]
        while stabilized_interval(self.family,s)<magnitude:
            s+=2 if self.family.startswith('rkl') else 1
            if s>self.max_stages:raise ValueError('spectral bound requires more than max_stages')
        return s

    @property
    def order(self):return int(self.family[-1])
    @property
    def name(self):return f'{self.family.upper()} (s={self.stages})' if self.stages is not None else f'{self.family.upper()} (lambda_max={self.lambda_max})'
    @property
    def identifier(self):return SimpleNamespace(name=f'stabilized_{self.family}_{self.stages}_{self.lambda_max}_{self.max_stages}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 2
    @property
    def metadata(self):
        return SchemeMetadata(stage_count=self.stages+1 if self.stages is not None else None,
            embedded_order=1,estimate_order=2)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.stabilized.stabilized import step
        return step(self,*args,**kwargs)
