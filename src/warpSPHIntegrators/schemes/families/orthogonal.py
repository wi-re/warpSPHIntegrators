"""Bound ROCK/SERK/RKMC recurrences with owned full-state increment execution."""
from dataclasses import dataclass
import math
from numbers import Real
from types import SimpleNamespace


from warpSPHIntegrators.schemes.coefficients.orthogonal import CAPS, MINIMUM, available_stages, sizing_bound
from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class OrthogonalRK:
    """ROCK2/4, SERK2 or RKMC2 with fixed stages or a supplied spectral bound.

    Fixed counts round up to a source-supported configuration. lambda_max is a
    bound on negative-real spectral magnitude; exceeding the cap raises rather
    than truncating stability needs. The source selector inversion is conservative
    and numerically qualified; no nonnormal/nonlinear stability certificate.
    Increment coordinates preserve zero-rate fields. Full stages/AD graphs are
    retained; no low-storage, source spectral-estimator or FSAL policy claim.
    """
    method: str='ROCK2'
    stages: int|None=None
    lambda_max: float|None=None
    max_stages: int|None=None

    def __post_init__(self):
        if self.method not in CAPS:raise ValueError(f'method must be one of {tuple(CAPS)}')
        cap=CAPS[self.method] if self.max_stages is None else self.max_stages
        if type(cap) is not int or not MINIMUM[self.method]<=cap<=CAPS[self.method]:raise ValueError('Invalid maximum stage count')
        object.__setattr__(self,'max_stages',cap)
        if self.stages is not None and self.lambda_max is not None:raise ValueError('Choose stages or lambda_max')
        if self.lambda_max is not None:
            if isinstance(self.lambda_max,bool) or not isinstance(self.lambda_max,Real) or not math.isfinite(self.lambda_max) or self.lambda_max<0:
                raise ValueError('lambda_max must be finite and nonnegative')
        else:
            target=10 if self.stages is None else self.stages
            if type(target) is not int or target<MINIMUM[self.method]:raise ValueError('Invalid fixed stage count')
            if self.method=='ROCK4' and target in (133,152):
                raise ValueError('Pinned ROCK4 stage counts 133/152 have unstable interior bands')
            actual=next((s for s in available_stages(self.method) if s>=target and s<=cap),None)
            if actual is None:raise ValueError('Requested stages exceed the available cap')
            object.__setattr__(self,'stages',actual)

    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name=f'orthogonal_{self.method}_{self.stages}_{None if self.lambda_max is None else float(self.lambda_max).hex()}_{self.max_stages}')
    @property
    def order(self):return 4 if self.method=='ROCK4' else 2
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return True
    @property
    def estimate_order(self):return 4 if self.method=='ROCK4' else 2
    @property
    def metadata(self):return SchemeMetadata(stage_count=None if self.stages is None else self.stages+1,
        embedded_order=3 if self.method=='ROCK4' else 1,estimate_order=self.estimate_order)

    def stages_for(self,dt):
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if self.stages is not None:return self.stages
        demand=abs(dt)*self.lambda_max
        for stages in available_stages(self.method):
            if stages>self.max_stages:break
            if sizing_bound(self.method,stages)>=demand:return stages
        raise ValueError('Spectral stability requirement exceeds max_stages; reduce dt')

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.stabilized.orthogonal import step
        return step(self,*args,**kwargs)
