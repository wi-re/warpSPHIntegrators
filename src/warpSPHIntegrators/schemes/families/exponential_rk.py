
"""Operator-valued exponential RK maps; bounded dense AD and matrix-free actions."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import torch
from warpSPHIntegrators.schemes.coefficients.exponential import PROGRAMS
from warpSPHIntegrators.schemes.metadata import SchemeMetadata

SEMILINEAR=('LawsonEuler','NorsettEuler','ETDRK2','ETDRK3','ETDRK4','HochOst4')
ORDERS=dict(LawsonEuler=1,NorsettEuler=1,ETDRK2=2,ETDRK3=3,ETDRK4=4,HochOst4=4,
    Exp4=4,EPIRK4s3A=4,EPIRK4s3B=4,EPIRK5P1=5,EPIRK5P2=5,EXPRB53s3=5,Exprb32=3,Exprb43=4)


@dataclass(frozen=True)
class ExponentialQuadrature:
    """Operator-weighted increments, rather than scalar weights over raw rates."""
    initial_increment: torch.Tensor
    increments: tuple
    field_names: tuple


@dataclass(frozen=True)
class ExponentialRK:
    """Source-qualified semilinear or full-Jacobian exponential maps.

    Semilinear methods require a constant, state-independent linear RHS accessor.
    Linearized methods augment time and use a full-JVP frozen Jacobian, with
    isolated central f_t probes (or exact f_t). Dense actions for <=dense_limit
    entries retain complete operator derivatives. Larger constant operators use
    Arnoldi and transpose adjoints; tracked operator derivatives beyond that
    bound are rejected. phi_dim is a budget, not an accuracy certificate: large
    actions require an invariant Krylov space or dimension-refinement agreement.
    Source EPIRK5s3 fails the order gate and is deliberately unavailable.
    No source FSAL/controller, mass matrix or arbitrary nonlinear projector.
    """
    method: str='ETDRK4'
    phi_dim: int=60
    phi_tol: float=1e-11
    dense_limit: int=64

    def __post_init__(self):
        if self.method not in ORDERS:raise ValueError(f'Choose from {tuple(ORDERS)}')
        if type(self.phi_dim) is not int or self.phi_dim<2:raise ValueError('phi_dim must be an integer >=2')
        if type(self.dense_limit) is not int or self.dense_limit<0:raise ValueError('dense_limit must be nonnegative')
        if isinstance(self.phi_tol,bool) or not math.isfinite(self.phi_tol) or self.phi_tol<=0:raise ValueError('positive finite phi_tol required')
    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name=f'exponential_{self.method}_{self.phi_dim}_{float(self.phi_tol).hex()}_{self.dense_limit}')
    @property
    def order(self):return ORDERS[self.method]
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return PROGRAMS[self.method]['error'] is not None
    @property
    def estimate_order(self):return {'Exprb32':3,'Exprb43':4}.get(self.method)
    @property
    def metadata(self):return SchemeMetadata(stage_count=len(PROGRAMS[self.method]['calls']),
        estimate_order=self.estimate_order,embedded_order=self.order-1 if self.has_error_estimate else None,
        required_capabilities=frozenset({'linear'}) if self.method in SEMILINEAR else frozenset())

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.exponential.exponential_rk import step
        return step(self,*args,**kwargs)
