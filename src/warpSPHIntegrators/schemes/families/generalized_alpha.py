
"""Classical second-order generalized-alpha in explicit acceleration form.

Old-state weights follow Chung & Hulbert (1993). The nonlinear unknown includes
algorithmic acceleration; recomputing it from endpoint force loses order. This
parameterized, fixed-step factory is intentionally outside the fixed catalog.
"""
from dataclasses import dataclass
import math
from numbers import Real
from types import SimpleNamespace
from typing import ClassVar

import torch

from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class GeneralizedAlphaMemory:
    """Accepted algorithmic acceleration carried in HistoryEntry.state.

    In-memory runner checkpoints preserve this value and its gradient graph.
    Application serialization is required for custom second-order systems.
    """
    acceleration: torch.Tensor
    t: float
    rho_inf: float
    layout: tuple
    uid: object = None


@dataclass(frozen=True)
class GeneralizedAlpha:
    """Fixed-step mechanics driver with controllable high-frequency damping.

    ``rho_inf`` is the limiting spectral radius in [0, 1]; 1 recovers average
    acceleration Newmark for linear forces. RHS must return x'=v and explicit acceleration for
    a constant-mass second-order system. Nonlinear, velocity-dependent and
    time-dependent forces are allowed. Other integrated quantities use a
    coupled endpoint trapezoid. Custom finalize must preserve integrated values.

    Use ``solve(GeneralizedAlpha(.5), ...)`` to own acceleration memory, or feed
    every accepted ``result.history`` into the next direct call. Starting with
    empty history initializes acceleration from the physical RHS exactly once.
    There is no estimator, dense output, event localization or priorStep reuse.
    """
    rho_inf: float = .5
    name: ClassVar[str] = 'Generalized-alpha'
    order: ClassVar[int] = 2
    implicit: ClassVar[bool] = True
    fsal: ClassVar[bool] = False
    has_error_estimate: ClassVar[bool] = False

    def __post_init__(self):
        if (not isinstance(self.rho_inf, Real) or isinstance(self.rho_inf, bool)
                or not math.isfinite(self.rho_inf) or not 0 <= self.rho_inf <= 1):
            raise ValueError('rho_inf must be finite and lie in [0, 1]')

    @property
    def identifier(self):
        return SimpleNamespace(name='generalized_alpha_' + float(self.rho_inf).hex())

    @property
    def metadata(self):
        return SchemeMetadata(state_kind='second_order', history_length=1,
                              history_requires_constant_dt=False)

    @property
    def parameters(self):
        rho = self.rho_inf
        am, af = (2*rho-1)/(rho+1), rho/(rho+1)
        return am, af, .5-am+af, (1-am+af)**2/4  # am, af, gamma, beta

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.mechanics.generalized_alpha import step
        return step(self,*args,**kwargs)
