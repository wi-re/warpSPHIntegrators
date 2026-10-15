"""Improved explicit Nyström history with corrected, physically timed startup.

The warm recurrences follow the pinned IRKN3/4 mutable kernels. Startup uses
Nyström's fourth-order velocity-independent method, then seeds the old physical
acceleration and internal nodes explicitly; it does not copy the source seed bug.
"""
from dataclasses import dataclass

from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata
from warpSPHIntegrators.schemes.families.nystrom import Nystrom


@dataclass(frozen=True)
class ImprovedNystromWeights:
    """Quadrature paired with stages, including the explicitly retained old rates.

    q1=q0+h*(initial_velocity*v0+previous_velocity_weight*previous_velocity)
       +h²*sum(position[i]*stages[i].update.acceleration).
    v1=v0+h*sum(velocity[i]*stages[i].update.acceleration).
    Extra integrated fields are passive. Endpoint refresh has zero weight.
    """
    position: tuple
    velocity: tuple
    quantity: tuple
    initial_velocity: float
    previous_velocity_weight: float
    previous_velocity: object = None


@dataclass(frozen=True)
class ImprovedNystrom:
    """Fixed-step IRKN3/4 for q'=v, velocity-independent force and passive extras.

    Corrected startup is deliberately distinct from the faulty pinned source
    startup. Feed accepted result.history to direct calls or let solve own it.
    State/time/layout/UID/step changes restart; changed RHS parameters require
    restart_history=True. Custom finalize must opt into irkn_compatible and
    preserve integrated fields. Supports in-memory mechanical checkpoints.
    """
    order: int = 4
    structure: str = 'probe'
    structure_atol: float = 1e-10

    def __post_init__(self):
        if type(self.order) is not int or self.order not in (3,4):
            raise ValueError('Improved Nyström order must be 3 or 4')
        Nystrom('Nystrom4VelocityIndependent',self.structure,self.structure_atol)

    @property
    def name(self):return f'Improved Nyström {self.order} (corrected startup)'
    @property
    def identifier(self):return SimpleNamespace(name=f'improved_nystrom_{self.order}_{self.structure}_{float(self.structure_atol).hex()}')
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def metadata(self):return SchemeMetadata(state_kind='second_order',requires_separable=True,history_length=1)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.mechanics.improved_nystrom import step
        return step(self,*args,**kwargs)
