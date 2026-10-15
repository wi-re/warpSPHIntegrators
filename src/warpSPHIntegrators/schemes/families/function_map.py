

"""Explicit discrete state maps with endpoint clock and owned lifecycle."""
from dataclasses import dataclass

from types import SimpleNamespace

from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class MapTransition:
    """A discrete transition, not derivative quadrature or an error estimate."""
    input_state: object
    evaluation_time: float
    mapped_state: object


@dataclass(frozen=True)
class FunctionMap:
    """Apply one declared StateMap per fixed step at the source's t+h clock.

    h advances the clock and is available to the map. This has no continuous
    ODE order or error estimate. Source scale_by_time behavior is expressible
    as tensor_map(lambda t,y,h:y+h*rate(t,y)); no overloaded RHS flag is needed.
    """
    @property
    def name(self):return 'Function map'
    @property
    def identifier(self):return SimpleNamespace(name='function_map_v1')
    @property
    def order(self):return None
    @property
    def implicit(self):return False
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def estimate_order(self):return None
    @property
    def metadata(self):return SchemeMetadata(stage_count=1)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.maps.function_map import step
        return step(self,*args,**kwargs)
