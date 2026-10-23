"""A single evaluation of a system-owned projected state map."""

from dataclasses import dataclass
from types import SimpleNamespace
from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class ProjectedStep:
    """Execute one full projected map with the ordinary lifecycle hooks.

    The consumer's map advances its stage state using projection temporal
    primitives. It returns an auxiliary update for diagnostics/compatibility;
    the driver does not integrate that update a second time. The system must
    declare projected_map=True and finalize the advanced stage into output.
    This is a consumer map, with no universal ODE order claim.
    """

    @property
    def name(self):
        return "Projected state map"

    @property
    def identifier(self):
        return SimpleNamespace(name="projected_state_map_v1")

    @property
    def order(self):
        return None

    @property
    def implicit(self):
        return False

    @property
    def fsal(self):
        return False

    @property
    def has_error_estimate(self):
        return False

    @property
    def estimate_order(self):
        return None

    @property
    def metadata(self):
        return SchemeMetadata(stage_count=1, state_kind="projected_map")

    @property
    def function(self):
        return self.__call__

    def __call__(self, state, dt, f, *args, **kwargs):
        from warpSPHIntegrators.drivers.mechanics.projection import step

        return step(self, state, dt, f, *args, **kwargs)
