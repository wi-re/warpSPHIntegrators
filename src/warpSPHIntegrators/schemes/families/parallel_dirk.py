
"""Pinned parallel DIRK map executed by the serial full-state DIRK solver.

PDIRK44 is an ordinary first-order RK method, not a partitioned RKN scheme.
Two pairs of independent stages are available to a future scheduling backend;
this implementation makes no stage-threading or parallel-speed claim.
"""
from dataclasses import dataclass

from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class ParallelDIRK:
    """Fixed fourth-order L-stable PDIRK44 map, with serial stage execution.

    JFNK options/preconditioners follow DIRK. No embedded estimate, FSAL or
    source nonlinear-cache extrapolation. Physical endpoint refresh gives copied
    fields their endpoint values. Full-state quantities use the same RK weights.
    """
    @property
    def name(self):return 'Parallel DIRK (4 stages, order 4)'
    @property
    def identifier(self):return SimpleNamespace(name='pdirk44_serial')
    @property
    def order(self):return 4
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def metadata(self):return SchemeMetadata(stage_count=5)
    @property
    def stage_groups(self):return ((0,1),(2,3))

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.parallel_dirk import step
        return step(self,*args,**kwargs)
