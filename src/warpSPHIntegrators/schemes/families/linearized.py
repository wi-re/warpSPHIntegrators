"""Source-qualified Rosenbrock maps with exact Jacobian actions and adjoints."""

from warpSPHIntegrators.schemes.coefficients.rosenbrock import rosenbrock_tableau
from dataclasses import dataclass


from types import SimpleNamespace


from warpSPHIntegrators.schemes.metadata import SchemeMetadata


@dataclass(frozen=True)
class RosenbrockQuadrature:
    """Weights over flat, already h-scaled solved increments, paired with stages.

    flatten(output)=flatten(initial)+sum(weights[i]*increments[i]).
    Raw stage updates are physical rates, not the solved increments. Endpoint
    refresh has a zero increment and weight. Names declare the flattening order.
    """
    weights: tuple
    increments: tuple
    field_names: tuple


@dataclass(frozen=True)
class Rosenbrock:
    """Rosenbrock23/32 or W6S4OS using exact full-RHS JVP and linear GMRES stages.

    Rosenbrock23/32 correct the pinned third-stage time coefficient by autonomous
    augmentation; this is separate from unchanged autonomous-source equivalence.
    No mass matrices, approximate-W substitution, FSAL or source cache controller.
    Default f_t uses isolated central time probes; a callable f_t returns an
    exact partial-time update. solver_opts accepts GMRES/preconditioner controls.
    Implicit adjoints retain derivatives through J, including Hessian and RHS-
    parameter terms. Custom finalize must opt into rosenbrock_compatible and
    consume RosenbrockQuadrature, whose increments differ from raw RHS rates.
    """
    method: str='Rosenbrock23'

    def __post_init__(self):rosenbrock_tableau(self.method)
    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name='rosenbrock_full_jvp_'+self.method)
    @property
    def order(self):return rosenbrock_tableau(self.method).order
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return rosenbrock_tableau(self.method).btilde is not None
    @property
    def estimate_order(self):return 3 if self.has_error_estimate else None
    @property
    def metadata(self):return SchemeMetadata(stage_count=len(rosenbrock_tableau(self.method).b)+1,
        embedded_order=(3 if self.order==2 else 2) if self.has_error_estimate else None,
        estimate_order=self.estimate_order)

    def __call__(self,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.linearized import step
        return step(self,*args,**kwargs)
