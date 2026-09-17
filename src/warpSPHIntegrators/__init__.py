from .util import verbosePrint

from .fields import *
from .integration import *
from .specs import *

from .util import IntegrationScheme, updateStateEuler, updateStateSemiImplicitEuler
from .util import applyStateUpdate, applyPositionUpdate, applyVelocityUpdate, applyQuantityUpdate
from .specs import (
    StateBlend,
    ComponentUpdateSpec,
    PositionUpdateSpec,
    StageResult,
    StepEvaluation,
    IntegrationResult,
    blend_state,
    explicit_step,
    semi_implicit_position_step,
    verlet_position_step,
)
from .protocol import IntegrationSystem, BaseIntegrationSystem
from .fields import (
    BaseState,
    BlockState,
    integrated,
    constant,
    copied,
    ephemeral,
    custom,
    tagged,
    reference_state,
    find_tagged_field,
    get_tagged_attr,
    set_tagged_attr,
    get_reference_state,
    clone_value,
    empty_value,
    move_value,
    register_clone_handler,
    CloneHandler,
    state_difference,
    state_norm,
    integrated_field_names,
    flatten_integrated,
    unflatten_integrated,
    replace_integrated_fields,
)
from .history import HistoryEntry, StepHistory
from .solvers import (NonlinearSolver, FixedPointSolver, RelaxedFixedPointSolver,
                     SolveDiagnostics, SolveResult, SolverOptions)
from .jfnk import (JFNKSolver, fd_matvec, jvp_matvec, gmres,
                  identity_preconditioner, diagonal_preconditioner)
from .bdf import BDF1, BDF2, BDF3, BDF4, BDF5
from .multistep import AM2, AM3, AM4
from .imex import IMEXEuler
from .imexmultistep import SBDF2, SBDF3, CNAB2
from .rhs import (RHS, IMEXRHS, SemilinearRHS, resolve, ResolvedRHS,
                 check_contracts, add_updates, sub_updates, CAPABILITIES)
from .newmark import newmark, newmark_average_acceleration, newmark_linear_acceleration
from .dirk import TRBDF2, ESDIRK324L2SA, ESDIRK436L2SA
from .fullyimplicit import gaussLegendre2, radauIia2, BlockTableau
from .ark import ARK324L2SA, ARK436L2SA
from .rkc import RKC1, RKC2, RKL2, stage_count
from .rosenbrock import integrateROS3P as ROS3P
from .exponential import integrateETD2RK as ETD2RK, integrateEXPRB32 as EXPRB32
from .adaptive import estimate_error_norm, propose_dt, dormand_prince_dense_output

from .util import IntegrationSchemeType
from .enums import (
    ExplicitRK,
    Symplectic,
    RelaxedChebyshev,
    MultistepExplicit,
    DIRK,
    CoupledRK,
    MultistepImplicit,
    IMEX,
    LinearlyImplicit,
    Exponential,
    Newmark,
    FAMILY_ENUMS,
    SCHEME_FAMILY,
)

from .integration import (getPreferredScheme, getIntegrator, getIntegrationEnum, IntegrationSchemes)
from .reuse import (
    ReuseAnalysis,
    step_reuse_analysis,
    step_reuse_order,
    supports_step_reuse,
    is_fsal,
)
from .util import unpack_prior_step

__version__ = "0.6.0"

__all__ = [
    "verbosePrint",
    "BaseState",
    "BlockState",
    "integrated",
    "constant",
    "copied",
    "ephemeral",
    "custom",
    "tagged",
    "reference_state",
    "find_tagged_field",
    "get_tagged_attr",
    "set_tagged_attr",
    "get_reference_state",
    "clone_value",
    "empty_value",
    "move_value",
    "register_clone_handler",
    "CloneHandler",
    "state_difference",
    "state_norm",
    "integrated_field_names",
    "flatten_integrated",
    "unflatten_integrated",
    "replace_integrated_fields",
    "HistoryEntry",
    "StepHistory",
    "NonlinearSolver",
    "FixedPointSolver",
    "RelaxedFixedPointSolver",
    "SolveDiagnostics",
    "SolveResult",
    "SolverOptions",
    "JFNKSolver",
    "fd_matvec",
    "jvp_matvec",
    "gmres",
    "identity_preconditioner",
    "diagonal_preconditioner",
    "BDF1",
    "BDF2",
    "BDF3",
    "BDF4",
    "BDF5",
    "IMEXEuler",
    "SBDF2",
    "SBDF3",
    "CNAB2",
    "AM2",
    "AM3",
    "AM4",
    "newmark",
    "newmark_average_acceleration",
    "newmark_linear_acceleration",
    "TRBDF2",
    "ESDIRK324L2SA",
    "ESDIRK436L2SA",
    "gaussLegendre2",
    "radauIia2",
    "BlockTableau",
    "ARK324L2SA",
    "ARK436L2SA",
    "RKC1",
    "RKC2",
    "RKL2",
    "stage_count",
    "ROS3P",
    "ETD2RK",
    "EXPRB32",
    "estimate_error_norm",
    "propose_dt",
    "dormand_prince_dense_output",
    "StateBlend",
    "ComponentUpdateSpec",
    "PositionUpdateSpec",
    "StageResult",
    "StepEvaluation",
    "RHS",
    "IMEXRHS",
    "SemilinearRHS",
    "resolve",
    "ResolvedRHS",
    "check_contracts",
    "add_updates",
    "sub_updates",
    "CAPABILITIES",
    "IntegrationResult",
    "blend_state",
    "explicit_step",
    "semi_implicit_position_step",
    "verlet_position_step",
    "IntegrationSystem",
    "BaseIntegrationSystem",
    "IntegrationSchemeType",
    "ExplicitRK",
    "Symplectic",
    "RelaxedChebyshev",
    "MultistepExplicit",
    "DIRK",
    "CoupledRK",
    "MultistepImplicit",
    "IMEX",
    "LinearlyImplicit",
    "Exponential",
    "Newmark",
    "FAMILY_ENUMS",
    "SCHEME_FAMILY",
    "getPreferredScheme",
    "getIntegrator",
    "getIntegrationEnum",
    "IntegrationSchemes",
    "IntegrationScheme",
    "ReuseAnalysis",
    "step_reuse_analysis",
    "step_reuse_order",
    "supports_step_reuse",
    "is_fsal",
    "unpack_prior_step",
    "updateStateEuler",
    "updateStateSemiImplicitEuler",
    "applyStateUpdate",
    "applyPositionUpdate",
    "applyVelocityUpdate",
    "applyQuantityUpdate",
    "update_position",
    "update_component"
]