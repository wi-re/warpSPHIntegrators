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
from .solvers import NonlinearSolver, FixedPointSolver, RelaxedFixedPointSolver, SolveResult
from .jfnk import JFNKSolver, fd_matvec, jvp_matvec, gmres
from .bdf import BDF1, BDF2
from .imex import IMEXEuler
from .newmark import newmark, newmark_average_acceleration, newmark_linear_acceleration

from .util import IntegrationSchemeType

from .integration import (getPreferredScheme, getIntegrator, getIntegrationEnum, IntegrationSchemes)
from .reuse import (
    ReuseAnalysis,
    step_reuse_analysis,
    step_reuse_order,
    supports_step_reuse,
    is_fsal,
)
from .util import unpack_prior_step

__version__ = "0.5.0"

__all__ = [
    "verbosePrint",
    "BaseState",
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
    "SolveResult",
    "JFNKSolver",
    "fd_matvec",
    "jvp_matvec",
    "gmres",
    "BDF1",
    "BDF2",
    "IMEXEuler",
    "newmark",
    "newmark_average_acceleration",
    "newmark_linear_acceleration",
    "StateBlend",
    "ComponentUpdateSpec",
    "PositionUpdateSpec",
    "StageResult",
    "StepEvaluation",
    "IMEXRHS",
    "IntegrationResult",
    "blend_state",
    "explicit_step",
    "semi_implicit_position_step",
    "verlet_position_step",
    "IntegrationSystem",
    "BaseIntegrationSystem",
    "IntegrationSchemeType",
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