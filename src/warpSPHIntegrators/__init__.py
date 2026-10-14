from .util import verbosePrint
from .catalog import SchemeMetadata, filter_schemes
from .tensor import TensorState, tensor_rhs

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
from .preconditioners import stage_block_preconditioner, stage_shifted_preconditioner, stage_schur_preconditioner
from .bdf import BDF1, BDF2, BDF3, BDF4, BDF5
from .multistep import AM2, AM3, AM4, ABM5
from .imex import IMEXEuler
from .imexmultistep import SBDF2, SBDF3, CNAB2, CNLF2
from .rhs import (RHS, IMEXRHS, SemilinearRHS, resolve, ResolvedRHS,
                 check_contracts, add_updates, sub_updates, CAPABILITIES)
from .newmark import newmark, newmark_average_acceleration, newmark_linear_acceleration
from .generalized_alpha import GeneralizedAlpha, GeneralizedAlphaMemory
from .collocation import Collocation, collocation_tableau
from .variable_adams import VariableAdams, adams_weights
from .stabilized import StabilizedRK, stabilized_coefficients, stabilized_interval
from .nystrom import Nystrom, NystromTableau, NystromWeights, nystrom_tableau
from .ndf import NDF, NDFQuadrature, ndf_difference_transform
from .adaptive_ndf import AdaptiveNDF, ndf_order_proposals
from .extrapolation import Extrapolation, ExtrapolationQuadrature, extrapolation_coefficients
from .switching import AlgorithmSwitch
from .dirk import (TRBDF2, MEBDF2, ESDIRK324L2SA, ESDIRK436L2SA,
                   Kvaerno3, Kvaerno4, Kvaerno5, KenCarp3, KenCarp4, KenCarp47, KenCarp5,
                   KenCarp58, ESDIRK325L2SA, ESDIRK436L2SA2, ESDIRK437L2SA, ESDIRK547L2SA2,
                   ESDIRK54I8L2SA, ESDIRK659L2SA, SSPSDIRK2, ARS222, ARS232, ARS343, ARS443,
                   BHR553, Cash4, CFNLIRK3, Hairer4, Hairer42, SFSDIRK4, SFSDIRK5, SFSDIRK6,
                   SFSDIRK7, SFSDIRK8)
from .fullyimplicit import gaussLegendre2, radauIia2, gaussLegendre3, radauIia3, BlockTableau
from .ark import ARK324L2SA, ARK436L2SA
from .rkc import RKC1, RKC2, RKL2, stage_count
from .rosenbrock import integrateROS3P as ROS3P
from .exponential import integrateETD2RK as ETD2RK, integrateEXPRB32 as EXPRB32
from .prrk import pRRK22, pRRK33, pRRK54
from .splitting import (ConservativeDissipativeRHS, compose, LieTrotter, Strang, Yoshida4,
    SplitScheme, LinearExponentialFlow, classify_separable, SeparabilityResult)
from .adaptive import estimate_error_norm, propose_dt, dormand_prince_dense_output
from .checkpoint import save_checkpoint, load_checkpoint
from .run import (solve, assess_step, snapshot_state, StepStatus, StepRecord,
                  RunSample, RunResult, RunError, RunCheckpoint, Event, EventHit)

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

from .integration import (getPreferredScheme, getPreferredIntegrator, getIntegrator, getIntegrationEnum, IntegrationSchemes)
from .reuse import (
    ReuseAnalysis,
    step_reuse_analysis,
    step_reuse_order,
    supports_step_reuse,
    is_fsal,
)
from .util import unpack_prior_step
from .drift import DriftField, drift_fields, drift_fields_active

__version__ = "0.6.1"

__all__ = [
    "NDF", "NDFQuadrature", "ndf_difference_transform",
    "AdaptiveNDF", "ndf_order_proposals",
    "Extrapolation", "ExtrapolationQuadrature", "extrapolation_coefficients",
    "AlgorithmSwitch",
    "Nystrom", "NystromTableau", "NystromWeights", "nystrom_tableau",
    "StabilizedRK", "stabilized_coefficients", "stabilized_interval",
    "MEBDF2", "CNLF2",
    "ABM5",
    "VariableAdams", "adams_weights",
    "Collocation", "collocation_tableau",
    "GeneralizedAlpha", "GeneralizedAlphaMemory",
    "gaussLegendre3", "radauIia3",
    "ConservativeDissipativeRHS", "compose", "LieTrotter", "Strang", "Yoshida4", "SplitScheme",
    "LinearExponentialFlow", "classify_separable", "SeparabilityResult",
    "stage_block_preconditioner", "stage_shifted_preconditioner", "stage_schur_preconditioner",
    "save_checkpoint", "load_checkpoint",
    "solve", "assess_step", "snapshot_state", "StepStatus", "StepRecord",
    "RunSample", "RunResult", "RunError", "RunCheckpoint", "Event", "EventHit",
    "SchemeMetadata",
    "filter_schemes",
    "TensorState",
    "tensor_rhs",
    "getPreferredIntegrator",
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
    "DriftField",
    "drift_fields",
    "drift_fields_active",
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
    # Wave 8 (SCHEME_EXPANSION_PLAN, 2026-10-13): the remaining ODE v7.8.1
    # OrdinaryDiffEqSDIRK tableaus (NOTES.md 3.28).
    "Kvaerno3",
    "Kvaerno4",
    "Kvaerno5",
    "KenCarp3",
    "KenCarp4",
    "KenCarp47",
    "KenCarp5",
    "KenCarp58",
    "ESDIRK325L2SA",
    "ESDIRK436L2SA2",
    "ESDIRK437L2SA",
    "ESDIRK547L2SA2",
    "ESDIRK54I8L2SA",
    "ESDIRK659L2SA",
    "SSPSDIRK2",
    "ARS222",
    "ARS232",
    "ARS343",
    "ARS443",
    "BHR553",
    "Cash4",
    "CFNLIRK3",
    "Hairer4",
    "Hairer42",
    "SFSDIRK4",
    "SFSDIRK5",
    "SFSDIRK6",
    "SFSDIRK7",
    "SFSDIRK8",
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
    "pRRK22",
    "pRRK33",
    "pRRK54",
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
