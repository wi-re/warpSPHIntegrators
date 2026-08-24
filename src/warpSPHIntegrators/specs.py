from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, NamedTuple


StepSize = Union[float, List[float]]


@dataclass(frozen=True)
class StateBlend:
    self_scale: Optional[float] = None
    reference_state: Optional[Any] = None
    reference_weight: Optional[float] = None

    def __post_init__(self):
        if (self.reference_state is None) != (self.reference_weight is None):
            raise ValueError(
                "StateBlend requires reference_state and reference_weight together; got "
                f"reference_state={'set' if self.reference_state is not None else 'None'}, "
                f"reference_weight={self.reference_weight!r}. A reference state without a "
                "weight fails much later, inside update_component, as None * tensor."
            )


@dataclass(frozen=True)
class ComponentUpdateSpec:
    derivative_dt: StepSize
    blend: StateBlend = field(default_factory=StateBlend)


@dataclass(frozen=True)
class PositionUpdateSpec:
    derivative_dt: StepSize
    current_velocity_dt: Optional[float] = None
    update_velocity_dt: Optional[float] = None
    blend: StateBlend = field(default_factory=StateBlend)

    def __post_init__(self):
        if self.current_velocity_dt is not None and self.update_velocity_dt is not None:
            raise ValueError(
                "PositionUpdateSpec cannot combine current_velocity_dt and update_velocity_dt in compatibility mode"
            )
        if self.current_velocity_dt is not None and self.derivative_dt != 0.0:
            raise ValueError(
                f"PositionUpdateSpec with current_velocity_dt (semi-implicit mode) requires derivative_dt=0.0, "
                f"but got {self.derivative_dt}. Use semi_implicit_position_step(dt) to construct this spec."
            )


class StageResult(NamedTuple):
    """Result from a single RK stage: paired auxiliary value and update/derivative."""
    aux: Any = None
    update: Any = None


@dataclass(frozen=True)
class StepEvaluation:
    update: Any
    aux: Any = None


@dataclass(frozen=True)
class IntegrationResult:
    """Result from a full integration step.

    Attributes:
        state: The final integrated state after the step.
        stages: List of StageResult, one per RK stage. Each contains the auxiliary
                return value (aux) and the k-value (update) from that stage.
                Access the last stage with stages[-1] to get (aux, k) for priorStep.
        error:  For embedded pairs only, otherwise None. A state of the same type as
                `state` whose integrated fields hold the difference between the
                propagated solution and the lower-order embedded one. Drives step
                size control; the initial state cancels, so this is a pure
                difference, not a state you can continue integrating from.
        history: Set only by schemes that were passed a ``history=`` kwarg (see
                `warpSPHIntegrators.history.StepHistory`); `None` otherwise. Feed it
                back in as `history=` on the next step to keep the run threaded.
    """
    state: Any
    stages: List[StageResult] = field(default_factory=list)
    error: Optional[Any] = None
    history: Optional[Any] = None


def blend_state(
    *,
    self_scale: Optional[float] = None,
    reference_state: Optional[Any] = None,
    reference_weight: Optional[float] = None,
) -> StateBlend:
    return StateBlend(
        self_scale=self_scale,
        reference_state=reference_state,
        reference_weight=reference_weight,
    )


def explicit_step(dt: StepSize, *, blend: Optional[StateBlend] = None) -> ComponentUpdateSpec:
    return ComponentUpdateSpec(derivative_dt=dt, blend=blend or StateBlend())


def semi_implicit_position_step(
    dt: float,
    *,
    blend: Optional[StateBlend] = None,
) -> PositionUpdateSpec:
    return PositionUpdateSpec(
        derivative_dt=0.0,
        current_velocity_dt=dt,
        blend=blend or StateBlend(),
    )


def verlet_position_step(
    derivative_dt: StepSize,
    *,
    update_velocity_dt: float,
    blend: Optional[StateBlend] = None,
) -> PositionUpdateSpec:
    return PositionUpdateSpec(
        derivative_dt=derivative_dt,
        update_velocity_dt=update_velocity_dt,
        blend=blend or StateBlend(),
    )