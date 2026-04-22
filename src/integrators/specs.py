from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


StepSize = Union[float, List[float]]


@dataclass(frozen=True)
class StateBlend:
    self_scale: Optional[float] = None
    reference_state: Optional[Any] = None
    reference_weight: Optional[float] = None


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


@dataclass(frozen=True)
class StepEvaluation:
    update: Any
    aux: Any = None


@dataclass(frozen=True)
class IntegrationResult:
    state: Any
    evaluations: List[StepEvaluation]


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