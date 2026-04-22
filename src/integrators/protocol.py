"""
Integration system protocol and base mixin.

A user-facing system only needs to implement two things:
  1. ``initializeNewState`` — return a fresh copy of the current state as a
     stage buffer for intermediate calculations.
  2. One of the three targeted update methods:

     Legacy protocol (backwards-compatible):
       * ``integratePosition(update, dt, *, selfScale, semiImplicitScale,
                            verletScale, referenceState, referenceWeight)``
       * ``integrateVelocity(update, dt, *, selfScale, referenceState,
                            referenceWeight)``
       * ``integrateQuantities(update, dt, *, selfScale, referenceState,
                              referenceWeight)``
       * ``integrate(update, dt)``  — combined fallback used by Euler / Butcher paths

     Typed protocol (preferred for new code):
       * ``apply_position_update(update, spec: PositionUpdateSpec)``
       * ``apply_velocity_update(update, spec: ComponentUpdateSpec)``
       * ``apply_quantity_update(update, spec: ComponentUpdateSpec)``
       * ``apply_state_update(update, spec: ComponentUpdateSpec)``

     If the typed methods are present they take priority; otherwise the library
     falls back to the legacy method names.

All lifecycle hooks (``initialize``, ``preprocess``, ``postprocess``,
``finalize``) are optional — subclass ``BaseIntegrationSystem`` to get no-op
defaults so you only override what you need.
"""

from typing import Any, List, Protocol, Union, runtime_checkable

from .specs import ComponentUpdateSpec, PositionUpdateSpec


@runtime_checkable
class IntegrationSystem(Protocol):
    """Structural protocol that describes the full expected interface.

    Use this for type annotations. You do **not** have to inherit from it;
    any object that implements the described methods satisfies the protocol.
    """

    t: float

    # ------------------------------------------------------------------ #
    # Lifecycle hooks — called by the integrator around each stage        #
    # ------------------------------------------------------------------ #

    def initialize(self, dt: float, *args: Any, **kwargs: Any) -> "IntegrationSystem":
        """Called once at the start of an integration step.

        Args:
            dt: The full step size for the upcoming step.
        """
        ...

    def initializeNewState(self, *args: Any, **kwargs: Any) -> "IntegrationSystem":
        """Return a shallow/deep copy of this state for use as a stage buffer.

        The returned object will be mutated in-place by the integrator;
        it should not share mutable tensor storage with ``self``.
        """
        ...

    def preprocess(
        self,
        initialState: "IntegrationSystem",
        dt: float,
        *args: Any,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Called before each force/RHS evaluation.

        Args:
            initialState: The state at the beginning of the full step.
            dt: The sub-step size used for this evaluation.
        """
        ...

    def postprocess(
        self,
        currentState: "IntegrationSystem",
        dt: float,
        r: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Called after each force/RHS evaluation.

        Args:
            currentState: The stage state that was evaluated.
            dt: The sub-step size used for this evaluation.
            r: Auxiliary return values from the force function, or None.
        """
        ...

    def finalize(
        self,
        initialState: "IntegrationSystem",
        dt: float,
        returnValues: List[Any],
        updateValues: List[Any],
        weights: List[float],
        *args: Any,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Called once after all stages are complete and the final state is assembled.

        Args:
            initialState: The state at the beginning of the full step.
            dt: The full step size.
            returnValues: List of auxiliary return values from each stage evaluation.
            updateValues: List of derivative objects (k-values) from each stage.
            weights: Butcher tableau weights or equivalent stage weights.
        """
        ...

    # ------------------------------------------------------------------ #
    # Legacy update interface                                             #
    # ------------------------------------------------------------------ #

    def integrate(
        self,
        update: Any,
        dt: Union[float, List[float]],
        *,
        selfScale: float = None,
        referenceState: "IntegrationSystem" = None,
        referenceWeight: float = None,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Apply a combined position + velocity + quantity update.

        Used by the Euler and Butcher-tableau paths as a combined fallback.
        If you implement the three separate methods below you do not need this.
        """
        ...

    def integratePosition(
        self,
        update: Any,
        dt: Union[float, List[float]],
        *,
        selfScale: float = None,
        semiImplicitScale: float = None,
        verletScale: float = None,
        referenceState: "IntegrationSystem" = None,
        referenceWeight: float = None,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Update position fields.

        Two special position-update modes are encoded via keyword arguments:

        * ``semiImplicitScale=s``: x += s * v_current  (velocity drift)
        * ``verletScale=s``:       x += s * update.velocity  (Verlet correction)

        At most one of these may be non-None at the same time.
        """
        ...

    def integrateVelocity(
        self,
        update: Any,
        dt: Union[float, List[float]],
        *,
        selfScale: float = None,
        referenceState: "IntegrationSystem" = None,
        referenceWeight: float = None,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Update velocity fields."""
        ...

    def integrateQuantities(
        self,
        update: Any,
        dt: Union[float, List[float]],
        *,
        selfScale: float = None,
        referenceState: "IntegrationSystem" = None,
        referenceWeight: float = None,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Update scalar quantity fields (e.g., energy, density)."""
        ...

    # ------------------------------------------------------------------ #
    # Typed update interface (preferred for new code)                     #
    # ------------------------------------------------------------------ #

    def apply_state_update(
        self,
        update: Any,
        spec: ComponentUpdateSpec,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Apply a combined position + velocity + quantity update from a typed spec."""
        ...

    def apply_position_update(
        self,
        update: Any,
        spec: PositionUpdateSpec,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Apply a position update described by a typed ``PositionUpdateSpec``."""
        ...

    def apply_velocity_update(
        self,
        update: Any,
        spec: ComponentUpdateSpec,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Apply a velocity update described by a typed ``ComponentUpdateSpec``."""
        ...

    def apply_quantity_update(
        self,
        update: Any,
        spec: ComponentUpdateSpec,
        **kwargs: Any,
    ) -> "IntegrationSystem":
        """Apply a scalar-quantity update described by a typed ``ComponentUpdateSpec``."""
        ...


class BaseIntegrationSystem:
    """Mixin that provides no-op defaults for all lifecycle hooks.

    Subclass this alongside your own state dataclass to avoid implementing
    hooks you don't need::

        @dataclass
        class MySystem(BaseIntegrationSystem):
            position: torch.Tensor
            velocity: torch.Tensor
            t: float = 0.0

            def initializeNewState(self, **kwargs):
                return MySystem(
                    position=self.position.clone(),
                    velocity=self.velocity.clone(),
                )

            def integrate(self, update, dt, **kwargs):
                ...
    """

    t: float = 0.0

    def initialize(self, dt: float, *args: Any, **kwargs: Any) -> "BaseIntegrationSystem":
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"[Initialize] Initializing at t={self.t:.4f} with dt={dt:.4f} (default hook)")
        return self

    def preprocess(
        self,
        initialState: Any,
        dt: float,
        *args: Any,
        **kwargs: Any,
    ) -> "BaseIntegrationSystem":
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"[Pre] Preprocessing at t={initialState.t:.4f} with dt={dt:.4f} (default hook)")
        return self

    def postprocess(
        self,
        currentState: Any,
        dt: float,
        r: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "BaseIntegrationSystem":
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"[Post] Postprocessing at t={currentState.t:.4f} with dt={dt:.4f} (default hook)")
        return self

    def finalize(
        self,
        initialState: Any,
        dt: float,
        returnValues: List[Any],
        updateValues: List[Any],
        weights: List[float] = (),
        *args: Any,
        **kwargs: Any,
    ) -> "BaseIntegrationSystem":
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f"[Finalize] Finalizing at t={initialState.t:.4f} with dt={dt:.4f} (default hook)")
        return self
