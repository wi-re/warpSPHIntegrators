


from typing import NamedTuple, Optional


from warpSPHIntegrators.schemes.enums import IntegrationSchemeType, SCHEME_FAMILY

from typing import Callable


class IntegrationScheme(NamedTuple):
    function: Callable
    name: str
    identifier: IntegrationSchemeType
    order: int

    dissipation: bool = False

    #: Convergence order retained when `priorStep` reuse is used, or None when the
    #: scheme does not implement reuse at all. Filled in at registration time from
    #: `integrators.reuse.step_reuse_analysis`; see that module for the derivation.
    reuse_order: Optional[int] = None
    #: True when the tableau is FSAL, so reuse is exact rather than merely lossless.
    fsal: bool = False

    # ------------------------------------------------------------------ #
    # docs/research/integrators/notes-index.md S3: metadata a driver needs to tell schemes apart without    #
    # special-casing them by name. All default to the "plain explicit RK"  #
    # answer, so every scheme registered before this landed is unaffected. #
    # ------------------------------------------------------------------ #

    #: True for a scheme whose stage equations require solving for an unknown that
    #: appears on both sides (DIRK, fully implicit RK, BDF/Adams-Moulton). False for
    #: every explicit scheme and for linear multistep predictors (Adams-Bashforth).
    implicit: bool = False
    #: How many past-step states/derivatives a step needs beyond the current one: 1
    #: for every one-step method (RK, DIRK), k for a k-step linear multistep method.
    steps: int = 1
    #: True when the last stage IS the step (`a[-1] == b`), the implicit analogue of
    #: FSAL (`reuse.tableau_reuse_analysis` already detects this for explicit
    #: tableaus; this field is what lets an implicit driver ask the same question
    #: without re-deriving it from `a`/`b` at call time).
    stiffly_accurate: bool = False
    #: Linear stability region, or None for a scheme with no useful one to name
    #: (mixed-order embedded pairs, symplectic-but-not-A/L-stable methods). 'A':
    #: stable for all Re(lambda*dt) <= 0. 'L': A-stable and additionally damps
    #: infinitely stiff modes to zero in one step. 'A(alpha)': A-stable only within a
    #: cone of half-angle alpha from the negative real axis (BDF3+).
    stability: Optional[str] = None
    #: Convergence order actually reached from a cold start (history length < steps),
    #: or None for a one-step method, where the question does not apply. A k-step
    #: multistep method needs k-1 prior derivatives it does not have yet at t=0;
    #: Repeated cold starts follow this method's global order. A finite startup
    #: contributes its local defect: e.g. one Euler step is O(h^2), sufficient
    #: to start a globally second-order formula (but not order three or higher).
    startup_order: Optional[int] = None

    @property
    def metadata(self):
        """Stage, companion, RHS and per-step requirements as SchemeMetadata.

        Derived properties preserve the existing NamedTuple fields, positional
        construction and unpacking contract.
        """
        from warpSPHIntegrators.schemes.metadata import scheme_metadata
        return scheme_metadata(self)

    @property
    def has_error_estimate(self) -> bool:
        return self.metadata.has_error_estimate

    @property
    def estimate_order(self) -> Optional[int]:
        """Power q in the step's O(h^q) estimate; pass this to propose_dt."""
        return self.metadata.estimate_order

    @property
    def embedded_order(self) -> Optional[int]:
        return self.metadata.embedded_order

    @property
    def stage_count(self) -> Optional[int]:
        """Stored tableau stages, or None for parameterized/non-tableau drivers."""
        return self.metadata.stage_count

    @property
    def required_capabilities(self):
        """Mandatory structured RHS accessors; empty means a plain callable works."""
        return self.metadata.required_capabilities

    @property
    def supports_reuse(self) -> bool:
        """True iff `priorStep` reuse costs this scheme no convergence order."""
        return self.reuse_order is not None and self.reuse_order >= self.order

    @property
    def family(self):
        """The family enum class this scheme belongs to (e.g. `ExplicitRK`, `DIRK`).

        One of the `enums.FAMILY_ENUMS` views of `IntegrationSchemeType` — the
        same fact as `SCHEME_FAMILY[self.identifier]`, on the registered
        scheme object.
        """
        return SCHEME_FAMILY[self.identifier]

    @property
    def definition(self):
        """The immutable tableau definition or explicit hand-written driver."""
        return self.function

    def __call__(self, state, dt, f, *args, **kwargs):
        if kwargs.get('priorStep') is not None and self.name not in _warned_reuse:
            from warpSPHIntegrators.analysis.reuse import step_reuse_analysis
            import warnings
            analysis=step_reuse_analysis(self)
            if analysis.order is None or analysis.order<self.order:
                _warned_reuse.add(self.name)
                detail=analysis.reason if analysis.order is None else f'first-stage reuse (priorStep) drops the convergence order from {self.order} to {analysis.order}, because {analysis.reason}'
                warnings.warn(f'{self.name}: {detail}. Use an FSAL scheme such as Dormand-Prince 5(4) for lossless reuse.',RuntimeWarning,stacklevel=2)
        if getattr(state, '_tensor_adapter', False) and self.metadata.state_kind != 'tensor':
            raise ValueError(f'{self.name} requires a position/velocity system, not TensorState')
        return self.function(state, dt, f, *args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


_warned_reuse = set()
