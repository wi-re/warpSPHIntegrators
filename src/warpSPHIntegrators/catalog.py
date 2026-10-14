"""Catalog metadata derived from the registered driver's actual tableau.

The two built-in estimators and the two exceptional companion orders are explicit
driver facts. A plain callable supplies the combined RHS to every method except
ETD2RK. IMEX splits are optional capabilities, not mandatory ones.
"""

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class SchemeMetadata:
    stage_count: Optional[int] = None
    embedded_order: Optional[int] = None
    estimate_order: Optional[int] = None
    required_capabilities: FrozenSet[str] = frozenset()
    optional_capabilities: FrozenSet[str] = frozenset()
    required_options: Tuple[str, ...] = ()
    dense_output: bool = False
    estimate_extra_stages: int = 0
    state_kind: str = 'tensor'
    requires_separable: bool = False
    supports_drift_fields: bool = False
    history_length: int = 0

    @property
    def has_error_estimate(self) -> bool:
        return self.estimate_order is not None


def scheme_metadata(scheme) -> SchemeMetadata:
    """Inspect tableau companions; do not change the scheme's tuple layout."""
    fn = scheme.function
    key = scheme.identifier.name
    stages = None
    embedded = None
    extra = 0
    for attribute in ('butcherTableau', 'dirkTableau', 'arkTableau', 'blockTableau'):
        tab = getattr(fn, attribute, None)
        if tab is None:
            continue
        stages = len(tab.c)
        if attribute in ('butcherTableau', 'dirkTableau'):
            if isinstance(tab.b, tuple):
                embedded = 3 if key == 'trbdf2' else scheme.order - 1
        elif attribute == 'arkTableau':
            if tab.d_explicit is not None and tab.d_implicit is not None:
                embedded = scheme.order - 1
        elif tab.companion_b is not None:
            embedded = 3 if key == 'radauIia3' else 2
            extra = int(tab.companion_b[0] != 0)  # optional null RHS evaluation
        break
    if key in {'ros3p', 'exprb32'}:
        stages, embedded = (3 if key == 'ros3p' else 2), 2
    if key == 'etd2rk':
        stages = 2
    if key in {'tvdRK2', 'tvdRK3', 'semiImplicitEuler', 'explicitEuler'}:
        stages = {'tvdRK2': 2, 'tvdRK3': 3,
                  'semiImplicitEuler': 1, 'explicitEuler': 1}[key]
    # TR-BDF2 propagates the lower-order branch: its difference is O(h^3).
    q = min(scheme.order, embedded) + 1 if embedded is not None else None
    required = frozenset({'linear'}) if key == 'etd2rk' else frozenset()
    optional = frozenset()
    if scheme.family.__name__ == 'IMEX':
        optional = frozenset({'explicit', 'implicit'})
    elif key in {'ros3p', 'exprb32'}:
        optional = frozenset({'linear', 'nonlinear'})
    elif key == 'etd2rk':
        optional = frozenset({'nonlinear'})
    options = ('s', 'lambda_max') if key in {'rkc1', 'rkc2', 'rkl2'} else ()
    partitioned = scheme.family.__name__ == 'Symplectic'
    return SchemeMetadata(
        stage_count=stages, embedded_order=embedded, estimate_order=q,
        required_capabilities=required, optional_capabilities=optional,
        required_options=options, dense_output=key == 'dormandPrince',
        estimate_extra_stages=extra,
        state_kind='second_order' if partitioned or key == 'newmark' else 'tensor',
        requires_separable=partitioned and scheme.order > 1,
        supports_drift_fields=partitioned,
        history_length=scheme.steps if scheme.startup_order is not None else 0,
    )


def filter_schemes(*, family=None, order=None, min_order=None, max_order=None,
                   implicit=None, has_error_estimate=None, dense_output=None,
                   supports_reuse=None, state_kind=None, requires_history=None,
                   requires_separable=None, supports_drift_fields=None,
                   available_capabilities=None):
    """Return registered records matching all supplied criteria, in catalog order.

    ``family`` is a family enum class or its name. ``available_capabilities``
    admits schemes whose mandatory RHS accessors are a subset of that set.
    An empty tuple means no matches; invalid criteria raise ValueError.
    """
    from .integration import IntegrationSchemes
    from .enums import FAMILY_ENUMS
    from .rhs import CAPABILITIES
    if isinstance(family, str):
        families = {f.__name__: f for f in FAMILY_ENUMS}
        if family not in families:
            raise ValueError(f'Unknown family {family!r}; choose from {tuple(families)}')
        family = families[family]
    if family is not None and family not in FAMILY_ENUMS:
        raise ValueError('family must be a family enum class or its name')
    for name, value in (('order', order), ('min_order', min_order), ('max_order', max_order)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if min_order is not None and max_order is not None and min_order > max_order:
        raise ValueError('min_order must not exceed max_order')
    if state_kind not in (None, 'tensor', 'second_order'):
        raise ValueError("state_kind must be 'tensor' or 'second_order'")
    for name, value in (('implicit', implicit), ('has_error_estimate', has_error_estimate),
                        ('dense_output', dense_output), ('supports_reuse', supports_reuse),
                        ('requires_separable', requires_separable), ('supports_drift_fields', supports_drift_fields),
                        ('requires_history', requires_history)):
        if value is not None and type(value) is not bool:
            raise ValueError(f'{name} must be a bool')
    capabilities = None
    if available_capabilities is not None:
        if isinstance(available_capabilities, str):
            raise ValueError('available_capabilities must be a collection of capability names')
        capabilities = frozenset(available_capabilities)
        if not capabilities <= CAPABILITIES:
            raise ValueError(f'Unknown RHS capabilities: {capabilities - CAPABILITIES}')
    result = []
    for s in IntegrationSchemes:
        m = s.metadata
        if family is not None and s.family is not family:
            continue
        if order is not None and s.order != order:
            continue
        if min_order is not None and s.order < min_order:
            continue
        if max_order is not None and s.order > max_order:
            continue
        if any(value is not None and actual != value for actual, value in (
            (s.implicit, implicit), (s.has_error_estimate, has_error_estimate),
            (m.dense_output, dense_output), (s.supports_reuse, supports_reuse),
            (m.requires_separable, requires_separable), (m.supports_drift_fields, supports_drift_fields),
            (m.state_kind, state_kind), (m.history_length > 0, requires_history))):
            continue
        if capabilities is not None and not m.required_capabilities <= capabilities:
            continue
        result.append(s)
    return tuple(result)
