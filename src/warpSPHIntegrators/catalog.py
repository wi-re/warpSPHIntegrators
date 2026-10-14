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
            embedded = 2
            extra = 1  # the null stage f(t_n, y_n), outside the coupled solve
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
    return SchemeMetadata(
        stage_count=stages, embedded_order=embedded, estimate_order=q,
        required_capabilities=required, optional_capabilities=optional,
        required_options=options, dense_output=key == 'dormandPrince',
        estimate_extra_stages=extra,
    )
