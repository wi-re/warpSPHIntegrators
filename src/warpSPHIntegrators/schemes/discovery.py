


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
    from warpSPHIntegrators.schemes.registry import IntegrationSchemes
    from warpSPHIntegrators.schemes.enums import FAMILY_ENUMS
    from warpSPHIntegrators.adapters.rhs import CAPABILITIES
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
