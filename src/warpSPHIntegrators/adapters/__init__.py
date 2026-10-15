"""Tensor callbacks and declared RHS/state-map capabilities."""
from importlib import import_module as _import_module

_EXPORTS = {'TensorState': ('warpSPHIntegrators.adapters.tensor', 'TensorState'),
 'tensor_rhs': ('warpSPHIntegrators.adapters.tensor', 'tensor_rhs'),
 'RHS': ('warpSPHIntegrators.adapters.rhs', 'RHS'),
 'IMEXRHS': ('warpSPHIntegrators.adapters.rhs', 'IMEXRHS'),
 'SemilinearRHS': ('warpSPHIntegrators.adapters.rhs', 'SemilinearRHS'),
 'resolve': ('warpSPHIntegrators.adapters.rhs', 'resolve'),
 'ResolvedRHS': ('warpSPHIntegrators.adapters.rhs', 'ResolvedRHS'),
 'check_contracts': ('warpSPHIntegrators.adapters.rhs', 'check_contracts'),
 'add_updates': ('warpSPHIntegrators.adapters.rhs', 'add_updates'),
 'sub_updates': ('warpSPHIntegrators.adapters.rhs', 'sub_updates'),
 'CAPABILITIES': ('warpSPHIntegrators.adapters.rhs', 'CAPABILITIES'),
 'StateMap': ('warpSPHIntegrators.adapters.maps', 'StateMap'),
 'tensor_map': ('warpSPHIntegrators.adapters.maps', 'tensor_map')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
