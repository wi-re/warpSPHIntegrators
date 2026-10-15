"""Physical flow composition and whole-algorithm selection."""
from importlib import import_module as _import_module

_EXPORTS = {'AlgorithmSwitch': ('warpSPHIntegrators.wrappers.switching', 'AlgorithmSwitch'),
 'ConservativeDissipativeRHS': ('warpSPHIntegrators.adapters.rhs', 'ConservativeDissipativeRHS'),
 'compose': ('warpSPHIntegrators.wrappers.splitting', 'compose'),
 'LieTrotter': ('warpSPHIntegrators.wrappers.splitting', 'LieTrotter'),
 'Strang': ('warpSPHIntegrators.wrappers.splitting', 'Strang'),
 'Yoshida4': ('warpSPHIntegrators.wrappers.splitting', 'Yoshida4'),
 'SplitScheme': ('warpSPHIntegrators.wrappers.splitting', 'SplitScheme'),
 'LinearExponentialFlow': ('warpSPHIntegrators.wrappers.splitting', 'LinearExponentialFlow'),
 'classify_separable': ('warpSPHIntegrators.wrappers.splitting', 'classify_separable'),
 'SeparabilityResult': ('warpSPHIntegrators.wrappers.splitting', 'SeparabilityResult')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
