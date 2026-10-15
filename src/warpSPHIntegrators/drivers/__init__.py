"""Advanced numerical engines consuming explicit coefficient definitions."""
from importlib import import_module as _import_module

_EXPORTS = {'explicit_rk_step': ('warpSPHIntegrators.drivers.runge_kutta', 'RungeKuttaB'),
 'dirk_step': ('warpSPHIntegrators.drivers.implicit.dirk', 'DIRK'),
 'coupled_rk_step': ('warpSPHIntegrators.drivers.implicit.coupled', 'fullyImplicitBlock'),
 'additive_rk_step': ('warpSPHIntegrators.drivers.implicit.ark', 'ARK')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
