"""Coefficient construction and inspection; definitions do not select the registry."""
from importlib import import_module as _import_module

_EXPORTS = {'ButcherTableau': ('warpSPHIntegrators.schemes.coefficients.tableau', 'ButcherTableau'),
 'getButcherTableau': ('warpSPHIntegrators.schemes.coefficients.explicit', 'getButcherTableau'),
 'getDIRKTableau': ('warpSPHIntegrators.schemes.coefficients.implicit', 'getDIRKTableau'),
 'BlockTableau': ('warpSPHIntegrators.schemes.coefficients.coupled', 'BlockTableau'),
 'getBlockTableau': ('warpSPHIntegrators.schemes.coefficients.coupled', 'getBlockTableau'),
 'AdditiveTableau': ('warpSPHIntegrators.schemes.coefficients.additive', 'AdditiveTableau'),
 'getARKTableau': ('warpSPHIntegrators.schemes.coefficients.additive', 'getARKTableau'),
 'collocation_tableau': ('warpSPHIntegrators.schemes.coefficients.collocation', 'collocation_tableau'),
 'adams_weights': ('warpSPHIntegrators.schemes.coefficients.adams', 'adams_weights'),
 'stabilized_coefficients': ('warpSPHIntegrators.schemes.coefficients.stabilized',
                             'stabilized_coefficients'),
 'stabilized_interval': ('warpSPHIntegrators.schemes.coefficients.stabilized', 'stabilized_interval'),
 'NystromTableau': ('warpSPHIntegrators.schemes.coefficients.nystrom', 'NystromTableau'),
 'nystrom_tableau': ('warpSPHIntegrators.schemes.coefficients.nystrom', 'nystrom_tableau'),
 'ndf_difference_transform': ('warpSPHIntegrators.schemes.coefficients.ndf', 'ndf_difference_transform'),
 'parallel_dirk_tableau': ('warpSPHIntegrators.schemes.coefficients.parallel_dirk',
                           'parallel_dirk_tableau'),
 'RosenbrockTableau': ('warpSPHIntegrators.schemes.coefficients.rosenbrock', 'RosenbrockTableau'),
 'rosenbrock_tableau': ('warpSPHIntegrators.schemes.coefficients.rosenbrock', 'rosenbrock_tableau'),
 'extrapolation_coefficients': ('warpSPHIntegrators.schemes.coefficients.extrapolation',
                                'extrapolation_coefficients'),
 'expanded_rk_methods': ('warpSPHIntegrators.schemes.coefficients.expanded', 'expanded_rk_methods'),
 'expanded_rk_tableau': ('warpSPHIntegrators.schemes.coefficients.expanded', 'expanded_rk_tableau')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
