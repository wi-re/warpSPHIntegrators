"""Validated scheme selection, identifiers and family configurations."""
from importlib import import_module as _import_module

_EXPORTS = {'get_scheme': ('warpSPHIntegrators.schemes.registry', 'get_scheme'),
 'IntegrationScheme': ('warpSPHIntegrators.schemes.model', 'IntegrationScheme'),
 'IntegrationSchemeType': ('warpSPHIntegrators.schemes.enums', 'IntegrationSchemeType'),
 'ExplicitRK': ('warpSPHIntegrators.schemes.enums', 'ExplicitRK'),
 'Symplectic': ('warpSPHIntegrators.schemes.enums', 'Symplectic'),
 'RelaxedChebyshev': ('warpSPHIntegrators.schemes.enums', 'RelaxedChebyshev'),
 'MultistepExplicit': ('warpSPHIntegrators.schemes.enums', 'MultistepExplicit'),
 'DIRK': ('warpSPHIntegrators.schemes.enums', 'DIRK'),
 'CoupledRK': ('warpSPHIntegrators.schemes.enums', 'CoupledRK'),
 'MultistepImplicit': ('warpSPHIntegrators.schemes.enums', 'MultistepImplicit'),
 'IMEX': ('warpSPHIntegrators.schemes.enums', 'IMEX'),
 'LinearlyImplicit': ('warpSPHIntegrators.schemes.enums', 'LinearlyImplicit'),
 'Exponential': ('warpSPHIntegrators.schemes.enums', 'Exponential'),
 'Newmark': ('warpSPHIntegrators.schemes.enums', 'Newmark'),
 'FAMILY_ENUMS': ('warpSPHIntegrators.schemes.enums', 'FAMILY_ENUMS'),
 'SCHEME_FAMILY': ('warpSPHIntegrators.schemes.enums', 'SCHEME_FAMILY'),
 'SchemeMetadata': ('warpSPHIntegrators.schemes.metadata', 'SchemeMetadata'),
 'filter_schemes': ('warpSPHIntegrators.schemes.discovery', 'filter_schemes'),
 'IntegrationSchemes': ('warpSPHIntegrators.schemes.registry', 'IntegrationSchemes'),
 'Collocation': ('warpSPHIntegrators.schemes.families.collocation', 'Collocation'),
 'VariableAdams': ('warpSPHIntegrators.schemes.families.variable_adams', 'VariableAdams'),
 'AdaptiveAdams': ('warpSPHIntegrators.schemes.families.adaptive_adams', 'AdaptiveAdams'),
 'StabilizedRK': ('warpSPHIntegrators.schemes.families.stabilized', 'StabilizedRK'),
 'OrthogonalRK': ('warpSPHIntegrators.schemes.families.orthogonal', 'OrthogonalRK'),
 'Nystrom': ('warpSPHIntegrators.schemes.families.nystrom', 'Nystrom'),
 'ImprovedNystrom': ('warpSPHIntegrators.schemes.families.improved_nystrom', 'ImprovedNystrom'),
 'GeneralizedAlpha': ('warpSPHIntegrators.schemes.families.generalized_alpha', 'GeneralizedAlpha'),
 'ParallelDIRK': ('warpSPHIntegrators.schemes.families.parallel_dirk', 'ParallelDIRK'),
 'Rosenbrock': ('warpSPHIntegrators.schemes.families.linearized', 'Rosenbrock'),
 'ExponentialRK': ('warpSPHIntegrators.schemes.families.exponential_rk', 'ExponentialRK'),
 'NDF': ('warpSPHIntegrators.schemes.families.ndf', 'NDF'),
 'AdaptiveNDF': ('warpSPHIntegrators.schemes.families.adaptive_ndf', 'AdaptiveNDF'),
 'Extrapolation': ('warpSPHIntegrators.schemes.families.extrapolation', 'Extrapolation'),
 'ExpandedRK': ('warpSPHIntegrators.schemes.families.expanded_rk', 'ExpandedRK'),
 'FunctionMap': ('warpSPHIntegrators.schemes.families.function_map', 'FunctionMap'),
 'NystromWeights': ('warpSPHIntegrators.schemes.families.nystrom', 'NystromWeights'),
 'NDFQuadrature': ('warpSPHIntegrators.schemes.families.ndf', 'NDFQuadrature'),
 'ImprovedNystromWeights': ('warpSPHIntegrators.schemes.families.improved_nystrom',
                            'ImprovedNystromWeights'),
 'GeneralizedAlphaMemory': ('warpSPHIntegrators.schemes.families.generalized_alpha',
                            'GeneralizedAlphaMemory'),
 'RosenbrockQuadrature': ('warpSPHIntegrators.schemes.families.linearized', 'RosenbrockQuadrature'),
 'ExponentialQuadrature': ('warpSPHIntegrators.schemes.families.exponential_rk',
                           'ExponentialQuadrature'),
 'ExtrapolationQuadrature': ('warpSPHIntegrators.schemes.families.extrapolation',
                             'ExtrapolationQuadrature'),
 'MapTransition': ('warpSPHIntegrators.schemes.families.function_map', 'MapTransition')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
