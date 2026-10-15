"""Optional numerical inspection and reuse analysis."""
from importlib import import_module as _import_module

_EXPORTS = {'ReuseAnalysis': ('warpSPHIntegrators.analysis.reuse', 'ReuseAnalysis'),
 'step_reuse_analysis': ('warpSPHIntegrators.analysis.reuse', 'step_reuse_analysis'),
 'step_reuse_order': ('warpSPHIntegrators.analysis.reuse', 'step_reuse_order'),
 'supports_step_reuse': ('warpSPHIntegrators.analysis.reuse', 'supports_step_reuse'),
 'is_fsal': ('warpSPHIntegrators.analysis.reuse', 'is_fsal')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
