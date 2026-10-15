"""Run control, events and portable restart."""
from importlib import import_module as _import_module

_EXPORTS = {'solve': ('warpSPHIntegrators.runtime.run', 'solve'),
 'Event': ('warpSPHIntegrators.runtime.run', 'Event'),
 'EventHit': ('warpSPHIntegrators.runtime.run', 'EventHit'),
 'save_checkpoint': ('warpSPHIntegrators.runtime.checkpoint', 'save_checkpoint'),
 'load_checkpoint': ('warpSPHIntegrators.runtime.checkpoint', 'load_checkpoint'),
 'StepRecord': ('warpSPHIntegrators.runtime.run', 'StepRecord'),
 'RunSample': ('warpSPHIntegrators.runtime.run', 'RunSample'),
 'RunResult': ('warpSPHIntegrators.runtime.run', 'RunResult'),
 'RunError': ('warpSPHIntegrators.runtime.run', 'RunError'),
 'RunCheckpoint': ('warpSPHIntegrators.runtime.run', 'RunCheckpoint'),
 'estimate_error_norm': ('warpSPHIntegrators.runtime.adaptive', 'estimate_error_norm'),
 'propose_dt': ('warpSPHIntegrators.runtime.adaptive', 'propose_dt'),
 'dormand_prince_dense_output': ('warpSPHIntegrators.runtime.adaptive', 'dormand_prince_dense_output'),
 'assess_step': ('warpSPHIntegrators.state.ownership', 'assess_step'),
 'snapshot_state': ('warpSPHIntegrators.state.ownership', 'snapshot_state'),
 'StepStatus': ('warpSPHIntegrators.state.ownership', 'StepStatus')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
