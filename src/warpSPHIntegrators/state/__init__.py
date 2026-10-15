"""Full-state contracts, field ownership and typed component updates."""
from importlib import import_module as _import_module

_EXPORTS = {'BaseState': ('warpSPHIntegrators.state.fields', 'BaseState'),
 'BlockState': ('warpSPHIntegrators.state.fields', 'BlockState'),
 'IntegrationSystem': ('warpSPHIntegrators.state.protocol', 'IntegrationSystem'),
 'BaseIntegrationSystem': ('warpSPHIntegrators.state.protocol', 'BaseIntegrationSystem'),
 'integrated': ('warpSPHIntegrators.state.fields', 'integrated'),
 'constant': ('warpSPHIntegrators.state.fields', 'constant'),
 'copied': ('warpSPHIntegrators.state.fields', 'copied'),
 'ephemeral': ('warpSPHIntegrators.state.fields', 'ephemeral'),
 'custom': ('warpSPHIntegrators.state.fields', 'custom'),
 'tagged': ('warpSPHIntegrators.state.fields', 'tagged'),
 'reference_state': ('warpSPHIntegrators.state.fields', 'reference_state'),
 'find_tagged_field': ('warpSPHIntegrators.state.fields', 'find_tagged_field'),
 'get_tagged_attr': ('warpSPHIntegrators.state.fields', 'get_tagged_attr'),
 'set_tagged_attr': ('warpSPHIntegrators.state.fields', 'set_tagged_attr'),
 'get_reference_state': ('warpSPHIntegrators.state.fields', 'get_reference_state'),
 'clone_value': ('warpSPHIntegrators.state.fields', 'clone_value'),
 'empty_value': ('warpSPHIntegrators.state.fields', 'empty_value'),
 'move_value': ('warpSPHIntegrators.state.fields', 'move_value'),
 'register_clone_handler': ('warpSPHIntegrators.state.fields', 'register_clone_handler'),
 'CloneHandler': ('warpSPHIntegrators.state.fields', 'CloneHandler'),
 'state_difference': ('warpSPHIntegrators.state.fields', 'state_difference'),
 'state_norm': ('warpSPHIntegrators.state.fields', 'state_norm'),
 'integrated_field_names': ('warpSPHIntegrators.state.fields', 'integrated_field_names'),
 'flatten_integrated': ('warpSPHIntegrators.state.fields', 'flatten_integrated'),
 'unflatten_integrated': ('warpSPHIntegrators.state.fields', 'unflatten_integrated'),
 'replace_integrated_fields': ('warpSPHIntegrators.state.fields', 'replace_integrated_fields'),
 'StateBlend': ('warpSPHIntegrators.state.specs', 'StateBlend'),
 'ComponentUpdateSpec': ('warpSPHIntegrators.state.specs', 'ComponentUpdateSpec'),
 'PositionUpdateSpec': ('warpSPHIntegrators.state.specs', 'PositionUpdateSpec'),
 'StageResult': ('warpSPHIntegrators.state.specs', 'StageResult'),
 'StepEvaluation': ('warpSPHIntegrators.state.specs', 'StepEvaluation'),
 'IntegrationResult': ('warpSPHIntegrators.state.specs', 'IntegrationResult'),
 'blend_state': ('warpSPHIntegrators.state.specs', 'blend_state'),
 'explicit_step': ('warpSPHIntegrators.state.specs', 'explicit_step'),
 'semi_implicit_position_step': ('warpSPHIntegrators.state.specs', 'semi_implicit_position_step'),
 'verlet_position_step': ('warpSPHIntegrators.state.specs', 'verlet_position_step'),
 'HistoryEntry': ('warpSPHIntegrators.state.history', 'HistoryEntry'),
 'StepHistory': ('warpSPHIntegrators.state.history', 'StepHistory'),
 'updateStateEuler': ('warpSPHIntegrators.state.lifecycle', 'updateStateEuler'),
 'updateStateSemiImplicitEuler': ('warpSPHIntegrators.state.lifecycle', 'updateStateSemiImplicitEuler'),
 'applyStateUpdate': ('warpSPHIntegrators.state.lifecycle', 'applyStateUpdate'),
 'applyPositionUpdate': ('warpSPHIntegrators.state.lifecycle', 'applyPositionUpdate'),
 'applyVelocityUpdate': ('warpSPHIntegrators.state.lifecycle', 'applyVelocityUpdate'),
 'applyQuantityUpdate': ('warpSPHIntegrators.state.lifecycle', 'applyQuantityUpdate'),
 'deferHostTime': ('warpSPHIntegrators.state.lifecycle', 'deferHostTime'),
 'hostTime': ('warpSPHIntegrators.state.lifecycle', 'hostTime'),
 'update_component': ('warpSPHIntegrators.state.fields', 'update_component'),
 'update_position': ('warpSPHIntegrators.state.fields', 'update_position'),
 'Stepper': ('warpSPHIntegrators.state.contracts', 'Stepper'),
 'AcceptedStepControl': ('warpSPHIntegrators.state.contracts', 'AcceptedStepControl'),
 'NextStepControl': ('warpSPHIntegrators.state.contracts', 'NextStepControl')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attribute=_EXPORTS[name]
    value=getattr(_import_module(module),attribute)
    globals()[name]=value
    return value

def __dir__():return sorted(set(globals()) | set(__all__))
