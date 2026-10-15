"""drivers.history._adams_layout — extracted numerical definitions with unchanged formulas."""


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names

def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),str(getattr(ref,name).dtype))
                 for name in integrated_field_names(state))
