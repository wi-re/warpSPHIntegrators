"""drivers.mechanics._generalized — extracted numerical definitions with unchanged formulas."""


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names

def _layout(state):
    ref = get_reference_state(state)
    return tuple((name, tuple(getattr(ref, name).shape), getattr(ref, name).dtype,
                  getattr(ref, name).device) for name in integrated_field_names(state))
