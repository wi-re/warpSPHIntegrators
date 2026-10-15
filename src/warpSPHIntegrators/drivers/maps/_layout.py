"""drivers.maps._layout — extracted numerical definitions with unchanged formulas."""


import torch


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names

def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),getattr(ref,name).dtype,getattr(ref,name).device)
        for name in integrated_field_names(state))

def _uid_equal(a,b):
    return (a is None and b is None or isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
        and a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b))
