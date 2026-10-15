from dataclasses import dataclass


import torch


@dataclass(frozen=True)
class StateMap:
    """Declare function(owned_state_at_t_plus_h, h, ...)->mapped_state[, aux].

    Integrated inputs are the initialized old values. The returned state must
    preserve integrated layout/dtype/device, endpoint time and particle IDs.
    Numeric transitions belong here; lifecycle hooks preserve mapped values.
    """
    function: object
    def __post_init__(self):
        if not callable(self.function):raise TypeError('StateMap requires a callable')
    def __call__(self,state,dt,*args,**kwargs):return self.function(state,dt,*args,**kwargs)

def tensor_map(function,/,**parameters):
    """Adapt map(t_next,y_old,h,*args,**parameters)->y_next, not an ODE rate.

    Step options are not forwarded. Shape/dtype/device remain unchanged.
    The mapped tensor remains differentiable; lifecycle buffers are owned.
    """
    if not callable(function):raise TypeError('tensor_map requires a callable')
    def evaluate(system,dt,*args,**kwargs):
        if not getattr(system,'_tensor_adapter',False):raise TypeError('tensor_map requires TensorState')
        value=function(system.t,system.y,dt,*args,**parameters)
        if not isinstance(value,torch.Tensor):raise TypeError('Tensor map must return a torch.Tensor')
        if value.shape!=system.y.shape or value.dtype!=system.y.dtype or value.device!=system.y.device:
            raise ValueError('Tensor map must preserve shape, dtype and device')
        output=system.initializeNewState();output.state.y=value;output.t=system.t
        return output
    return StateMap(evaluate)
