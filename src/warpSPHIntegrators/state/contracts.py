"""Structural stepping and accepted-control contracts shared by all backends."""
from typing import Any,Protocol,runtime_checkable

@runtime_checkable
class Stepper(Protocol):
    name: str
    identifier: Any
    order: int|None
    implicit: bool
    fsal: bool
    metadata: Any
    def __call__(self,state,dt,rhs,*args,**kwargs): ...

@runtime_checkable
class AcceptedStepControl(Protocol):
    def adapt_step(self,result,h,proposal,*,error_norm,norm_estimate): ...

@runtime_checkable
class NextStepControl(Protocol):
    def next_step_size(self,result): ...


def is_subflow(value):
    """A physical subflow exposes a call, name, order and history metadata.

    It need not support restart identifiers or error estimates used by solve.
    History-bearing methods are rejected separately by the composition builder.
    """
    return callable(value) and all(hasattr(value,key) for key in ('name','order','implicit','metadata'))
