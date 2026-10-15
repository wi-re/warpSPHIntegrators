"""Time integration for tensor ODEs and complete simulation states.

Use focused schemes/state/solvers/runtime/adapters/wrappers packages for advanced
capabilities. Raw method functions are intentionally absent from the root API.
"""
from .schemes import get_scheme
from .runtime import solve
from .adapters import TensorState, tensor_rhs
__version__ = '0.7.0'
__all__ = ['get_scheme', 'solve', 'TensorState', 'tensor_rhs']
