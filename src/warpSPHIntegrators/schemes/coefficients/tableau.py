from typing import NamedTuple, Union, Tuple
import numpy as np
class ButcherTableau(NamedTuple):
    a: np.array
    b: Union[np.array,Tuple[np.array, np.array]]
    c: np.array


def freeze_tableau(tableau):
    """Own coefficient bytes and expose read-only arrays on bound definitions."""
    def freeze(value):
        if isinstance(value,np.ndarray):
            return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
        if isinstance(value,tuple):return tuple(freeze(v) for v in value)
        return value
    return type(tableau)(*(freeze(value) for value in tableau))
