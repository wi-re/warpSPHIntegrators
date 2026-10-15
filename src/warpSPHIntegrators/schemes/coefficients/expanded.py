"""schemes.coefficients.expanded — extracted numerical definitions with unchanged formulas."""


from functools import lru_cache

import difflib


import numpy as np

from warpSPHIntegrators.schemes.coefficients.expanded_data import _METHODS

from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau


def expanded_rk_methods():
    """Qualified source method names, sorted; unsupported source pins are omitted."""
    return tuple(sorted(_METHODS))

def _array(values):
    array=np.asarray(values,dtype=np.float64)
    return np.frombuffer(array.tobytes(),dtype=np.float64).reshape(array.shape)

@lru_cache(maxsize=64)
def expanded_rk_tableau(method):
    """Immutable full-stage coefficients, including physical endpoint refresh.

    Nineteen source companions pass the independently checked order gates. The
    pinned CKLLSRK85_4FM_4R main map passes, but its advertised companion does not;
    that configuration is deliberately fixed-step only in this backend.
    """
    if not isinstance(method,str) or method not in _METHODS:
        hint=difflib.get_close_matches(method,expanded_rk_methods(),n=3) if isinstance(method,str) else ()
        raise ValueError(f'Unknown qualified expanded RK method {method!r}; closest: {tuple(hint)}')
    data=_METHODS[method];main=_array(data['b']);error=data['error_weights']
    weights=main if error is None else (main,_array(np.asarray(data['b'])-np.asarray(error)))
    return butcherTableau(_array(data['a']),weights,_array(data['c']))
