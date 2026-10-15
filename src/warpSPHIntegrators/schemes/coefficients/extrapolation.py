"""schemes.coefficients.extrapolation — extracted numerical definitions with unchanged formulas."""


from fractions import Fraction

from functools import lru_cache


@lru_cache(maxsize=32)
def extrapolation_coefficients(base,order):
    """Immutable sequence/main/comparison weights, derived in exact arithmetic.

    Euler uses Romberg subdivisions 1,2,4,...; midpoint uses even harmonic
    subdivisions 2,4,6,... and cancels even inverse-step powers. The comparison
    drops the coarsest column. At midpoint order two it is Euler instead.
    """
    if base not in ('euler','midpoint'):raise ValueError('base must be euler or midpoint')
    if type(order) is not int or (base=='euler' and not 2<=order<=8) or (base=='midpoint' and (not 2<=order<=10 or order%2)):
        raise ValueError('Euler orders 2–8 or even midpoint orders 2–10 required')
    sequence=tuple(2**i for i in range(order)) if base=='euler' else tuple(2*(i+1) for i in range(order//2))
    power=1 if base=='euler' else 2
    def weights(ns):
        nodes=[Fraction(1,n**power) for n in ns];out=[]
        for i,x in enumerate(nodes):
            value=Fraction(1)
            for j,y in enumerate(nodes):
                if i!=j:value*=-y/(x-y)
            out.append(float(value))
        return tuple(out)
    main=weights(sequence)
    comparison=(0.,)+weights(sequence[1:]) if len(sequence)>1 else (0.,)
    return sequence,main,comparison
