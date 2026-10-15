"""schemes.coefficients.nystrom — extracted numerical definitions with unchanged formulas."""
from dataclasses import dataclass

from functools import lru_cache


from warpSPHIntegrators.schemes.coefficients.nystrom_data import TABLEAUS


_ORDER={'DPRKN4':4,'DPRKN5':5,'DPRKN6':6,'DPRKN6FM':6,'DPRKN8':8,'DPRKN12':12,
    'ERKN4':4,'ERKN5':5,'ERKN7':7,'Nystrom4':4,'Nystrom4VelocityIndependent':4,
    'Nystrom5VelocityIndependent':5,'FineRKN4':4,'FineRKN5':5}

_ESTIMATE={'DPRKN4':4,'DPRKN5':5,'DPRKN6':5,'DPRKN6FM':5,'DPRKN8':7,'DPRKN12':11,
    'ERKN4':4,'ERKN5':5,'ERKN7':6,'FineRKN4':4,'FineRKN5':5}

@dataclass(frozen=True)
class NystromTableau:
    a: tuple
    abar: tuple|None
    b: tuple
    bp: tuple
    btilde: tuple
    bptilde: tuple
    c: tuple
    order: int
    estimate_order: int|None
    position_only_estimate: bool

@lru_cache(maxsize=14)
def nystrom_tableau(method):
    """Immutable pinned RKN coefficients; c includes the initial zero node."""
    if method=='RKN4':
        raise ValueError('Pinned RKN4 is only order two with velocity-dependent forces; use Nystrom4 or a velocity-independent method')
    if method not in TABLEAUS:raise ValueError(f'Unknown Nyström method; choose from {tuple(_ORDER)}')
    data=TABLEAUS[method]
    return NystromTableau(tuple(tuple(row) for row in data['a']),
        tuple(tuple(row) for row in data['abar']) if data['velocity_dependent'] else None,
        *(tuple(data[key]) for key in ('b','bp','btilde','bptilde')),
        (0.,*data['c']),_ORDER[method],_ESTIMATE.get(method),data['pos_only_error'])
