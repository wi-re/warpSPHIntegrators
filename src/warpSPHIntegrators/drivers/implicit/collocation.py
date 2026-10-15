"""Numerical stepping for collocation; configuration and coefficients are supplied separately."""



import math


from warpSPHIntegrators.drivers.implicit.coupled import fullyImplicitBlock


def step(self, state, dt, rhs, *args, **kwargs):
    from warpSPHIntegrators.state.ownership import snapshot_state
    if isinstance(dt, bool) or not math.isfinite(dt) or dt == 0:
        raise ValueError('finite nonzero dt required')
    return fullyImplicitBlock(snapshot_state(state), dt, rhs, self.tableau, *args,
                              name=self.name, **kwargs)
