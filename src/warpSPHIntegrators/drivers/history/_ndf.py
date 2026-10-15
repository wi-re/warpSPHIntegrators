"""drivers.history._ndf — extracted numerical definitions with unchanged formulas."""


from dataclasses import replace


import torch


from warpSPHIntegrators.schemes.definitions.dirk import TRBDF2, ESDIRK324L2SA, KenCarp4, KenCarp5

from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, flatten_integrated

_STARTER={1:TRBDF2,2:TRBDF2,3:ESDIRK324L2SA,4:KenCarp4,5:KenCarp5}

_STARTER_EXPONENT={1:2,2:3,3:3,4:4,5:5}

def _record_rhs(diagnostics,calls):
    items=list(diagnostics) if isinstance(diagnostics,(tuple,list)) else [diagnostics]
    known=sum(d.rhs_evaluations+d.gmres_iterations for d in items if d is not None)
    for i,d in enumerate(items):
        if d is not None:
            items[i]=replace(d,rhs_evaluations=d.rhs_evaluations+max(0,calls-known));break
    return items if isinstance(diagnostics,(tuple,list)) else items[0]

def _layout(state):
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),str(getattr(ref,name).dtype))
        for name in integrated_field_names(state))

def _initial_differences(current,entries,h,order):
    """Interpolate accepted startup values onto the current uniform virtual grid."""
    values=[flatten_integrated(current)]+[flatten_integrated(entry.state) for entry in reversed(entries)]
    nodes=[0.]+[(entry.t-float(current.t))/h for entry in reversed(entries)]
    virtual=[values[0]]
    for j in range(1,order+1):
        value=values[0]
        for i,node in enumerate(nodes):
            weight=1.
            for k,other in enumerate(nodes):
                if k!=i:weight*=(-j-other)/(node-other)
            value=value+weight*(values[i]-values[0])
        virtual.append(value)
    differences=[]
    for _ in range(order):
        virtual=[a-b for a,b in zip(virtual,virtual[1:])];differences.append(virtual[0])
    return torch.stack([*differences,torch.zeros_like(values[0]),torch.zeros_like(values[0])])
