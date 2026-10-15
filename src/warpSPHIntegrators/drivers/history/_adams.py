"""drivers.history._adams — extracted numerical definitions with unchanged formulas."""


from dataclasses import fields


import torch


from warpSPHIntegrators.state.fields import get_reference_state, integrated_field_names, flatten_integrated


def _defect(template, updates, weights, dt):
    # Work on declared derivative fields directly: no RHS/lifecycle calls and no
    # subtraction of large state offsets in these neighboring-order defects.
    ref=get_reference_state(template);names=integrated_field_names(template);parts=[]
    for field in fields(ref):
        if field.name not in names:continue
        total=torch.zeros_like(getattr(ref,field.name))
        for update,weight in zip(updates,weights):
            if weight:total=total+float(weight)*getattr(update,field.metadata['update_key'])
        parts.append((dt*total).reshape(-1))
    return torch.cat(parts) if parts else flatten_integrated(template)*0
