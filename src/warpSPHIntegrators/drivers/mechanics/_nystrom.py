"""drivers.mechanics._nystrom — extracted numerical definitions with unchanged formulas."""


def _sum(value,updates,weights,key,scale):
    for update,weight in zip(updates,weights):
        if weight:value=value+(scale*weight)*getattr(update,key)
    return value
