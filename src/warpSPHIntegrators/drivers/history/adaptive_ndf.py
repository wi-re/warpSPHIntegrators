"""Numerical stepping for adaptive_ndf; configuration and coefficients are supplied separately."""



from warpSPHIntegrators.state.history import StepHistory

from warpSPHIntegrators.schemes.families.ndf import NDF


def step(self,state,dt,f,*args,history=None,restart_history=False,**kwargs):
    if isinstance(history,StepHistory) and history.maxlen<self.max_order:
        raise ValueError('StepHistory capacity must cover max_order')
    context=history.context if isinstance(history,StepHistory) else None
    compatible=(isinstance(context,dict) and context.get('adaptive_ndf')==self.identifier.name
        and type(context.get('order')) is int and 1<=context['order']<=self.max_order)
    order=context['order'] if compatible and not restart_history else 1
    if history is not None and not compatible:
        if not isinstance(history,StepHistory):raise TypeError('StepHistory required')
        history=StepHistory(history.maxlen)
    result=NDF(order,self.max_step_ratio)(state,dt,f,*args,history=history,restart_history=restart_history,**kwargs)
    if result.step_status.successful:
        result.history.context.update(adaptive_ndf=self.identifier.name,last_order=order)
    return result
