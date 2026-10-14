"""Transactional user-selected switching with owned per-method history."""
from dataclasses import dataclass,replace
import hashlib
import math
from types import SimpleNamespace
import torch
from .catalog import SchemeMetadata
from .history import StepHistory
from .integration import getIntegrator
from .util import IntegrationScheme,reject_prior_step


def _layout(state):
    from .fields import get_reference_state,integrated_field_names
    ref=get_reference_state(state)
    return tuple((name,tuple(getattr(ref,name).shape),str(getattr(ref,name).dtype))
        for name in integrated_field_names(state))


@dataclass(frozen=True)
class AlgorithmSwitch:
    """Select one complete method per trial with an owned state probe.

    selector(state, previous_index) returns a zero-based int. It must be pure;
    previous_index is the last accepted selection, or None after reset. policy_id
    names its semantics for portable restart and must change when they change.
    Same-method trials retain owned history; switching restarts the new method.
    Cross-method history transfer, FSAL and dense interpolation are excluded.
    All candidates must be estimate emitters for adaptive runs. No automatic
    stiffness estimator, upstream cache-transfer or solver-controller claim.
    """
    methods: tuple
    selector: object
    policy_id: str

    def __post_init__(self):
        from .collocation import Collocation
        from .extrapolation import Extrapolation
        from .nystrom import Nystrom
        from .stabilized import StabilizedRK
        from .ndf import NDF
        from .adaptive_adams import AdaptiveAdams
        from .adaptive_ndf import AdaptiveNDF
        from .variable_adams import VariableAdams
        from .generalized_alpha import GeneralizedAlpha
        from .splitting import SplitScheme
        from .expanded_rk import ExpandedRK
        if not callable(self.selector):raise TypeError('selector must be callable')
        if not isinstance(self.policy_id,str) or not self.policy_id.strip():raise ValueError('nonempty policy_id required')
        if not isinstance(self.methods,(tuple,list)) or len(self.methods)<2:raise ValueError('at least two methods required')
        selected=[];bound=(IntegrationScheme,Collocation,Extrapolation,Nystrom,StabilizedRK,
            NDF,AdaptiveNDF,AdaptiveAdams,VariableAdams,GeneralizedAlpha,SplitScheme,ExpandedRK)
        for method in self.methods:
            method=method if isinstance(method,bound) else getIntegrator(method)
            selected.append(method)
        object.__setattr__(self,'methods',tuple(selected))
    @property
    def name(self):return f'Algorithm switch ({self.policy_id})'
    @property
    def identifier(self):
        description=repr((tuple(m.identifier.name for m in self.methods),self.policy_id))
        return SimpleNamespace(name='algorithm_switch_'+hashlib.sha256(description.encode()).hexdigest()[:24])
    @property
    def order(self):return None  # selection/startup frequency determines trajectory order
    @property
    def implicit(self):return any(method.implicit for method in self.methods)
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return all(method.has_error_estimate for method in self.methods)
    @property
    def estimate_order(self):
        return min(m.metadata.estimate_order for m in self.methods) if self.has_error_estimate else None
    @property
    def metadata(self):
        partitioned=any(m.metadata.state_kind=='second_order' for m in self.methods)
        return SchemeMetadata(estimate_order=self.estimate_order,
            history_length=max(1,*(m.metadata.history_length for m in self.methods)),history_requires_constant_dt=False,
            state_kind='second_order' if partitioned else 'tensor',
            required_capabilities=frozenset().union(*(m.metadata.required_capabilities for m in self.methods)),
            optional_capabilities=frozenset().union(*(m.metadata.optional_capabilities for m in self.methods)))

    def __call__(self,state,dt,f,*args,history=None,restart_history=False,**kwargs):
        from .run import snapshot_state
        from .fields import flatten_integrated,get_reference_state
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if type(restart_history) is not bool:raise ValueError('restart_history must be bool')
        if history is not None and not isinstance(history,StepHistory):raise TypeError('StepHistory required')
        reject_prior_step(self.name,kwargs.pop('priorStep',None))
        if getattr(state,'_tensor_adapter',False) and self.metadata.state_kind!='tensor':
            raise ValueError('Every candidate must support TensorState')
        if history is not None and history.maxlen<self.metadata.history_length:
            raise ValueError('StepHistory capacity must cover every candidate')
        accepted=snapshot_state(history) if history is not None else StepHistory(self.metadata.history_length)
        context=accepted.context;previous=None
        if (not restart_history and isinstance(context,dict) and context.get('method')==self.identifier.name
                and context.get('t')==float(state.t) and type(context.get('index')) is int
                and 0<=context['index']<len(self.methods)):
            previous=context['index']
        if previous is not None and context.get('stateful'):
            current=context.get('current');flat=flatten_integrated(state)
            current_uid=getattr(get_reference_state(state),'uid',None);old_uid=context.get('uid')
            if (not isinstance(current,torch.Tensor) or current.shape!=flat.shape
                    or current.dtype!=flat.dtype or current.device!=flat.device or not torch.equal(current,flat)
                    or context.get('layout')!=_layout(state) or (current_uid is None)!=(old_uid is None)
                    or current_uid is not None and not torch.equal(current_uid,old_uid)):
                previous=None
        base=snapshot_state(state);probe=snapshot_state(base)
        before=flatten_integrated(probe).clone();uid=snapshot_state(getattr(get_reference_state(probe),'uid',None))
        index=self.selector(probe,previous)
        after_uid=getattr(get_reference_state(probe),'uid',None)
        if (float(probe.t)!=float(base.t) or not torch.equal(before,flatten_integrated(probe))
                or (uid is None)!=(after_uid is None) or uid is not None and not torch.equal(uid,after_uid)):
            raise ValueError('Algorithm selector must preserve probe state values, time and IDs')
        if type(index) is not int or not 0<=index<len(self.methods):
            raise ValueError('selector must return a zero-based candidate index')
        child=self.methods[index]
        child_history=(StepHistory(accepted.maxlen,accepted.entries,context=context.get('child_context'))
            if previous==index else StepHistory(accepted.maxlen))
        if (child.metadata.history_requires_constant_dt and child_history.latest is not None
                and not math.isclose(child_history.latest.dt,dt,rel_tol=1e-12,abs_tol=0)):
            child_history=StepHistory(accepted.maxlen)
        result=child(base,dt,f,*args,history=child_history,**kwargs)
        if not result.step_status.successful:return replace(result,history=accepted)
        if (previous==index and child.metadata.history_length and child_history.latest is not None
                and not torch.equal(before,flatten_integrated(base))):
            raise ValueError('Warm switched method must preserve its initial integrated anchor; restart history')
        child_history=result.history if result.history is not None else StepHistory(accepted.maxlen)
        memory=StepHistory(accepted.maxlen,child_history.entries,context=dict(method=self.identifier.name,index=index,
            selected=child.identifier.name,t=float(result.state.t),child_context=child_history.context))
        memory.context['stateful']=bool(child.metadata.history_length)
        if memory.context['stateful']:
            memory.context.update(current=flatten_integrated(result.state).detach().clone(),layout=_layout(result.state),
                uid=snapshot_state(getattr(get_reference_state(result.state),'uid',None)))
        exponent=result.estimate_order if result.estimate_order is not None else child.metadata.estimate_order
        return replace(result,history=memory,estimate_order=exponent)

    def adapt_step(self,result,h,proposal,*,error_norm,norm_estimate):
        """Delegate accepted control to the selected child on unwrapped memory."""
        context=result.history.context;child=self.methods[context['index']]
        history=StepHistory(result.history.maxlen,result.history.entries,context=context['child_context'])
        child_result=replace(result,history=history)
        if hasattr(child,'adapt_step'):
            proposal=child.adapt_step(child_result,h,proposal,error_norm=error_norm,norm_estimate=norm_estimate)
        elif hasattr(child,'next_step_size'):
            proposal=child.next_step_size(child_result,h,proposal)
        context['child_context']=history.context
        if (isinstance(proposal,bool) or not isinstance(proposal,(int,float))
                or not math.isfinite(proposal) or proposal<=0):
            raise ValueError('Child adaptive policy must return a positive finite step')
        return min(proposal,h*child.max_step_ratio) if hasattr(child,'max_step_ratio') else proposal
