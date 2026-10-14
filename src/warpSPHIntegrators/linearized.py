"""Source-qualified Rosenbrock maps with exact Jacobian actions and adjoints."""
from dataclasses import dataclass, fields, replace
from functools import lru_cache
import math
from types import SimpleNamespace

import torch

from ._rosenbrock_data import W6S4OS
from .catalog import SchemeMetadata
from .fields import get_reference_state, integrated_field_names, flatten_integrated, unflatten_integrated
from .jfnk import gmres, _implicit_diff_reattach
from .protocol import BaseIntegrationSystem
from .solvers import SolveDiagnostics, validate_solver_options
from .specs import IntegrationResult, StageResult
from .tensor import TensorState
from .util import initializeSystem, finalizeSystem, updateStep, reject_prior_step, split_return


@dataclass(frozen=True)
class RosenbrockTableau:
    """Rodas-form increments: M*K_i=h*gamma*f_i+gamma*sum(C_ij*K_j)+h²*gamma*d_i*f_t."""
    A: tuple
    C: tuple
    gamma: float
    c: tuple
    d: tuple
    b: tuple
    btilde: tuple|None
    order: int


@lru_cache(maxsize=3)
def rosenbrock_tableau(method):
    if method=='RosenbrockW6S4OS':
        return RosenbrockTableau(**W6S4OS,order=4)
    if method not in ('Rosenbrock23','Rosenbrock32'):raise ValueError('Unknown Rosenbrock method')
    g=1/(2+math.sqrt(2));c32=6+math.sqrt(2)
    # Algebraic change of source slopes: K1=h*g*k1, K2=h*g*(k2-k1),
    # K3=h*g*k3-2*K1-c32*K2. This preserves both main maps and raw defects.
    A=((0.,0.,0.),(1/(2*g),0.,0.),(1/g,1/g,0.))
    C=((0.,0.,0.),(-1/g,0.,0.),(-2/g,-c32/g,0.))
    low=(1/g,1/g,0.);high=(7/(6*g),(4+c32)/(6*g),1/(6*g))
    # Third-stage partial-time coefficient is corrected by exact autonomous
    # augmentation t'=1: source dt*f_t must be dt*g*f_t. In these variables
    # d3=-g (the uncorrected source would give 1-2*g and an O(h²) defect).
    return RosenbrockTableau(A,C,g,(0.,.5,1.),(g,0.,-g),
        low if method=='Rosenbrock23' else high,(1/(6*g),(c32-2)/(6*g),1/(6*g)),
        2 if method=='Rosenbrock23' else 3)


@dataclass(frozen=True)
class RosenbrockQuadrature:
    """Weights over flat, already h-scaled solved increments, paired with stages.

    flatten(output)=flatten(initial)+sum(weights[i]*increments[i]).
    Raw stage updates are physical rates, not the solved increments. Endpoint
    refresh has a zero increment and weight. Names declare the flattening order.
    """
    weights: tuple
    increments: tuple
    field_names: tuple


@dataclass(frozen=True)
class Rosenbrock:
    """Rosenbrock23/32 or W6S4OS using exact full-RHS JVP and linear GMRES stages.

    Rosenbrock23/32 correct the pinned third-stage time coefficient by autonomous
    augmentation; this is separate from unchanged autonomous-source equivalence.
    No mass matrices, approximate-W substitution, FSAL or source cache controller.
    Default f_t uses isolated central time probes; a callable f_t returns an
    exact partial-time update. solver_opts accepts GMRES/preconditioner controls.
    Implicit adjoints retain derivatives through J, including Hessian and RHS-
    parameter terms. Custom finalize must opt into rosenbrock_compatible and
    consume RosenbrockQuadrature, whose increments differ from raw RHS rates.
    """
    method: str='Rosenbrock23'

    def __post_init__(self):rosenbrock_tableau(self.method)
    @property
    def name(self):return self.method
    @property
    def identifier(self):return SimpleNamespace(name='rosenbrock_full_jvp_'+self.method)
    @property
    def order(self):return rosenbrock_tableau(self.method).order
    @property
    def implicit(self):return True
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return rosenbrock_tableau(self.method).btilde is not None
    @property
    def estimate_order(self):return 3 if self.has_error_estimate else None
    @property
    def metadata(self):return SchemeMetadata(stage_count=len(rosenbrock_tableau(self.method).b)+1,
        embedded_order=(3 if self.order==2 else 2) if self.has_error_estimate else None,
        estimate_order=self.estimate_order)

    def __call__(self,state,dt,f,*args,solver_opts=None,f_t=None,ft_eps=None,**kwargs):
        from .run import snapshot_state, assess_step
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        reject_prior_step(self.name,kwargs.pop('priorStep',None));kwargs.pop('history',None)
        if f_t is not None and not callable(f_t):raise TypeError('f_t must be callable or None')
        opts=dict(solver_opts or {});validate_solver_options(opts);validate_solver_options({'ft_eps':ft_eps})
        tol=opts.get('gmres_tol') or opts.get('tol') or 1e-12;restart=opts.get('gmres_restart',30)
        prec=opts.get('preconditioner');side=opts.get('preconditioning','right')
        base=snapshot_state(state);t=float(base.t)
        if not math.isfinite(t):raise ValueError('finite initial time required')
        hook=getattr(base.finalize,'__func__',None)
        if not kwargs.get('_composition_substep',False) and hook is not BaseIntegrationSystem.finalize and not getattr(base,'rosenbrock_compatible',False):
            raise TypeError('Custom finalize requires rosenbrock_compatible=True and RosenbrockQuadrature support')
        initializeSystem(base,dt,*args,**kwargs)
        t=float(base.t)
        if not math.isfinite(t):raise ValueError('finite initialized time required')
        ref=get_reference_state(base);names=tuple(integrated_field_names(base))
        declared={field.name:field for field in fields(ref)}
        y=flatten_integrated(base);n=y.numel()
        if n==0 or not y.is_floating_point():raise ValueError('Nonempty floating integrated fields required')
        if any(not isinstance(getattr(ref,name),torch.Tensor) or not getattr(ref,name).is_floating_point()
                or getattr(ref,name).dtype!=y.dtype or getattr(ref,name).device!=y.device for name in names):
            raise ValueError('Integrated fields must be floating tensors with a shared dtype and device')
        budget=opts.get('gmres_maxiter') or n
        calls=0
        def flat_update(update):
            values=[]
            for name in names:
                value=getattr(update,declared[name].metadata['update_key'])
                template=getattr(ref,name)
                if value.shape!=template.shape or value.dtype!=template.dtype or value.device!=template.device:
                    raise ValueError('Rosenbrock rates must match integrated field layouts')
                values.append(value.reshape(-1))
            return torch.cat(values)
        def evaluate(z,time,*,probe=False):
            nonlocal calls
            calls+=1;anchor=snapshot_state(base) if probe else base
            stage=anchor.initializeNewState(*args,**kwargs);current=get_reference_state(stage);offset=0
            for name in names:
                template=getattr(ref,name);size=template.numel()
                setattr(current,name,z[offset:offset+size].reshape(template.shape));offset+=size
            stage.t=float(time)
            update,aux=updateStep(anchor,stage,dt,f,*args,**kwargs)
            return flat_update(update),StageResult(aux,update),stage
        def J(v):
            return torch.func.jvp(lambda z:evaluate(z,t,probe=True)[0],(y,),(v,))[1]
        table=rosenbrock_tableau(self.method);g=table.gamma;h=dt
        initial,initial_stage,initial_state=evaluate(y,t)
        if f_t is None:
            eps=ft_eps or torch.finfo(y.dtype).eps**(1/3)*max(1.,abs(t),abs(h))
            plus,_,_=evaluate(y,t+eps,probe=True);minus,_,_=evaluate(y,t-eps,probe=True)
            partial=(plus-minus)/(2*eps)
        else:
            update,_=split_return(f_t(snapshot_state(initial_state),dt,*args,**kwargs));partial=flat_update(update)
        increments=[];stages=[];diagnostics=[]
        M=lambda v:v-h*g*J(v)
        previous_count=0
        for i,c in enumerate(table.c):
            z=y
            for a,k in zip(table.A[i],increments):
                if a:z=z+a*k
            rate,stage,_=evaluate(z,t+c*h) if i else (initial,initial_stage,None)
            stages.append(stage);rhs=h*g*rate+h*h*g*table.d[i]*partial
            for coefficient,k in zip(table.C[i],increments):
                if coefficient:rhs=rhs+g*coefficient*k
            if not torch.isfinite(rhs).all():
                diagnostics.append(SolveDiagnostics(converged=False,termination='invalid_residual',rhs_evaluations=calls-previous_count))
                break
            context=dict(opts.get('preconditioner_context') or {})
            context.update(dt=h,stage_index=i,stage_time=t+c*h,stage_coefficient=h*g,state=base)
            bound_prec=(lambda v:prec(v,base,context)) if prec is not None else None
            with torch.no_grad():
                value,iterations=gmres(M,rhs,tol=tol,maxiter=budget,restart=restart,
                    preconditioner=bound_prec,preconditioning=side)
                residual=float(torch.linalg.vector_norm(M(value)-rhs))
                successful=math.isfinite(residual) and residual<=tol*max(1.,float(torch.linalg.vector_norm(rhs)))
            if torch.is_grad_enabled() and (rhs.requires_grad or y.requires_grad) and successful:
                def linear_step(state,rhs=rhs):return TensorState(rhs+h*g*J(state.y))
                value=_implicit_diff_reattach(linear_step,TensorState(value),gmres_tol=tol,
                    gmres_maxiter=budget,gmres_restart=restart).y
            increments.append(value)
            diagnostics.append(SolveDiagnostics(residual=residual,gmres_iterations=int(iterations),
                rhs_evaluations=calls-previous_count,termination='tolerance' if successful else 'max_iterations',converged=successful))
            previous_count=calls
            if not successful:break
        output=y
        for b,k in zip(table.b,increments):
            if b:output=output+b*k
        result_state=unflatten_integrated(output,base);result_state.t=float(t+h)
        result=IntegrationResult(result_state,stages=stages,solver_diagnostics=diagnostics)
        if not assess_step(result).successful:return result
        error=None
        if table.btilde is not None:
            defect=sum((b*k for b,k in zip(table.btilde,increments)),torch.zeros_like(y))
            error=unflatten_integrated(defect,base);error.t=result_state.t
        _,endpoint_stage,endpoint=evaluate(output,t+h)
        stages.append(endpoint_stage)
        diagnostics[-1]=replace(diagnostics[-1],rhs_evaluations=diagnostics[-1].rhs_evaluations+calls-previous_count)
        weights=RosenbrockQuadrature((*table.b,0.),(*increments,torch.zeros_like(y)),names)
        result=replace(result,error=error,estimate_order=self.estimate_order,quadrature=weights)
        if assess_step(result).successful:
            finalizeSystem(result_state,base,h,[s.aux for s in stages],[s.update for s in stages],weights,
                *args,lastStageSystem=endpoint,**kwargs)
        return result
