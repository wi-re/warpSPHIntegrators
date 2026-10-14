"""Autonomous physical operator splitting with one outer lifecycle boundary.

Factories are parameterized, unregistered schemes. Structural probes are local
checks, not a proof of a Hamiltonian or conservation. Production SPH compatible
energy/entropy coupling needs a composition-aware system and downstream baseline.
"""
from dataclasses import dataclass,fields
from types import SimpleNamespace
import hashlib
import math
import torch
from .catalog import SchemeMetadata
from .fields import (get_reference_state,find_tagged_field,integrated_field_names,
    flatten_integrated,unflatten_integrated)
from .integration import getIntegrator
from .protocol import BaseIntegrationSystem
from .rhs import RHS,resolve,ConservativeDissipativeRHS
from .specs import IntegrationResult,StageResult
from .util import (IntegrationScheme,initializeSystem,finalizeSystem,updateStep,split_return)


def _snapshot(value):
    from .run import snapshot_state
    return snapshot_state(value)


@dataclass(frozen=True)
class SeparabilityResult:
    separable: bool
    force_velocity_norm: float | None
    extra_quantity_rate: float | None
    reason: str


def classify_separable(rhs,state,*,dt=1.,probes=3,atol=1e-10,seed=0,rhs_args=(),rhs_kwargs=None):
    """Probe force velocity dependence and evolving extra quantities locally.

    Position rate must match the tagged velocity. Additional integrated fields
    must be passive under this conservative flow. This deliberately refuses the
    integrated-thermal-energy shortcut. Several perturbed velocities are tested;
    finite local probes do not prove global separability or a variational force.
    Unsupported AD must not be treated as a positive certificate.
    """
    if type(probes) is not int or probes<1 or not math.isfinite(atol) or atol<0:
        raise ValueError('positive probe count and finite nonnegative atol required')
    reference=get_reference_state(state)
    try:
        position=find_tagged_field(reference,tag='position');velocity=find_tagged_field(reference,tag='velocity')
    except (LookupError,TypeError):return SeparabilityResult(False,None,None,'no position/velocity tags')
    v=getattr(reference,velocity.name)
    if not isinstance(v,torch.Tensor):return SeparabilityResult(False,None,None,'velocity is not a tensor')
    def evaluate(value):
        current=_snapshot(state);ref=get_reference_state(current);setattr(ref,velocity.name,value)
        update,_=split_return(rhs(current,dt,*rhs_args,**(rhs_kwargs or {})))
        return update
    def acceleration(value):
        update=evaluate(value)
        return getattr(update,find_tagged_field(update,tag='velocity_derivative').name)
    generator=torch.Generator(device=v.device).manual_seed(seed)
    force=extra=0.;baseline_acceleration=None
    try:
        for i in range(probes):
            direction=torch.randn(v.shape,dtype=v.dtype,device=v.device,generator=generator)
            direction=direction/torch.linalg.vector_norm(direction).clamp_min(torch.finfo(v.dtype).eps)
            point=v if i==0 else v+direction*(1+torch.linalg.vector_norm(v))
            _,derivative=torch.func.jvp(acceleration,(point,),(direction,))
            if not torch.isfinite(derivative).all():return SeparabilityResult(False,float('inf'),None,'nonfinite force derivative')
            force=max(force,float(torch.linalg.vector_norm(derivative).detach()))
            update=evaluate(point)
            value=getattr(update,find_tagged_field(update,tag='velocity_derivative').name)
            if not torch.isfinite(value).all():return SeparabilityResult(False,float('inf'),None,'nonfinite acceleration')
            if baseline_acceleration is None:baseline_acceleration=value.detach().clone()
            else:
                distance=torch.linalg.vector_norm(point-v).clamp_min(torch.finfo(v.dtype).eps)
                force=max(force,float((torch.linalg.vector_norm(value-baseline_acceleration)/distance).detach()))
            drift=getattr(update,position.metadata['update_key'])
            if not torch.allclose(drift,point,rtol=0,atol=atol):
                return SeparabilityResult(False,force,extra,'position rate does not equal velocity')
            for field in fields(reference):
                if field.name not in (position.name,velocity.name) and field.name in integrated_field_names(state):
                    rate=getattr(update,field.metadata['update_key'])
                    if not torch.isfinite(rate).all():return SeparabilityResult(False,force,float('inf'),'nonfinite quantity rate')
                    extra=max(extra,float(torch.linalg.vector_norm(rate).detach()))
    except (RuntimeError,NotImplementedError,LookupError) as error:
        return SeparabilityResult(False,None,None,f'probe unsupported: {type(error).__name__}')
    passed=math.isfinite(force) and math.isfinite(extra) and force<=atol and extra<=atol
    return SeparabilityResult(passed,force,extra,'local structural probes pass' if passed else 'velocity-dependent force or evolving extra quantity')


def check_split_structure(rhs,state,dt,*,atol=1e-10,require_separable=False,rhs_args=(),
                          rhs_kwargs=None,conservative_kwargs=None,dissipative_kwargs=None):
    resolved=resolve(rhs,scheme_name='Composition',need_conservative=True,need_dissipative=True)
    reference=get_reference_state(state)
    try:position=find_tagged_field(reference,tag='position')
    except (LookupError,TypeError):position=None
    if position is not None and not getattr(state,'_tensor_adapter',False):
        probes=[_snapshot(state)]
        try:
            velocity=find_tagged_field(reference,tag='velocity')
            for sign in (-1,1):
                probe=_snapshot(state);ref=get_reference_state(probe)
                setattr(ref,velocity.name,getattr(ref,velocity.name)+sign)
                probes.append(probe)
        except LookupError:pass
        for probe in probes:
            options=rhs_kwargs if dissipative_kwargs is None else dissipative_kwargs
            update,_=split_return(resolved.dissipative(probe,dt,*rhs_args,**(options or {})))
            rate=getattr(update,position.metadata['update_key'])
            if not torch.isfinite(rate).all() or not torch.allclose(rate,torch.zeros_like(rate),atol=atol,rtol=0):
                raise ValueError('Dissipative flow must preserve configuration (zero position rate)')
    if require_separable:
        options=rhs_kwargs if conservative_kwargs is None else conservative_kwargs
        verdict=classify_separable(resolved.conservative,state,dt=dt,atol=atol,rhs_args=rhs_args,rhs_kwargs=options)
        if not verdict.separable:raise ValueError(f'Conservative flow is not structurally separable: {verdict.reason}')


@dataclass(frozen=True)
class LinearExponentialFlow:
    """Dense exact flow for a declared linear part, with full matrix gradients.

    operator is an N-by-N tensor or callable(state)->tensor in flattened field
    order. This is an explicit small-system path; supply a different flow/solver
    for large sparse SPH systems. The matrix is frozen within each autonomous
    subflow and must agree with the declared homogeneous linear RHS.
    """
    operator: object
    name: str = 'LinearExponentialFlow'
    order: int = 100
    implicit: bool = False
    metadata: object = SchemeMetadata()
    def __call__(self,state,dt,rhs,*args,**kwargs):
        if not isinstance(rhs,RHS) or 'linear' not in rhs.provides:
            raise TypeError('LinearExponentialFlow requires a declared linear RHS')
        initializeSystem(state,dt,*args,**kwargs)
        stage=state.initializeNewState(*args,**kwargs)
        update,aux=updateStep(state,stage,dt,rhs,*args,**kwargs)
        matrix=self.operator(state) if callable(self.operator) else self.operator
        y=flatten_integrated(state)
        if not isinstance(matrix,torch.Tensor) or matrix.shape!=(y.numel(),y.numel()) or matrix.dtype!=y.dtype or matrix.device!=y.device or not torch.isfinite(matrix).all():
            raise ValueError('Linear flow matrix must be finite and match flattened shape/dtype/device')
        # Check the operator's declared action at this state, without replacing
        # its differentiable matrix parameter path by a frozen-action surrogate.
        from .util import updateStateEuler
        derivative=flatten_integrated(updateStateEuler(state,update,1.))-y
        if not torch.allclose(matrix@y,derivative,rtol=1e-8,atol=1e-10):
            raise ValueError('Linear flow matrix does not match declared RHS action')
        output=unflatten_integrated(torch.matrix_exp(dt*matrix)@y,state);output.t=float(state.t+dt)
        finalizeSystem(output,state,dt,[aux],[update],[1.],*args,lastStageSystem=stage,**kwargs)
        return IntegrationResult(output,stages=[StageResult(aux,update)])


@dataclass(frozen=True)
class SubstepRecord:
    part: str
    coefficient: float
    result: IntegrationResult


@dataclass(frozen=True)
class SplitScheme:
    parts: tuple
    coefficients: tuple
    order: int
    name: str
    conservative_options: dict
    dissipative_options: dict
    negative_dissipation_limit: float | None = None
    @property
    def identifier(self):
        description=repr((self.name,tuple((part,method.name) for part,method in self.parts),self.coefficients))
        return SimpleNamespace(name='split_'+hashlib.sha256(description.encode()).hexdigest()[:20])
    @property
    def implicit(self):return any(method.implicit for _,method in self.parts)
    @property
    def fsal(self):return False
    @property
    def has_error_estimate(self):return False
    @property
    def metadata(self):
        partitioned=any(method.metadata.state_kind=='second_order' for _,method in self.parts)
        return SchemeMetadata(required_capabilities=frozenset({'conservative','dissipative'}),
                              state_kind='second_order' if partitioned else 'tensor',requires_separable=partitioned)
    def __call__(self,state,dt,rhs,*args,**kwargs):
        if isinstance(dt,bool) or not math.isfinite(dt) or dt==0:raise ValueError('finite nonzero dt required')
        if not isinstance(rhs,RHS) or not rhs.autonomous:raise ValueError('Composition requires a declared autonomous physical split')
        if any(isinstance(part,RHS) and not part.autonomous for part in (rhs._conservative,rhs._dissipative)):
            raise ValueError('Every composition part must declare autonomous dynamics')
        if self.negative_dissipation_limit is not None:
            for (part,method),coefficient in zip(self.parts,self.coefficients):
                if part!='dissipative':continue
                if not isinstance(method,LinearExponentialFlow) or not isinstance(method.operator,torch.Tensor):
                    raise ValueError('Negative dissipative steps require a static exact linear flow')
                matrix=method.operator.detach()
                if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not torch.isfinite(matrix).all():
                    raise ValueError('Finite square dissipative matrix required')
                if float(torch.linalg.eigvalsh((matrix+matrix.T)/2).max())>1e-12:
                    raise ValueError('Matrix must be dissipative in the Euclidean norm')
                if coefficient*dt<0 and abs(coefficient*dt)*float(torch.linalg.matrix_norm(matrix,ord=2))>self.negative_dissipation_limit:
                    raise ValueError('Negative dissipative step exceeds the declared weak-dissipation limit')
        require=self.metadata.requires_separable
        if getattr(state,'_tensor_adapter',False) and require:raise ValueError('Geometric split needs position/velocity fields')
        check_split_structure(rhs,state,dt,require_separable=require,rhs_args=args,
            conservative_kwargs={**kwargs,**self.conservative_options},
            dissipative_kwargs={**kwargs,**self.dissipative_options})
        hook=getattr(state.finalize,'__func__',None)
        if hook is not BaseIntegrationSystem.finalize and not getattr(state,'composition_compatible',False):
            raise TypeError('Custom finalize requires composition_compatible=True and grouped substep support')
        if kwargs.get('priorStep') is not None:raise ValueError('Composition does not reuse priorStep')
        original=_snapshot(state);initializeSystem(original,dt,*args,**kwargs)
        current=_snapshot(original);records=[];stages=[];diagnostics=[]
        for (part,method),coefficient in zip(self.parts,self.coefficients):
            if coefficient==0:continue
            current.t=float(original.t)  # autonomous pseudo-time, never sum both flow clocks
            start=_snapshot(current)
            options=dict(kwargs)
            options.update(self.conservative_options if part=='conservative' else self.dissipative_options)
            options.update(_composition_substep=True,split_part=part)
            options.pop('history',None);options.pop('priorStep',None)
            result=method(current,coefficient*dt,getattr(rhs,'_'+part),*args,**options)
            stages.extend(result.stages)
            diags=result.solver_diagnostics
            if diags is not None:diagnostics.extend(d for d in (diags if isinstance(diags,(tuple,list)) else [diags]) if d is not None)
            records.append(SubstepRecord(part,coefficient,result))
            if not result.step_status.successful:
                failed=result.state;failed.t=float(original.t+dt)
                return IntegrationResult(failed,stages=stages,solver_diagnostics=diagnostics)
            if part=='dissipative' and not getattr(state,'_tensor_adapter',False):
                try:field=find_tagged_field(get_reference_state(state),tag='position')
                except LookupError:field=None
                if field is not None:
                    before=getattr(get_reference_state(start),field.name)
                    after=getattr(get_reference_state(result.state),field.name)
                    if isinstance(method,LinearExponentialFlow):
                        # Dense exp can perturb an identity row by rounding.
                        # The declared D flow has already passed the zero-rate
                        # checks; restore that invariant only within roundoff.
                        eps=torch.finfo(before.dtype).eps
                        if not torch.allclose(before,after,rtol=16*eps,atol=16*eps):
                            raise ValueError('Dissipative sub-integrator changed configuration')
                        setattr(get_reference_state(result.state),field.name,before.clone())
                    elif not torch.equal(before,after):
                        raise ValueError('Dissipative sub-integrator changed configuration')
            current=_snapshot(result.state)
        current.t=float(original.t+dt)
        finalizeSystem(current,original,dt,[r.result for r in records],
            [r.result.stages for r in records],[r.coefficient for r in records],*args,
            composition_substeps=tuple(records),**kwargs)
        return IntegrationResult(current,stages=stages,solver_diagnostics=diagnostics)


def compose(parts,coefficients,*,order=1,name='Composition',conservative_options=None,dissipative_options=None):
    """Build a fixed-step autonomous composition; no multistep or negative D weights.

    parts is a sequence of (physical-part name, one-step scheme/linear flow).
    Each part's coefficients must sum to one. Second-order recipes must be
    palindromic; higher-order dissipative compositions remain gated.
    """
    if type(order) is not int or order not in (1,2):raise ValueError('Supported composition orders are 1 and 2')
    from .nystrom import Nystrom
    from .stabilized import StabilizedRK
    from .collocation import Collocation
    from .extrapolation import Extrapolation
    from .expanded_rk import ExpandedRK
    bound=(IntegrationScheme,LinearExponentialFlow,Nystrom,StabilizedRK,Collocation,Extrapolation,ExpandedRK)
    entries=tuple((part,method if isinstance(method,bound) else getIntegrator(method)) for part,method in parts)
    weights=tuple(float(c) for c in coefficients)
    if not entries or len(entries)!=len(weights) or any(not math.isfinite(c) for c in weights):raise ValueError('Matching finite parts/coefficients required')
    for part,method in entries:
        if part not in ('conservative','dissipative') or method.metadata.history_length:
            raise ValueError('Physical parts require one-step sub-integrators')
    for part in ('conservative','dissipative'):
        if not math.isclose(sum(c for (p,_),c in zip(entries,weights) if p==part),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Each physical part must advance one full step')
    if any(c<0 and part=='dissipative' for (part,_),c in zip(entries,weights)):
        raise ValueError('Negative dissipative composition coefficients are gated')
    if order==2 and (entries!=entries[::-1] or weights!=weights[::-1]):raise ValueError('Second-order recipe must be palindromic')
    actual=min(order,*(method.order for _,method in entries))
    return SplitScheme(entries,weights,actual,name,dict(conservative_options or {}),dict(dissipative_options or {}))


def LieTrotter(conservative='RK4',dissipative='Implicit Midpoint',**options):
    return compose([('conservative',conservative),('dissipative',dissipative)],[1.,1.],name='LieTrotter',**options)


def Strang(conservative='RK4',dissipative='Implicit Midpoint',**options):
    return compose([('dissipative',dissipative),('conservative',conservative),('dissipative',dissipative)],
                   [.5,1.,.5],order=2,name='Strang',**options)


def Yoshida4(conservative='Implicit Midpoint',dissipative=None,*,negative_step_limit=.5,
             conservative_options=None,dissipative_options=None):
    """Fourth-order triple jump, gated on symmetric H and static exact linear D.

    The user-visible anti-diffusion bound is |tau_negative|*||D||_2 <= limit;
    its default .5 is an engineering restriction, not unconditional stability.
    Generic/nonlinear/stiff dissipative compositions remain unsupported.
    """
    h=conservative if isinstance(conservative,IntegrationScheme) else getIntegrator(conservative)
    symmetric={'Implicit Midpoint','Trapezoidal (Crank-Nicolson)','Velocity Verlet','Leap Frog','PEFRL','VEFRL'}
    if h.name not in symmetric:raise ValueError('Yoshida4 requires a supported symmetric conservative integrator')
    if not isinstance(dissipative,LinearExponentialFlow) or not isinstance(dissipative.operator,torch.Tensor):
        raise ValueError('Yoshida4 requires a static exact linear dissipative flow')
    if isinstance(negative_step_limit,bool) or not math.isfinite(negative_step_limit) or negative_step_limit<=0:
        raise ValueError('Finite positive negative_step_limit required')
    w=1/(2-2**(1/3));middle=1-2*w
    entries=(('dissipative',dissipative),('conservative',h),('dissipative',dissipative),
             ('conservative',h),('dissipative',dissipative),('conservative',h),('dissipative',dissipative))
    weights=(w/2,w,(w+middle)/2,middle,(middle+w)/2,w,w/2)
    return SplitScheme(entries,weights,4,'Yoshida4',dict(conservative_options or {}),
                       dict(dissipative_options or {}),float(negative_step_limit))
