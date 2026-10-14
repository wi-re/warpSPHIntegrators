"""Source Nyström maps, structural boundaries and component quadrature contracts."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from dataclasses import dataclass

from warpSPHIntegrators import (Nystrom,nystrom_tableau,NystromWeights,testing,solve,
    BaseState,integrated,copied,constant,reference_state,Strang,StabilizedRK,Collocation)

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave12/nystrom.json').read_text())
METHODS=[name for name in SOURCE['tableaus'] if name!='RKN4']


def initial(q=(1.,),v=(1.,),e=None,t=0.):
    q=torch.tensor(q,dtype=torch.float64);v=torch.tensor(v,dtype=torch.float64)
    return testing.ParticleSystem(testing.ParticleState(x=q,u=v,
        e=torch.zeros_like(q) if e is None else torch.tensor(e,dtype=torch.float64),m=torch.ones_like(q)),t=t)


@pytest.mark.parametrize('name',METHODS)
@pytest.mark.parametrize('forced',[False,True])
def test_tableaus_and_execution_match_pinned_source_step_bodies(name,forced):
    record=SOURCE['records'][name+('_forced' if forced else '_autonomous')]
    source=SOURCE['tableaus'][name];calls=[]
    tableau=nystrom_tableau(name)
    for key in ['a','b','bp','btilde','bptilde']:
        assert np.array_equal(np.asarray(getattr(tableau,key)),np.asarray(source[key]))
    assert tableau.c==(0.,*source['c'])
    if source['velocity_dependent']:assert np.array_equal(tableau.abar,source['abar'])
    def rhs(state,dt,**kwargs):
        q,v=state.state.x,state.state.u;t=state.t
        a=-2*q+.1*q*q+(torch.tensor([.2*math.sin(t),.1*math.cos(t)],dtype=q.dtype) if forced else 0)
        if source['velocity_dependent']:a=a-.3*v+.05*q*v+.1*v*v
        calls.append(dict(t=t,q=q.clone(),v=v.clone(),a=a.clone()))
        return testing.ParticleUpdate(v,a,torch.zeros_like(state.state.e)),None
    state=initial(record['initial_q'],record['initial_v'],t=record['t'])
    result=Nystrom(name,structure='declared')(state,record['h'],rhs)
    assert result.step_status.successful
    assert result.state.state.x.tolist()==pytest.approx(record['q'],abs=2e-13)
    assert result.state.state.u.tolist()==pytest.approx(record['v'],abs=2e-13)
    for actual,expected in zip(calls,record['calls']):
        assert actual['t']==pytest.approx(expected['t'],abs=1e-14)
        for key in ['q','v','a']:assert actual[key].tolist()==pytest.approx(expected[key],abs=2e-13)
    assert len(calls)==len(record['calls'])==Nystrom(name).metadata.stage_count
    weights=result.quadrature
    assert len(weights.position)==len(weights.velocity)==len(weights.quantity)==len(result.stages)
    assert sum(weights.position)==pytest.approx(.5,abs=1e-12)
    assert sum(weights.velocity)==pytest.approx(1.,abs=1e-12)
    if result.error is not None:
        assert result.error.state.x.tolist()==pytest.approx(record['error_q'],abs=2e-13)
        assert result.error.state.u.tolist()==pytest.approx(record['error_v'] or [0.,0.],abs=2e-13)
    else:assert not record['error_q'] and not record['error_v']


def manufactured(name):
    dependent=nystrom_tableau(name).abar is not None
    def rhs(state,dt,**kwargs):
        q,v=state.state.x,state.state.u;target=math.exp(state.t)
        a=-q+.1*q*q+2*target-.1*target*target
        if dependent:a=a-.3*v+.05*q*v+.1*v*v+.3*target-.15*target*target
        return testing.ParticleUpdate(v,a,torch.zeros_like(state.state.e)),None
    return rhs


@pytest.mark.parametrize('name',[name for name in METHODS if name!='DPRKN12'])
def test_nonlinear_nonautonomous_order(name):
    # FineRKN4 reaches its asymptotic regime later on this general VD force.
    hs=[.125,.0625,.03125,.015625] if name=='FineRKN4' else [.5,.25,.125,.0625]
    errors=[];method=Nystrom(name,structure='declared')
    for h in hs:
        result=solve(method,initial(),1.,manufactured(name),dt=h)
        assert result.successful
        errors.append(max(abs(result.state.state.x.item()-math.e),abs(result.state.state.u.item()-math.e)))
    usable=[(e,h) for e,h in zip(errors,hs) if e>1e-12]
    assert len(usable)>=2
    order=math.log(usable[-2][0]/usable[-1][0],2)
    assert order>=method.order-.15,(name,errors,order)


def test_dprkn12_oscillator_coarse_grid_above_roundoff_floor():
    errors=[];method=Nystrom('DPRKN12',structure='declared')
    for h in [2.,1.,.5,.25]:
        def rhs(state,dt,**kwargs):
            s=state.state;return testing.ParticleUpdate(s.u,-16*s.x,torch.zeros_like(s.e)),None
        result=solve(method,initial(q=[1.],v=[.2]),2.,rhs,dt=h)
        q=math.cos(8)+.05*math.sin(8);v=-4*math.sin(8)+.2*math.cos(8)
        errors.append(max(abs(result.state.state.x.item()-q),abs(result.state.state.u.item()-v)))
    order=math.log(errors[-2]/errors[-1],2)
    assert 11.7<order<13.5,(errors,order)


def test_pinned_rkn4_velocity_order_loss_and_vi_equivalence_are_explicit():
    tab=SOURCE['tableaus']['RKN4'];abar=np.array(tab['abar']);bp=np.array(tab['bp']);c=np.r_[0,tab['c']]
    assert bp@abar@c==pytest.approx(1/12)  # required 1/6 even for v'=f(v)
    with pytest.raises(ValueError,match='order two'):Nystrom('RKN4')
    vi=SOURCE['tableaus']['Nystrom4VelocityIndependent']
    for key in ['a','b','bp','c']:assert tab[key]==vi[key]


@pytest.mark.parametrize('name',METHODS)
def test_oscillator_polynomial_conditions_through_the_claimed_order(name):
    tab=nystrom_tableau(name);a=np.array(tab.a);b=np.array(tab.b);bp=np.array(tab.bp);c=np.array(tab.c)
    ones=np.ones(len(c));power=np.eye(len(c))
    for k in range(1,tab.order//2+1):
        # Coefficients in powers of z=-omega²*h² for the q/(h*v) map.
        assert (b@power@ones)*math.factorial(2*k)==pytest.approx(1,abs=3e-12)
        assert (bp@power@c)*math.factorial(2*k)==pytest.approx(1,abs=3e-12)
        assert (bp@power@ones)*math.factorial(2*k-1)==pytest.approx(1,abs=3e-12)
        if 2*k+1<=tab.order:
            assert (b@power@c)*math.factorial(2*k+1)==pytest.approx(1,abs=3e-12)
        power=power@a


@pytest.mark.parametrize('name',[name for name in METHODS if nystrom_tableau(name).abar is None])
def test_velocity_independent_guards_reject_drag_and_evolving_thermal_fields(name):
    method=Nystrom(name)
    def drag(state,dt,**kwargs):
        s=state.state;return testing.ParticleUpdate(s.u,-s.x-.1*s.u,torch.zeros_like(s.e)),None
    with pytest.raises(ValueError,match='independent acceleration'):method(initial(),.1,drag)
    def thermal(state,dt,**kwargs):
        s=state.state;return testing.ParticleUpdate(s.u,-s.x,s.e+1),None
    for structure in ['probe','declared']:
        with pytest.raises(ValueError,match='passive extra fields'):Nystrom(name,structure)(initial(),.1,thermal)


@pytest.mark.parametrize('name',['Nystrom4','FineRKN4','FineRKN5'])
def test_velocity_tableaus_lift_coupled_first_order_quantities_at_full_order(name):
    def rhs(state,dt,**kwargs):
        s=state.state;t=state.t;target=math.exp(t)
        return testing.ParticleUpdate(s.u,-s.x-.3*s.u+.1*s.e*s.e+2.3*target-.1*target*target,
            s.e+.1*s.x*s.u-.1*target*target),None
    errors=[];method=Nystrom(name)
    for h in [.1,.05,.025]:
        result=solve(method,initial(e=[1.]),1.,rhs,dt=h)
        errors.append(max(abs(getattr(result.state.state,key).item()-math.e) for key in ['x','u','e']))
    assert math.log(errors[-2]/errors[-1],2)>=method.order-.2,(name,errors)


@pytest.mark.parametrize('name',[name for name in METHODS if Nystrom(name).has_error_estimate])
def test_difference_exponents_from_independent_time_polynomial_moments(name):
    method=Nystrom(name,structure='declared');q=method.estimate_order
    for field,degree in [('x',q-2),('u',q-1)]:
        if field=='u' and method.tableau.position_only_estimate:continue
        errors=[]
        for h in [.5,.25,.125]:
            def rhs(state,dt,**kwargs):
                s=state.state;return testing.ParticleUpdate(s.u,torch.full_like(s.u,state.t**degree),torch.zeros_like(s.e)),None
            result=method(initial(q=[0.],v=[0.]),h,rhs)
            errors.append(abs(getattr(result.error.state,field).item()))
        assert errors[0]>0 and math.log(errors[-2]/errors[-1],2)==pytest.approx(q,abs=1e-8)


@pytest.mark.parametrize('name',METHODS)
def test_initial_and_parameter_gradients(name):
    def forward(value,tracking=False):
        state=initial(q=[.7],v=[-.2]);state.state.x.requires_grad_(tracking)
        k=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
        def rhs(state,dt,**kwargs):
            s=state.state;a=-k*s.x-.1*s.x**3
            if nystrom_tableau(name).abar is not None:a=a-.2*s.u
            return testing.ParticleUpdate(s.u,a,torch.zeros_like(s.e)),None
        result=solve(Nystrom(name),state,.3,rhs,dt=.05)
        return result.state.state.x.sum(),state.state.x,k
    value,x,k=forward(2.,True);dx,dk=torch.autograd.grad(value,(x,k));eps=1e-5
    finite=(forward(2+eps)[0]-forward(2-eps)[0])/(2*eps)
    assert torch.isfinite(dx).all()
    assert dk.item()==pytest.approx(finite.item(),rel=2e-7,abs=1e-9)


@pytest.mark.parametrize('name',['DPRKN6','FineRKN5','ERKN5','ERKN7'])
def test_adaptive_rejection_accuracy_refinement_and_in_memory_restart(name):
    method=Nystrom(name);rhs=manufactured(name);errors=[]
    for tolerance in [1e-3,1e-7]:
        first=solve(method,initial(),.5,rhs,dt=1.,adaptive=True,rtol=tolerance,atol=tolerance*.01)
        resumed=solve(method,None,1.,rhs,restart=first.checkpoint)
        assert resumed.successful
        errors.append(max(abs(resumed.state.state.x.item()-math.e),abs(resumed.state.state.u.item()-math.e)))
        direct=solve(method,None,1.,rhs,restart=first.checkpoint)
        assert torch.equal(resumed.state.state.x,direct.state.state.x)
    assert errors[-1]<errors[0]/5 and errors[-1]<1e-5,(name,errors)


@pytest.mark.parametrize('name',['Nystrom4','Nystrom4VelocityIndependent','Nystrom5VelocityIndependent'])
def test_nonembedded_methods_refuse_adaptive_runs(name):
    with pytest.raises(ValueError,match='adaptive|Adaptive|estimate'):
        solve(Nystrom(name),initial(),1.,manufactured(name),dt=.1,adaptive=True)


@dataclass
class Fields(BaseState):
    x: torch.Tensor=integrated('dxdt',tags=('position',))
    u: torch.Tensor=integrated('dudt',tags=('velocity',))
    e: torch.Tensor=integrated('dedt',tags=('quantity',))
    uid: torch.Tensor=constant()
    density: torch.Tensor=copied()


@dataclass
class System(testing.ParticleSystem):
    state: Fields=reference_state()
    nystrom_compatible=True
    composition_compatible=True
    initialized: int=0
    finalized: int=0
    def initializeNewState(self,*args,**kwargs):
        return System(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
    def initialize(self,dt,**kwargs):self.initialized+=1;return self
    def preprocess(self,initial,dt,**kwargs):self.state.density=self.state.x.square()+self.state.e;return self
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        if 'composition_substeps' in kwargs:
            conservative=[r for r in kwargs['composition_substeps'] if r.part=='conservative']
            assert isinstance(conservative[0].result.quadrature,NystromWeights)
        else:
            assert isinstance(weights,NystromWeights)
            assert len(auxes)==len(updates)==len(weights.position)==len(weights.velocity)
            expected=initial.state.x+dt*initial.state.u+dt*dt*sum(w*k.dudt for w,k in zip(weights.position,updates))
            assert torch.allclose(self.state.x,expected,atol=1e-13,rtol=0)
        self.finalized=initial.finalized+1
        return self


def system():
    return System(Fields(torch.ones(2,dtype=torch.float64),torch.full((2,),.2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))


@pytest.mark.parametrize('name',['DPRKN6','FineRKN5','Nystrom4VelocityIndependent'])
def test_lifecycle_quadrature_copied_endpoint_input_ownership_and_failed_trials(name):
    start=system();vi=nystrom_tableau(name).abar is None
    def rhs(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x if vi else -s.x-.2*s.u+.1*s.e,
            torch.zeros_like(s.e) if vi else s.e+.2*s.x*s.u),None
    method=Nystrom(name);result=solve(method,start,.2,rhs,dt=.05)
    assert result.successful and result.state.initialized==result.state.finalized==4
    assert start.initialized==start.finalized==0
    assert torch.equal(start.state.density,torch.zeros(2,dtype=torch.float64))
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert torch.equal(result.state.state.uid,start.state.uid)
    def broken(state,dt,**kwargs):
        update,aux=rhs(state,dt)
        if state.t>=.025:update.dudt=update.dudt*float('nan')
        return update,aux
    failed=solve(method,start,.2,broken,dt=.05,raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0
    assert failed.state.initialized==failed.state.finalized==0


def test_custom_finalize_requires_component_weight_support():
    @dataclass
    class Incompatible(System):nystrom_compatible=False
    with pytest.raises(TypeError,match='nystrom_compatible'):
        Nystrom()(Incompatible(system().state),.1,manufactured('DPRKN6'))


def test_detached_velocity_dependence_cannot_pass_a_zero_jvp_probe():
    from warpSPHIntegrators import classify_separable
    def rhs(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x-.2*s.u.detach(),torch.zeros_like(s.e)),None
    assert not classify_separable(rhs,initial()).separable
    with pytest.raises(ValueError,match='independent acceleration'):
        Nystrom('DPRKN6')(initial(),.1,rhs)


def test_nystrom_composes_with_one_outer_finalize_and_second_order_damped_behavior():
    from test_operator_splitting import damped_split,error
    method=Strang(Nystrom('Nystrom4VelocityIndependent'),'RK4')
    errors=[error(method,h) for h in [.1,.05,.025]]
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.1)
    start=system()
    result=solve(method,start,.2,damped_split(),dt=.05)
    assert result.successful and result.state.initialized==result.state.finalized==4
    assert start.initialized==start.finalized==0


@pytest.mark.parametrize('method',[StabilizedRK('rkg2',5),Collocation('radau',3)])
def test_other_bound_one_step_factories_share_the_composition_contract(method):
    from test_operator_splitting import damped_split
    result=solve(Strang('RK4',method),testing.damped_problem().initial(),.1,damped_split(),dt=.05,
        solver_opts=dict(matvec='jvp',tol=1e-13,newton_tol=1e-8))
    assert result.successful


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_mechanical_vectors_and_scalar_quantities_keep_shapes_and_dtype(dtype):
    start=system();start.state.x=torch.ones((2,3),dtype=dtype)
    start.state.u=torch.zeros((2,3),dtype=dtype);start.state.e=torch.ones(2,dtype=dtype)
    # The plain system avoids this fixture's deliberately vector-shaped density.
    start=testing.ParticleSystem(start.state)
    def rhs(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x+.1*s.e[:,None],.2*s.u.square().sum(-1)),None
    result=solve(Nystrom('FineRKN5'),start,.1,rhs,dt=.01)
    assert result.successful and result.state.state.x.shape==(2,3) and result.state.state.e.shape==(2,)
    assert result.state.state.x.dtype==result.state.state.e.dtype==dtype


@pytest.mark.parametrize('options',[dict(method='bad'),dict(structure='bad'),
    dict(structure_atol=-1),dict(structure_atol=float('nan')),dict(structure_atol=True)])
def test_invalid_nystrom_configuration(options):
    with pytest.raises(ValueError):Nystrom(**options)
