"""Generalized-alpha: independent modes, state memory, lifecycle and gradients."""
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

from warpSPHIntegrators import (GeneralizedAlpha, StepHistory, SolveResult, SolveDiagnostics,
                               get_reference_state, solve, snapshot_state, BaseState,
                               integrated, constant, copied, reference_state)
from warpSPHIntegrators import testing

spec=importlib.util.spec_from_file_location('alpha_demo',Path(__file__).resolve().parents[1]/'examples/structural_modes.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)
OPTIONS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7)


def system(q=None,v=None):
    q=torch.ones(1,dtype=torch.float64) if q is None else q
    v=q*0 if v is None else v
    return testing.ParticleSystem(testing.ParticleState(q,v,q*0,torch.ones_like(q)))


def rhs(state,dt,**kwargs):
    s=get_reference_state(state)
    return testing.ParticleUpdate(s.u,-4*s.x-.2*s.u,s.e*0),None


@pytest.mark.parametrize('rho',[0.,.5,.8,1.])
def test_driver_matches_independent_dense_trajectory_and_acceleration(rho):
    model=demo.two_mass_problem();h=.1;end=1.
    actual,result=demo.generalized_alpha_trajectory(model,h,end,rho)
    expected=demo.trajectory(model,h,end,rho)
    assert len(actual)==len(expected)
    assert max(float((a.q-b.q).abs().max()) for a,b in zip(actual,expected))<2e-9
    assert max(float((a.v-b.v).abs().max()) for a,b in zip(actual,expected))<2e-8
    assert torch.allclose(result.history.latest.state.acceleration,expected[-1].a,atol=2e-7,rtol=1e-8)


@pytest.mark.parametrize('rho',[0.,.5,1.])
def test_resolved_order_and_selective_damping_use_production_driver(rho):
    model=demo.two_mass_problem(slow=1.,fast=3.);errors=[]
    for h in (.05,.025,.0125):
        values,_=demo.generalized_alpha_trajectory(model,h,1.,rho)
        q,v=model.exact(1.);errors.append(float(torch.linalg.vector_norm(values[-1].q-q)+torch.linalg.vector_norm(values[-1].v-v)))
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.08)
    if rho==.5:
        values,_=demo.generalized_alpha_trajectory(demo.two_mass_problem(),.1,4.,rho)
        energy=demo.two_mass_problem().modal_energy(values[-1])/demo.two_mass_problem().modal_energy(values[0])
        assert energy[0]>.999 and energy[1]<1e-5


def test_shortened_steps_and_in_memory_restart_keep_algorithmic_memory():
    method=GeneralizedAlpha(.5);initial=system()
    whole=solve(method,initial,.35,rhs,dt=.1,solver_opts=OPTIONS)
    first=solve(method,initial,.2,rhs,dt=.1,solver_opts=OPTIONS)
    restarted=solve(method,None,.35,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
    assert torch.allclose(whole.state.state.x,restarted.state.state.x,atol=1e-12)
    assert torch.allclose(whole.history.latest.state.acceleration,restarted.history.latest.state.acceleration,atol=1e-12)
    state=initial;history=None
    for h in (.1,.1,.1,.05):
        result=method(state,h,rhs,history=history,solver_opts=OPTIONS)
        state,history=result.state,result.history
    assert torch.allclose(whole.state.state.x,state.state.x,atol=1e-12)
    assert whole.history.latest.dt==pytest.approx(.05)
    with pytest.raises(ValueError,match='scheme'):
        solve(GeneralizedAlpha(.8),None,.4,rhs,restart=first.checkpoint)


def test_long_fixed_grid_has_no_roundoff_remainder_step():
    method=GeneralizedAlpha(0.)
    actual=solve(method,system(),12.,rhs,dt=.1,solver_opts=OPTIONS)
    assert actual.accepted_steps==120 and actual.state.t==12.
    state=system();history=None
    for _ in range(120):
        result=method(state,.1,rhs,history=history,solver_opts=OPTIONS)
        state,history=result.state,result.history
    assert torch.allclose(actual.history.latest.state.acceleration,history.latest.state.acceleration,atol=1e-10)


def test_mismatched_history_is_rejected_and_caller_owns_old_values():
    initial=system();saved=snapshot_state(initial);method=GeneralizedAlpha(.5)
    first=method(initial,.1,rhs,solver_opts=OPTIONS)
    assert torch.equal(initial.state.x,saved.state.x) and initial.t==0
    remembered=first.history.latest.state.acceleration.clone()
    second=method(first.state,.07,rhs,history=first.history,solver_opts=OPTIONS)
    assert torch.equal(first.history.latest.state.acceleration,remembered)
    assert second.history.latest.state.t==pytest.approx(.17)
    for state,scheme in ((initial,method),(first.state,GeneralizedAlpha(.8)),(system(torch.ones(2,dtype=torch.float64)),method)):
        with pytest.raises(ValueError,match='history'):
            scheme(state,.1,rhs,history=first.history,solver_opts=OPTIONS)


@dataclass
class LifecycleSystem(testing.ParticleSystem):
    calls: list=field(default_factory=list)
    def initialize(self,dt,**kwargs):
        self.calls.append(('initialize',self.t));return self
    def initializeNewState(self,*args,**kwargs):
        return LifecycleSystem(self.state.initializeNewState(),self.t,self.calls)
    def finalize(self,*args,lastStageSystem=None,**kwargs):
        self.calls.append(('finalize',self.t));return self


class FailedSolver:
    def solve(self,step,y0,norm,**kwargs):
        return SolveResult(y0,False,0,SolveDiagnostics(termination='max_iterations'))


def test_failure_rolls_back_state_memory_and_lifecycle_then_can_resume():
    method=GeneralizedAlpha();s=system();initial=LifecycleSystem(s.state)
    first=solve(method,initial,.1,rhs,dt=.1,solver_opts=OPTIONS)
    failed=solve(method,None,.2,rhs,restart=first.checkpoint,solver=FailedSolver(),raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0 and failed.rejected_steps==1
    assert failed.state.t==.1 and failed.state.calls==first.state.calls
    assert initial.calls==[]
    assert torch.equal(failed.history.latest.state.acceleration,first.history.latest.state.acceleration)
    resumed=solve(method,None,.2,rhs,restart=failed.checkpoint,solver_opts=OPTIONS)
    assert resumed.successful and len(resumed.state.calls)==4


def test_forced_velocity_dependent_and_quantity_coupled_problem_is_second_order():
    # q=cos(t), v=-sin(t), e=t; e contributes to the force and must be
    # interpolated at the weighted time while its endpoint rate is solved.
    def force(state,dt,**kwargs):
        s=get_reference_state(state);t=state.t
        a=-4*s.x-s.x**3-.2*s.u+s.e+3*math.cos(t)+math.cos(t)**3-.2*math.sin(t)-t
        return testing.ParticleUpdate(s.u,a,torch.ones_like(s.e)),None
    errors=[]
    for h in (.1,.05,.025):
        result=solve(GeneralizedAlpha(.5),system(),1.,force,dt=h,solver_opts=OPTIONS)
        s=result.state.state
        assert s.e.item()==pytest.approx(1,abs=1e-12)
        errors.append(abs(s.x.item()-math.cos(1))+abs(s.u.item()+math.sin(1)))
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.1)


def test_initial_state_and_force_parameter_gradients_cross_memory_and_restart():
    def forward(k_value,track,restart=False):
        k=torch.tensor(k_value,dtype=torch.float64,requires_grad=track)
        q=torch.ones(1,dtype=torch.float64,requires_grad=track)
        def force(state,dt,**kwargs):
            s=get_reference_state(state)
            return testing.ParticleUpdate(s.u,-k*s.x-s.x**3,s.e*0),None
        method=GeneralizedAlpha(.5)
        first=solve(method,system(q),.2 if restart else .5,force,dt=.1,solver_opts=OPTIONS)
        result=solve(method,None,.5,force,restart=first.checkpoint,solver_opts=OPTIONS) if restart else first
        return result.state.state.x.sum()+result.state.state.u.sum(),q,k
    output,q,k=forward(4.,True,True);gq,gk=torch.autograd.grad(output,(q,k));eps=1e-5
    finite=(forward(4+eps,False)[0]-forward(4-eps,False)[0])/(2*eps)
    def vary_initial(value):
        state=system(torch.tensor([value],dtype=torch.float64))
        def force(state,dt,**kwargs):
            s=state.state
            return testing.ParticleUpdate(s.u,-4*s.x-s.x**3,s.e*0),None
        result=solve(GeneralizedAlpha(.5),state,.5,force,dt=.1,solver_opts=OPTIONS)
        return (result.state.state.x+result.state.state.u).item()
    finite_q=(vary_initial(1+eps)-vary_initial(1-eps))/(2*eps)
    assert gq.item()==pytest.approx(finite_q,rel=2e-5,abs=1e-8)
    assert gk.item()==pytest.approx(finite.item(),rel=2e-5,abs=1e-8)


@pytest.mark.parametrize('rho',[-.1,1.1,float('nan'),float('inf'),True])
def test_bad_parameters_are_rejected(rho):
    with pytest.raises(ValueError):GeneralizedAlpha(rho)


def test_runner_rejects_unsupported_adaptive_sampling_and_wrong_drift():
    with pytest.raises(ValueError,match='Adaptive'):
        solve(GeneralizedAlpha(),system(),1.,rhs,dt=.1,adaptive=True)
    with pytest.raises(ValueError,match='Off-grid'):
        solve(GeneralizedAlpha(),system(),1.,rhs,dt=.1,sample_times=(.15,))
    def wrong(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u+1,-s.x,s.e*0),None
    with pytest.raises(ValueError,match='position rate'):
        GeneralizedAlpha()(system(),.1,wrong)


@dataclass
class TaggedFields(BaseState):
    q: torch.Tensor=integrated('dxdt',tags=('position',))
    v: torch.Tensor=integrated('dudt',tags=('velocity',))
    thermal: torch.Tensor=integrated('dedt',tags=('quantity',))
    uid: torch.Tensor=constant()
    density: torch.Tensor=copied()


@dataclass
class TaggedSystem(testing.ParticleSystem):
    state: TaggedFields=reference_state()
    def initializeNewState(self,*args,**kwargs):
        return TaggedSystem(self.state.initializeNewState(),self.t)
    def preprocess(self,initial,dt,**kwargs):
        self.state.density=self.state.q.square()
        return self


def test_tags_copied_endpoint_fields_and_uid_continuity():
    initial=TaggedSystem(TaggedFields(torch.tensor([1.,2.],dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))
    def force(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.v,-s.q,torch.ones_like(s.thermal)),None
    method=GeneralizedAlpha();first=method(initial,.1,force,solver_opts=OPTIONS)
    assert torch.equal(initial.state.density,torch.zeros(2,dtype=torch.float64))
    assert torch.allclose(first.state.state.density,first.state.state.q.square(),atol=1e-14)
    assert torch.allclose(first.state.state.thermal,torch.full((2,),.1,dtype=torch.float64))
    second=method(first.state,.05,force,history=first.history,solver_opts=OPTIONS)
    assert second.step_status.tolerance_converged
    changed=snapshot_state(first.state);changed.state.uid=changed.state.uid.flip(0)
    with pytest.raises(ValueError,match='particle layout'):
        method(changed,.1,force,history=first.history)


def test_finalize_that_changes_integrated_state_cannot_leave_stale_memory():
    @dataclass
    class ModifyingSystem(LifecycleSystem):
        def initializeNewState(self,*args,**kwargs):
            return ModifyingSystem(self.state.initializeNewState(),self.t,self.calls)
        def finalize(self,*args,**kwargs):
            self.state.x=self.state.x+1
            return self
    s=system();initial=ModifyingSystem(s.state)
    with pytest.raises(ValueError,match='preserve integrated'):
        GeneralizedAlpha()(initial,.1,rhs,solver_opts=OPTIONS)
    assert initial.t==0 and initial.state.x.item()==1


def test_nonfinite_endpoint_rhs_does_not_finalize_or_commit_memory():
    def endpoint_bad(state,dt,**kwargs):
        s=state.state
        acceleration=torch.full_like(s.x,float('inf')) if state.t>=.1 else -s.x
        return testing.ParticleUpdate(s.u,acceleration,s.e*0),None
    s=system();initial=LifecycleSystem(s.state)
    result=GeneralizedAlpha()(initial,.1,endpoint_bad,solver_opts=OPTIONS)
    assert not result.step_status.successful and not result.history.entries
    assert all(call[0]!='finalize' for call in result.state.calls)


def test_default_solver_and_fixed_depth_solver_work_with_acceleration_memory():
    from warpSPHIntegrators import FixedPointSolver
    for options in ({},dict(solver=FixedPointSolver(),solver_opts={'iterations':8})):
        result=GeneralizedAlpha()(system(),.01,rhs,**options)
        assert result.step_status.successful and result.history.latest is not None
        assert result.state.state.x.item()==pytest.approx(.9998,abs=1e-6)


def test_finalize_receives_paired_endpoint_updates_and_quantity_weights():
    @dataclass
    class SignatureSystem(LifecycleSystem):
        def initializeNewState(self,*args,**kwargs):
            return SignatureSystem(self.state.initializeNewState(),self.t,self.calls)
        def finalize(self,initial,dt,returnValues,updateValues,weights,**kwargs):
            assert len(returnValues)==len(updateValues)==len(weights)==2
            assert weights==[.5,.5]
            expected=initial.state.e+.5*dt*(updateValues[0].dedt+updateValues[1].dedt)
            assert torch.allclose(self.state.e,expected,atol=1e-12)
            self.calls.append(('finalize',self.t))
            return self
    def quantity(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x,s.e+1),None
    s=system();initial=SignatureSystem(s.state)
    result=GeneralizedAlpha()(initial,.1,quantity,solver_opts=OPTIONS)
    assert result.step_status.tolerance_converged and len(result.stages)==2
    assert result.state.calls==[('initialize',0.),('finalize',.1)]
