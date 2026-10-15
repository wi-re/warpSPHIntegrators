"""Pinned source step maps, nonlinear order, stiff modes and restart contracts."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from dataclasses import dataclass

from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.adapters import IMEXRHS
from warpSPHIntegrators.state import StepHistory, HistoryEntry
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint
from warpSPHIntegrators.solvers import SolveResult, SolveDiagnostics
from warpSPHIntegrators.analysis.stability import rk_stability_function
from warpSPHIntegrators.state import BaseState, integrated, copied, constant, reference_state
import warpSPHIntegrators.testing as testing

ORACLE = json.loads((Path(__file__).parents[1] /
    'scripts/coefficients/wave10/implicit_extensions.json').read_text())
OPTIONS = dict(matvec='jvp', tol=1e-13, newton_tol=1e-8)


def parts(forced):
    def implicit(t,y):
        return torch.stack((-2*y[0]+.1*y[1]**2+(.2*math.sin(t) if forced else 0),
            -.5*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    def explicit(t,y):
        return torch.stack((.3*y[1]+(.05*math.sin(t) if forced else 0),
            -.2*y[0]**2+(.03*math.cos(t) if forced else 0)))
    return implicit,explicit


@pytest.mark.parametrize('forced',[False,True])
def test_mebdf2_matches_unchanged_source_step_body(forced):
    record=ORACLE['records']['mebdf2_'+('forced' if forced else 'autonomous')]
    implicit,_=parts(forced)
    state=TensorState(torch.tensor(record['initial'],dtype=torch.float64),t=record['t'])
    result=get_scheme('mebdf2')(state,record['h'],tensor_rhs(implicit),solver_opts=OPTIONS)
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(record['state'],rel=0,abs=2e-13)
    assert record['solve_times']==pytest.approx([state.t+.1,state.t+.2,state.t+.1])


@pytest.mark.parametrize('forced',[False,True])
@pytest.mark.parametrize('warm',[False,True])
def test_cnlf2_matches_unchanged_source_step_body(forced,warm):
    key='cnlf2_'+('warm' if warm else 'startup')+'_'+('forced' if forced else 'autonomous')
    record=ORACLE['records'][key];implicit,explicit=parts(forced)
    state=TensorState(torch.tensor(record['initial'],dtype=torch.float64),t=record['t'])
    history=None
    if warm:
        previous=TensorState(torch.tensor(record['previous'],dtype=torch.float64),t=state.t-record['h'])
        update,_=tensor_rhs(implicit)(previous,record['h'])
        history=StepHistory(1,[HistoryEntry(previous.t,record['h'],update,None,previous)],
            context=dict(method='cnlf2.v1',layout=(('y',(2,),'torch.float64'),),uid=None))
    result=get_scheme('cnlf2')(state,record['h'],
        IMEXRHS(explicit=tensor_rhs(explicit),implicit=tensor_rhs(implicit)),
        history=history,solver_opts=OPTIONS)
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(record['state'],rel=0,abs=2e-13)
    assert torch.equal(result.history.latest.update.dy,implicit(state.t,state.y))


def test_mebdf2_stability_function_and_beyond_endpoint_stage():
    tab=get_scheme('mebdf2').function.dirkTableau
    assert np.allclose(tab.c,[1,2,1])
    for z in [-1000.,-1.,-.2+2j,3j]:
        assert rk_stability_function(tab,z)==pytest.approx((1-2*z+.5*z*z)/(1-z)**3)
        assert abs(rk_stability_function(tab,z))<=1+1e-14


@pytest.mark.parametrize('name',['mebdf2','cnlf2'])
def test_nonlinear_nonautonomous_second_order(name):
    errors=[]
    for h in [.1,.05,.025]:
        result=solve(name,TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=h,solver_opts=OPTIONS)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-1)))
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,rel=0,abs=.15)


def test_cnlf2_split_damped_rotation_order():
    errors=[]
    rhs=IMEXRHS(implicit=tensor_rhs(lambda t,y:-.4*y),
        explicit=tensor_rhs(lambda t,y:torch.stack((-2*y[1],2*y[0]))))
    target=math.exp(-.4)*torch.tensor([math.cos(2),math.sin(2)],dtype=torch.float64)
    for h in [.05,.025,.0125]:
        result=solve('cnlf2',TensorState(torch.tensor([1.,0.],dtype=torch.float64)),1.,rhs,dt=h,solver_opts=OPTIONS)
        errors.append(torch.linalg.vector_norm(result.state.y-target).item())
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,rel=0,abs=.1)


def test_cnlf2_warm_oscillator_keeps_two_parity_energy_levels():
    rhs=tensor_rhs(lambda t,y:torch.stack((2*y[1],-2*y[0])))
    levels=[]
    result=solve('cnlf2',TensorState(torch.tensor([1.,0.],dtype=torch.float64)),4.,rhs,dt=.1,
        solver_opts=OPTIONS,callback=lambda step:levels.append(float(step.result.state.y.square().sum())))
    assert result.successful
    assert levels[::2]==pytest.approx([1/1.04]*20,rel=0,abs=2e-12)
    assert levels[1::2]==pytest.approx([1.]*20,rel=0,abs=2e-12)


def test_cnlf2_parasitic_mode_is_not_stiff_decay():
    result=solve('cnlf2',TensorState(torch.ones(1,dtype=torch.float64)),.2,
        tensor_rhs(lambda t,y:-1000*y),dt=.1,solver_opts=OPTIONS)
    assert result.successful and result.state.y.item()==pytest.approx(-99/101,rel=0,abs=1e-11)
    for mu in [.1,.8]:assert np.allclose(abs(np.roots([1,-2j*mu,-1])),1)
    assert max(abs(np.roots([1,.4,-1])))>1.1


@pytest.mark.parametrize('name',['mebdf2','cnlf2'])
def test_new_implicit_defaults_match_linear_and_scaled_newton_tolerances(name):
    result=solve(name,TensorState(torch.ones(1,dtype=torch.float64)),.2,
        tensor_rhs(lambda t,y:-y),dt=.025,solver=None)
    assert result.successful and result.state.y.item()==pytest.approx(math.exp(-.2),rel=0,abs=1e-4)


class FailedSolver:
    def solve(self,step,y0,norm,**kwargs):
        return SolveResult(y0,False,0,SolveDiagnostics(converged=False,termination='max_iterations'))


def test_cnlf2_failure_rollback_changed_dt_and_portable_restart(tmp_path):
    initial=TensorState(torch.ones(1,dtype=torch.float64));rhs=tensor_rhs(lambda t,y:-y)
    first=solve('cnlf2',initial,.2,rhs,dt=.05,solver_opts=OPTIONS)
    path=tmp_path/'cnlf.pt';save_checkpoint(first.checkpoint,path)
    resumed=solve('cnlf2',None,.4,rhs,restart=load_checkpoint(path),solver_opts=OPTIONS)
    whole=solve('cnlf2',initial,.4,rhs,dt=.05,solver_opts=OPTIONS)
    assert torch.allclose(resumed.state.y,whole.state.y,atol=1e-14)
    method=get_scheme('cnlf2')
    failed=method(first.state,.05,rhs,history=first.history,solver=FailedSolver())
    assert not failed.step_status.successful
    assert torch.equal(failed.history.latest.state.y,first.history.latest.state.y)
    changed=method(first.state,.03,rhs,history=first.history,solver_opts=OPTIONS)
    cold=method(first.state,.03,rhs,solver_opts=OPTIONS)
    assert torch.equal(changed.state.y,cold.state.y)
    assert initial.y.item()==1 and initial.t==0


@pytest.mark.parametrize('name',['mebdf2','cnlf2'])
def test_implicit_extension_parameter_and_initial_gradients(name):
    def forward(value,tracking=False):
        y=torch.ones(1,dtype=torch.float64,requires_grad=tracking)
        k=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
        result=solve(name,TensorState(y),.4,tensor_rhs(lambda t,y:-k*y-.1*y**3),dt=.05,solver_opts=OPTIONS)
        return result.state.y.sum(),y,k
    value,y,k=forward(2.,True);dy,dk=torch.autograd.grad(value,(y,k));eps=1e-5
    finite=(forward(2+eps)[0]-forward(2-eps)[0])/(2*eps)
    assert dy.item()>0
    assert dk.item()==pytest.approx(finite.item(),rel=1e-6,abs=1e-9)


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
    def initializeNewState(self,*args,**kwargs):return System(self.state.initializeNewState(),self.t)
    def preprocess(self,initial,dt,**kwargs):self.state.density=self.state.x.square();return self


def test_cnlf2_full_quantities_copied_fields_and_particle_identity_reset():
    initial=System(Fields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))
    def force(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x,torch.ones_like(s.e)),None
    result=solve('cnlf2',initial,.3,force,dt=.05,solver_opts=OPTIONS)
    assert result.successful
    assert torch.allclose(result.state.state.e,torch.full((2,),.3,dtype=torch.float64))
    assert torch.equal(result.state.state.density,result.state.state.x.square())
    assert torch.equal(initial.state.density,torch.zeros(2,dtype=torch.float64))
    result.state.state.uid=result.state.state.uid.flip(0)
    method=get_scheme('cnlf2')
    changed=method(result.state,.05,force,history=result.history,solver_opts=OPTIONS)
    cold=method(result.state,.05,force,solver_opts=OPTIONS)
    assert torch.equal(changed.state.state.x,cold.state.state.x)
    assert torch.equal(result.history.context['uid'],torch.tensor([1,2]))
